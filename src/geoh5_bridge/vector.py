"""Convert GeoDataFrames to geoh5 Points, Curve and Surface objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoh5_bridge.utils import _add_data_columns

if TYPE_CHECKING:
    import geopandas as gpd
    from geoh5py.objects import Curve, Points, Surface
    from geoh5py.workspace import Workspace

# Decimal places used when rounding coordinates for vertex matching
# during polygon triangulation.
_COORD_PRECISION = 10


def _is_numeric_column(gdf: gpd.GeoDataFrame, col: str) -> bool:
    """Check if a GeoDataFrame column has a numeric dtype."""
    try:
        return np.issubdtype(gdf[col].dtype, np.number)
    except TypeError:
        return False


def geodataframe_to_points(
    gdf: gpd.GeoDataFrame,
    workspace: Workspace,
    *,
    name: str = "Points",
    data_columns: list[str] | None = None,
    z_column: str | None = None,
) -> Points:
    """Convert a GeoDataFrame with Point geometries to geoh5py Points.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame whose geometry column contains ``Point`` or
        ``MultiPoint`` geometries.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Points object.
    data_columns : list[str], optional
        Attribute columns to attach as data channels. If ``None``,
        all numeric columns are used.
    z_column : str, optional
        Column to use for the z-coordinate. When ``None``, the z
        value from the geometry is used (defaulting to 0 for 2D).

    Returns
    -------
    geoh5py.objects.Points

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import geodataframe_to_points
    >>> gdf = gpd.read_file("sample_points.geojson")
    >>> ws = Workspace.create("output.geoh5")
    >>> pts = geodataframe_to_points(gdf, ws, name="SamplePoints")
    """
    from geoh5py.objects import Points

    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values

    if z_column is not None:
        zs = gdf[z_column].values.astype(float)
    else:
        zs = np.array(
            [geom.z if geom.has_z else 0.0 for geom in gdf.geometry]
        )

    vertices = np.column_stack([xs, ys, zs])

    pts = Points.create(workspace, vertices=vertices, name=name)

    # Determine which columns to add
    if data_columns is None:
        data_columns = [
            col
            for col in gdf.columns
            if col != gdf.geometry.name and _is_numeric_column(gdf, col)
        ]

    data = {}
    for col in data_columns:
        if col in gdf.columns:
            data[col] = gdf[col].values.astype(float)

    if data:
        _add_data_columns(pts, data)

    return pts


def geodataframe_to_curve(
    gdf: gpd.GeoDataFrame,
    workspace: Workspace,
    *,
    name: str = "Curve",
    data_columns: list[str] | None = None,
) -> Curve:
    """Convert a GeoDataFrame with LineString geometries to a geoh5py Curve.

    All LineString features are concatenated into a single Curve object.
    The ``parts`` attribute records which vertices belong to which
    original feature, enabling per-feature data attachment.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with ``LineString`` or ``MultiLineString``
        geometries.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Curve object.
    data_columns : list[str], optional
        Attribute columns to attach. Only constant-per-feature (not
        per-vertex) data is supported.

    Returns
    -------
    geoh5py.objects.Curve

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import geodataframe_to_curve
    >>> gdf = gpd.read_file("roads.geojson")
    >>> ws = Workspace.create("output.geoh5")
    >>> curve = geodataframe_to_curve(gdf, ws, name="Roads")
    """
    from geoh5py.objects import Curve
    from shapely.geometry import LineString, MultiLineString

    all_vertices: list[np.ndarray] = []
    all_cells: list[np.ndarray] = []
    parts: list[int] = []
    vertex_offset = 0

    for idx, geom in enumerate(gdf.geometry):
        if isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        elif isinstance(geom, LineString):
            lines = [geom]
        else:
            continue

        for line in lines:
            coords = np.array(line.coords)
            if coords.shape[1] == 2:
                coords = np.column_stack(
                    [coords, np.zeros(len(coords))]
                )
            all_vertices.append(coords)

            n = len(coords)
            cells = np.column_stack(
                [
                    np.arange(vertex_offset, vertex_offset + n - 1),
                    np.arange(vertex_offset + 1, vertex_offset + n),
                ]
            )
            all_cells.append(cells)
            parts.extend([idx] * n)
            vertex_offset += n

    if not all_vertices:
        raise ValueError("No LineString geometries found in GeoDataFrame.")

    vertices = np.vstack(all_vertices)
    cells = np.vstack(all_cells).astype(np.uint32)

    curve = Curve.create(
        workspace, vertices=vertices, cells=cells, name=name
    )

    # Attach per-vertex data by repeating feature attributes
    if data_columns is None:
        data_columns = [
            col
            for col in gdf.columns
            if col != gdf.geometry.name and _is_numeric_column(gdf, col)
        ]

    parts_arr = np.array(parts, dtype=int)
    data = {}
    for col in data_columns:
        if col in gdf.columns:
            feature_vals = gdf[col].values.astype(float)
            per_vertex = feature_vals[parts_arr]
            data[col] = per_vertex

    if data:
        _add_data_columns(curve, data)

    return curve


def _triangulate_polygon(polygon) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a single Polygon using Delaunay triangulation.

    Returns vertices (N, 2-or-3) and triangle cells (M, 3) as arrays.
    Only triangles whose representative point falls inside the polygon
    are kept, so concave polygons are handled correctly.
    """
    from shapely.geometry import MultiPoint
    from shapely.ops import triangulate

    coords = np.array(polygon.exterior.coords[:-1])

    for interior in polygon.interiors:
        hole_coords = np.array(interior.coords[:-1])
        coords = np.vstack([coords, hole_coords])

    # Build a coordinate-to-index lookup (2D only for matching)
    coord_to_idx: dict[tuple[float, float], int] = {}
    for i in range(len(coords)):
        key = (round(float(coords[i, 0]), _COORD_PRECISION),
               round(float(coords[i, 1]), _COORD_PRECISION))
        coord_to_idx[key] = i

    triangles = triangulate(MultiPoint(coords.tolist()))

    cells: list[list[int]] = []
    for tri in triangles:
        if not polygon.contains(tri.representative_point()):
            continue
        tri_coords = np.array(tri.exterior.coords[:-1])
        indices: list[int] = []
        for tc in tri_coords:
            key = (round(float(tc[0]), _COORD_PRECISION),
                   round(float(tc[1]), _COORD_PRECISION))
            if key in coord_to_idx:
                indices.append(coord_to_idx[key])
            else:
                break
        if len(indices) == 3:
            cells.append(indices)

    if cells:
        return coords, np.array(cells, dtype=np.uint32)
    return coords, np.empty((0, 3), dtype=np.uint32)


def geodataframe_to_surface(
    gdf: gpd.GeoDataFrame,
    workspace: Workspace,
    *,
    name: str = "Surface",
    data_columns: list[str] | None = None,
) -> Surface:
    """Convert a GeoDataFrame with Polygon geometries to a geoh5py Surface.

    Each polygon is triangulated using Delaunay triangulation and all
    features are combined into a single triangulated Surface mesh.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with ``Polygon`` or ``MultiPolygon`` geometries.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Surface object.
    data_columns : list[str], optional
        Attribute columns to attach as per-vertex data. If ``None``,
        all numeric columns are used.

    Returns
    -------
    geoh5py.objects.Surface

    Examples
    --------
    >>> import geopandas as gpd
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import geodataframe_to_surface
    >>> gdf = gpd.read_file("parcels.geojson")
    >>> ws = Workspace.create("output.geoh5")
    >>> surf = geodataframe_to_surface(gdf, ws, name="Parcels")
    """
    from geoh5py.objects import Surface
    from shapely.geometry import MultiPolygon, Polygon

    all_vertices: list[np.ndarray] = []
    all_cells: list[np.ndarray] = []
    parts: list[int] = []
    vertex_offset = 0

    for idx, geom in enumerate(gdf.geometry):
        if isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        elif isinstance(geom, Polygon):
            polygons = [geom]
        else:
            continue

        for poly in polygons:
            coords, tri_cells = _triangulate_polygon(poly)
            if len(tri_cells) == 0:
                continue

            # Ensure 3D coords
            if coords.shape[1] == 2:
                coords = np.column_stack(
                    [coords, np.zeros(len(coords))]
                )

            all_vertices.append(coords)
            all_cells.append(tri_cells + vertex_offset)
            parts.extend([idx] * len(coords))
            vertex_offset += len(coords)

    if not all_vertices:
        raise ValueError("No Polygon geometries found in GeoDataFrame.")

    vertices = np.vstack(all_vertices)
    cells = np.vstack(all_cells).astype(np.uint32)

    surface = Surface.create(
        workspace, vertices=vertices, cells=cells, name=name
    )

    # Attach per-vertex data by repeating feature attributes
    if data_columns is None:
        data_columns = [
            col
            for col in gdf.columns
            if col != gdf.geometry.name and _is_numeric_column(gdf, col)
        ]

    parts_arr = np.array(parts, dtype=int)
    data = {}
    for col in data_columns:
        if col in gdf.columns:
            feature_vals = gdf[col].values.astype(float)
            per_vertex = feature_vals[parts_arr]
            data[col] = per_vertex

    if data:
        _add_data_columns(surface, data)

    return surface


def points_to_geodataframe(
    pts: Points,
    *,
    data_names: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Convert a geoh5py Points object to a GeoDataFrame.

    Parameters
    ----------
    pts : geoh5py.objects.Points
        Points object with vertices and optional data children.
    data_names : list[str], optional
        Names of data children to include as columns.  If ``None``,
        all data children with a ``values`` attribute are included.

    Returns
    -------
    geopandas.GeoDataFrame

    Examples
    --------
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import points_to_geodataframe
    >>> ws = Workspace("input.geoh5")
    >>> pts = ws.get_entity("SamplePoints")[0]
    >>> gdf = points_to_geodataframe(pts)
    """
    import geopandas as gpd
    from shapely.geometry import Point

    vertices = pts.vertices
    geometry = [Point(v[0], v[1], v[2]) for v in vertices]

    children = {
        c.name: c.values
        for c in pts.children
        if hasattr(c, "values")
    }
    if data_names is not None:
        children = {
            k: v for k, v in children.items() if k in data_names
        }

    data = {
        name: np.asarray(values, dtype=float)
        for name, values in children.items()
    }

    return gpd.GeoDataFrame(data, geometry=geometry)


def curve_to_geodataframe(
    curve: Curve,
    *,
    data_names: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Convert a geoh5py Curve object to a GeoDataFrame.

    Each connected polyline in the Curve becomes a single
    ``LineString`` feature in the resulting GeoDataFrame.
    Per-vertex data is reduced to per-feature by taking the value
    at the first vertex of each polyline.

    Parameters
    ----------
    curve : geoh5py.objects.Curve
        Curve object with vertices, cells and optional data children.
    data_names : list[str], optional
        Names of data children to include as columns.  If ``None``,
        all data children with a ``values`` attribute are included.

    Returns
    -------
    geopandas.GeoDataFrame

    Examples
    --------
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import curve_to_geodataframe
    >>> ws = Workspace("input.geoh5")
    >>> curve = ws.get_entity("Roads")[0]
    >>> gdf = curve_to_geodataframe(curve)
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    vertices = curve.vertices
    cells = curve.cells

    if cells is None or len(cells) == 0:
        return gpd.GeoDataFrame(geometry=[])

    # Reconstruct polylines from edge cells
    lines: list[list[int]] = []
    current_line = [int(cells[0][0]), int(cells[0][1])]
    for i in range(1, len(cells)):
        if cells[i][0] == cells[i - 1][1]:
            current_line.append(int(cells[i][1]))
        else:
            lines.append(current_line)
            current_line = [int(cells[i][0]), int(cells[i][1])]
    lines.append(current_line)

    geometry = [LineString(vertices[indices]) for indices in lines]

    # Collect per-vertex data children
    children = {
        c.name: c.values
        for c in curve.children
        if hasattr(c, "values")
    }
    if data_names is not None:
        children = {
            k: v for k, v in children.items() if k in data_names
        }

    # Reduce to per-feature using the first vertex of each line
    data: dict[str, list[float]] = {}
    for name, values in children.items():
        vals = np.asarray(values, dtype=float)
        data[name] = [float(vals[indices[0]]) for indices in lines]

    return gpd.GeoDataFrame(data, geometry=geometry)


def surface_to_geodataframe(
    surface: Surface,
    *,
    data_names: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Convert a geoh5py Surface object to a GeoDataFrame.

    Connected groups of triangles are merged into Polygon features
    using :func:`shapely.ops.unary_union`.  Per-vertex data is
    averaged over the vertices of each connected component.

    Parameters
    ----------
    surface : geoh5py.objects.Surface
        Surface object with vertices, triangle cells and optional
        data children.
    data_names : list[str], optional
        Names of data children to include as columns.  If ``None``,
        all data children with a ``values`` attribute are included.

    Returns
    -------
    geopandas.GeoDataFrame

    Examples
    --------
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import surface_to_geodataframe
    >>> ws = Workspace("input.geoh5")
    >>> surface = ws.get_entity("Parcels")[0]
    >>> gdf = surface_to_geodataframe(surface)
    """
    from collections import defaultdict

    import geopandas as gpd
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    vertices = surface.vertices
    cells = surface.cells

    # Build triangle adjacency via shared edges
    vertex_to_tris: dict[int, set[int]] = defaultdict(set)
    for i, tri in enumerate(cells):
        for v in tri:
            vertex_to_tris[int(v)].add(i)

    tri_adj: dict[int, set[int]] = defaultdict(set)
    for i, tri in enumerate(cells):
        edges = [
            (int(tri[0]), int(tri[1])),
            (int(tri[1]), int(tri[2])),
            (int(tri[0]), int(tri[2])),
        ]
        for e in edges:
            edge_key = (min(e), max(e))
            neighbours = (
                vertex_to_tris[edge_key[0]]
                & vertex_to_tris[edge_key[1]]
            )
            for j in neighbours:
                if j != i:
                    tri_adj[i].add(j)

    # BFS to find connected components of triangles
    visited: set[int] = set()
    components: list[set[int]] = []
    for i in range(len(cells)):
        if i in visited:
            continue
        component: set[int] = set()
        queue = [i]
        while queue:
            t = queue.pop()
            if t in visited:
                continue
            visited.add(t)
            component.add(t)
            queue.extend(tri_adj[t] - visited)
        components.append(component)

    # Merge triangles per component into polygons
    geometry = []
    component_vertex_sets: list[set[int]] = []
    for component in components:
        triangles = []
        vert_set: set[int] = set()
        for tri_idx in component:
            tri = cells[tri_idx]
            coords_2d = vertices[tri, :2]
            triangles.append(Polygon(coords_2d))
            for v in tri:
                vert_set.add(int(v))
        geometry.append(unary_union(triangles))
        component_vertex_sets.append(vert_set)

    # Collect data children
    children = {
        c.name: c.values
        for c in surface.children
        if hasattr(c, "values")
    }
    if data_names is not None:
        children = {
            k: v for k, v in children.items() if k in data_names
        }

    # Average per-vertex data over each component
    data: dict[str, list[float]] = {}
    for name, values in children.items():
        vals = np.asarray(values, dtype=float)
        feature_vals: list[float] = []
        for vert_set in component_vertex_sets:
            indices = list(vert_set)
            feature_vals.append(float(np.nanmean(vals[indices])))
        data[name] = feature_vals

    return gpd.GeoDataFrame(data, geometry=geometry)
