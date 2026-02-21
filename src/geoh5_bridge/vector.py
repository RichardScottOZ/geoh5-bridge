"""Convert GeoDataFrames to geoh5 Points, Curve and Surface objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoh5_bridge.utils import _add_data_columns

if TYPE_CHECKING:
    import geopandas as gpd
    from geoh5py.objects import Curve, Points, Surface
    from geoh5py.workspace import Workspace


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
        key = (round(float(coords[i, 0]), 10), round(float(coords[i, 1]), 10))
        coord_to_idx[key] = i

    triangles = triangulate(MultiPoint(coords.tolist()))

    cells: list[list[int]] = []
    for tri in triangles:
        if not polygon.contains(tri.representative_point()):
            continue
        tri_coords = np.array(tri.exterior.coords[:-1])
        indices: list[int] = []
        for tc in tri_coords:
            key = (round(float(tc[0]), 10), round(float(tc[1]), 10))
            if key in coord_to_idx:
                indices.append(coord_to_idx[key])
            else:
                break
        if len(indices) == 3:
            cells.append(indices)

    return coords, np.array(cells, dtype=np.uint32) if cells else np.empty((0, 3), dtype=np.uint32)


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
