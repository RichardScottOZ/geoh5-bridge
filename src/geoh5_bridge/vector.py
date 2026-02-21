"""Convert GeoDataFrames to geoh5 Points and Curve objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoh5_bridge.utils import _add_data_columns

if TYPE_CHECKING:
    import geopandas as gpd
    from geoh5py.objects import Curve, Points
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
    elif gdf.geometry.has_z.any():
        zs = np.array([geom.z if geom.has_z else 0.0 for geom in gdf.geometry])
    else:
        zs = np.zeros(len(gdf))

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
