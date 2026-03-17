"""Bidirectional conversions between geoh5py objects and PyVista meshes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoh5_bridge.utils import _add_data_columns

if TYPE_CHECKING:
    import pyvista as pv
    from geoh5py.objects import BlockModel, Curve, Grid2D, Points, Surface
    from geoh5py.workspace import Workspace


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_data_children(geoh5_object) -> dict[str, np.ndarray]:
    """Return a dict of {name: values} for all data children."""
    return {
        c.name: np.asarray(c.values, dtype=float)
        for c in geoh5_object.children
        if hasattr(c, "values")
    }


# ------------------------------------------------------------------
# Points ↔ PyVista
# ------------------------------------------------------------------


def points_to_pyvista(
    pts: Points,
    *,
    data_names: list[str] | None = None,
) -> pv.PolyData:
    """Convert a geoh5py Points object to a PyVista PolyData.

    Parameters
    ----------
    pts : geoh5py.objects.Points
        Points object with vertices and optional data children.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    pyvista.PolyData
    """
    import pyvista as pv

    polydata = pv.PolyData(np.asarray(pts.vertices, dtype=float))

    children = _get_data_children(pts)
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    for name, values in children.items():
        polydata.point_data[name] = values

    return polydata


def pyvista_to_points(
    polydata: pv.PolyData,
    workspace: Workspace,
    *,
    name: str = "Points",
    data_names: list[str] | None = None,
) -> Points:
    """Convert a PyVista PolyData (point cloud) to geoh5py Points.

    Parameters
    ----------
    polydata : pyvista.PolyData
        PolyData containing point coordinates (and optional point data).
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Points object.
    data_names : list[str], optional
        Point data arrays to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.Points
    """
    from geoh5py.objects import Points

    vertices = np.asarray(polydata.points, dtype=float)
    pts = Points.create(workspace, vertices=vertices, name=name)

    keys = list(polydata.point_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    data = {k: np.asarray(polydata.point_data[k], dtype=float) for k in keys}
    if data:
        _add_data_columns(pts, data)

    return pts


# ------------------------------------------------------------------
# Grid2D ↔ PyVista
# ------------------------------------------------------------------


def grid2d_to_pyvista(
    grid: Grid2D,
    *,
    data_names: list[str] | None = None,
) -> pv.StructuredGrid:
    """Convert a geoh5py Grid2D to a PyVista StructuredGrid.

    Parameters
    ----------
    grid : geoh5py.objects.Grid2D
        Grid2D object with data channels.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    pyvista.StructuredGrid
    """
    import pyvista as pv

    nx = grid.u_count
    ny = grid.v_count
    origin_x = float(grid.origin["x"])
    origin_y = float(grid.origin["y"])
    origin_z = float(grid.origin["z"])
    dx = float(grid.u_cell_size)
    dy = float(grid.v_cell_size)

    # Node coordinates (nx+1 × ny+1)
    x_nodes = origin_x + np.arange(nx + 1) * dx
    y_nodes = origin_y + np.arange(ny + 1) * dy

    xx, yy = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    zz = np.full_like(xx, origin_z)

    sg = pv.StructuredGrid(xx, yy, zz)

    children = _get_data_children(grid)
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    for ch_name, ch_vals in children.items():
        sg.cell_data[ch_name] = ch_vals

    return sg


def pyvista_to_grid2d(
    structured_grid: pv.StructuredGrid,
    workspace: Workspace,
    *,
    name: str = "Grid2D",
    data_names: list[str] | None = None,
) -> Grid2D:
    """Convert a PyVista StructuredGrid to a geoh5py Grid2D.

    The StructuredGrid is expected to be a flat 2D grid (one layer in
    the k-direction).

    Parameters
    ----------
    structured_grid : pyvista.StructuredGrid
        Flat 2D structured grid.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Grid2D object.
    data_names : list[str], optional
        Cell data arrays to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.Grid2D
    """
    from geoh5py.objects import Grid2D

    dims = structured_grid.dimensions  # (ni, nj, nk)
    # Node counts → cell counts
    nx = dims[0] - 1
    ny = dims[1] - 1

    pts = np.asarray(structured_grid.points, dtype=float)
    origin_x = float(pts[:, 0].min())
    origin_y = float(pts[:, 1].min())
    origin_z = float(pts[:, 2].min())

    # Compute cell sizes from node spacing
    x_unique = np.sort(np.unique(np.round(pts[:, 0], 10)))
    y_unique = np.sort(np.unique(np.round(pts[:, 1], 10)))
    dx = float(np.diff(x_unique).mean()) if len(x_unique) > 1 else 1.0
    dy = float(np.diff(y_unique).mean()) if len(y_unique) > 1 else 1.0

    grid = Grid2D.create(
        workspace,
        origin=[origin_x, origin_y, origin_z],
        u_cell_size=dx,
        v_cell_size=dy,
        u_count=nx,
        v_count=ny,
        name=name,
    )

    keys = list(structured_grid.cell_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    data = {
        k: np.asarray(structured_grid.cell_data[k], dtype=float) for k in keys
    }
    if data:
        _add_data_columns(grid, data)

    return grid


# ------------------------------------------------------------------
# Curve ↔ PyVista
# ------------------------------------------------------------------


def curve_to_pyvista(
    curve: Curve,
    *,
    data_names: list[str] | None = None,
) -> pv.PolyData:
    """Convert a geoh5py Curve to a PyVista PolyData with lines.

    Parameters
    ----------
    curve : geoh5py.objects.Curve
        Curve object with vertices and edge cells.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    pyvista.PolyData
    """
    import pyvista as pv

    vertices = np.asarray(curve.vertices, dtype=float)
    cells = curve.cells

    # Reconstruct polylines from edge cells (same as curve_to_geodataframe)
    lines_idx: list[list[int]] = []
    current_line = [int(cells[0][0]), int(cells[0][1])]
    for i in range(1, len(cells)):
        if cells[i][0] == cells[i - 1][1]:
            current_line.append(int(cells[i][1]))
        else:
            lines_idx.append(current_line)
            current_line = [int(cells[i][0]), int(cells[i][1])]
    lines_idx.append(current_line)

    # Build PyVista lines array: [n, idx0, idx1, ..., n, idx0, ...]
    pv_lines: list[int] = []
    for line in lines_idx:
        pv_lines.append(len(line))
        pv_lines.extend(line)

    polydata = pv.PolyData(vertices, lines=np.array(pv_lines))

    children = _get_data_children(curve)
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    for ch_name, ch_vals in children.items():
        polydata.point_data[ch_name] = ch_vals

    return polydata


def pyvista_to_curve(
    polydata: pv.PolyData,
    workspace: Workspace,
    *,
    name: str = "Curve",
    data_names: list[str] | None = None,
) -> Curve:
    """Convert a PyVista PolyData with lines to a geoh5py Curve.

    Parameters
    ----------
    polydata : pyvista.PolyData
        PolyData with line cells.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Curve object.
    data_names : list[str], optional
        Point data arrays to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.Curve
    """
    from geoh5py.objects import Curve

    vertices = np.asarray(polydata.points, dtype=float)

    # Parse PyVista lines array → edge cells
    lines_arr = np.asarray(polydata.lines)
    all_cells: list[list[int]] = []
    i = 0
    while i < len(lines_arr):
        n = lines_arr[i]
        indices = lines_arr[i + 1 : i + 1 + n].tolist()
        for j in range(len(indices) - 1):
            all_cells.append([indices[j], indices[j + 1]])
        i += 1 + n

    cells = np.array(all_cells, dtype=np.uint32)
    curve = Curve.create(
        workspace, vertices=vertices, cells=cells, name=name
    )

    keys = list(polydata.point_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    data = {k: np.asarray(polydata.point_data[k], dtype=float) for k in keys}
    if data:
        _add_data_columns(curve, data)

    return curve


# ------------------------------------------------------------------
# Surface ↔ PyVista
# ------------------------------------------------------------------


def surface_to_pyvista(
    surface: Surface,
    *,
    data_names: list[str] | None = None,
) -> pv.PolyData:
    """Convert a geoh5py Surface to a PyVista PolyData triangle mesh.

    Parameters
    ----------
    surface : geoh5py.objects.Surface
        Surface object with vertices and triangle cells.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    pyvista.PolyData
    """
    import pyvista as pv

    vertices = np.asarray(surface.vertices, dtype=float)
    cells = surface.cells  # (M, 3) triangle indices

    # PyVista face format: [3, v0, v1, v2, 3, v0, ...]
    n_faces = len(cells)
    pv_faces = np.column_stack(
        [np.full(n_faces, 3, dtype=np.uint32), cells]
    ).ravel()

    polydata = pv.PolyData(vertices, pv_faces)

    children = _get_data_children(surface)
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    for ch_name, ch_vals in children.items():
        polydata.point_data[ch_name] = ch_vals

    return polydata


def pyvista_to_surface(
    polydata: pv.PolyData,
    workspace: Workspace,
    *,
    name: str = "Surface",
    data_names: list[str] | None = None,
) -> Surface:
    """Convert a PyVista PolyData triangle mesh to a geoh5py Surface.

    The PolyData is expected to contain only triangular faces.

    Parameters
    ----------
    polydata : pyvista.PolyData
        Triangle mesh.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Surface object.
    data_names : list[str], optional
        Point data arrays to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.Surface

    Raises
    ------
    ValueError
        If the PolyData contains non-triangular faces.
    """
    from geoh5py.objects import Surface

    if not polydata.is_all_triangles:
        raise ValueError(
            "PolyData must contain only triangular faces. "
            "Use polydata.triangulate() first."
        )

    vertices = np.asarray(polydata.points, dtype=float)

    # Parse faces: [3, v0, v1, v2, 3, ...]
    faces = np.asarray(polydata.faces)
    n_faces = polydata.n_cells
    cells = faces.reshape(n_faces, 4)[:, 1:].astype(np.uint32)

    surface = Surface.create(
        workspace, vertices=vertices, cells=cells, name=name
    )

    keys = list(polydata.point_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    data = {k: np.asarray(polydata.point_data[k], dtype=float) for k in keys}
    if data:
        _add_data_columns(surface, data)

    return surface


# ------------------------------------------------------------------
# BlockModel ↔ PyVista
# ------------------------------------------------------------------


def blockmodel_to_pyvista(
    blockmodel: BlockModel,
    *,
    data_names: list[str] | None = None,
) -> pv.RectilinearGrid:
    """Convert a geoh5py BlockModel to a PyVista RectilinearGrid.

    Parameters
    ----------
    blockmodel : geoh5py.objects.BlockModel
        BlockModel object with delimiters and data children.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    pyvista.RectilinearGrid
    """
    import pyvista as pv

    origin_x = float(blockmodel.origin["x"])
    origin_y = float(blockmodel.origin["y"])
    origin_z = float(blockmodel.origin["z"])

    x_edges = origin_x + np.asarray(
        blockmodel.u_cell_delimiters, dtype=float
    )
    y_edges = origin_y + np.asarray(
        blockmodel.v_cell_delimiters, dtype=float
    )
    z_edges = origin_z + np.asarray(
        blockmodel.z_cell_delimiters, dtype=float
    )

    rg = pv.RectilinearGrid(x_edges, y_edges, z_edges)

    children = _get_data_children(blockmodel)
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    for ch_name, ch_vals in children.items():
        rg.cell_data[ch_name] = ch_vals

    return rg


def pyvista_to_blockmodel(
    rectilinear_grid: pv.RectilinearGrid,
    workspace: Workspace,
    *,
    name: str = "BlockModel",
    data_names: list[str] | None = None,
) -> BlockModel:
    """Convert a PyVista RectilinearGrid to a geoh5py BlockModel.

    Parameters
    ----------
    rectilinear_grid : pyvista.RectilinearGrid
        Rectilinear grid with cell data.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the BlockModel object.
    data_names : list[str], optional
        Cell data arrays to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.BlockModel
    """
    from geoh5py.objects import BlockModel

    x_edges = np.asarray(rectilinear_grid.x, dtype=float)
    y_edges = np.asarray(rectilinear_grid.y, dtype=float)
    z_edges = np.asarray(rectilinear_grid.z, dtype=float)

    origin = [float(x_edges[0]), float(y_edges[0]), float(z_edges[0])]

    # Delimiters are relative to origin
    u_delims = x_edges - x_edges[0]
    v_delims = y_edges - y_edges[0]
    z_delims = z_edges - z_edges[0]

    bm = BlockModel.create(
        workspace,
        origin=origin,
        u_cell_delimiters=u_delims,
        v_cell_delimiters=v_delims,
        z_cell_delimiters=z_delims,
        name=name,
    )

    keys = list(rectilinear_grid.cell_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    data = {
        k: np.asarray(rectilinear_grid.cell_data[k], dtype=float)
        for k in keys
    }
    if data:
        _add_data_columns(bm, data)

    return bm
