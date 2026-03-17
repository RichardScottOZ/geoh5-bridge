"""Bidirectional conversions between OMF elements and PyVista meshes.

This module improves upon ``omfvista`` (which only supports OMF → PyVista)
by adding the **reverse direction** (PyVista → OMF).  The forward
conversions are self-contained so that ``omfvista`` is *not* required as
a dependency — only the lightweight ``omf`` package is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import omf
    import pyvista as pv


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _scalar_data_for_element(
    element: omf.PointSetElement
    | omf.LineSetElement
    | omf.SurfaceElement
    | omf.VolumeElement,
) -> list[tuple[str, np.ndarray, str]]:
    """Return ``[(name, array, location), ...]`` for ScalarData children."""
    import omf as _omf

    result: list[tuple[str, np.ndarray, str]] = []
    for d in element.data:
        if isinstance(d, _omf.ScalarData):
            result.append((d.name, np.asarray(d.array, dtype=float), d.location))
    return result


# ------------------------------------------------------------------
# PointSet ↔ PyVista
# ------------------------------------------------------------------


def omf_pointset_to_pyvista(
    pointset: omf.PointSetElement,
) -> pv.PolyData:
    """Convert an OMF PointSetElement to a PyVista PolyData.

    Parameters
    ----------
    pointset : omf.PointSetElement
        OMF point-set element.

    Returns
    -------
    pyvista.PolyData
    """
    import pyvista as pv

    vertices = np.asarray(pointset.geometry.vertices, dtype=float)
    polydata = pv.PolyData(vertices)

    for name, arr, location in _scalar_data_for_element(pointset):
        if location == "vertices":
            polydata.point_data[name] = arr

    return polydata


def pyvista_to_omf_pointset(
    polydata: pv.PolyData,
    *,
    name: str = "PointSet",
    data_names: list[str] | None = None,
) -> omf.PointSetElement:
    """Convert a PyVista PolyData (point cloud) to an OMF PointSetElement.

    Parameters
    ----------
    polydata : pyvista.PolyData
        Point cloud.
    name : str, optional
        Element name.
    data_names : list[str], optional
        Point data arrays to include.  When *None*, all are included.

    Returns
    -------
    omf.PointSetElement
    """
    import omf as _omf

    vertices = np.asarray(polydata.points, dtype=float)
    data: list[_omf.ScalarData] = []

    keys = list(polydata.point_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    for k in keys:
        arr = np.asarray(polydata.point_data[k], dtype=float)
        data.append(
            _omf.ScalarData(name=k, array=arr, location="vertices")
        )

    return _omf.PointSetElement(
        name=name,
        geometry=_omf.PointSetGeometry(vertices=vertices),
        data=data,
    )


# ------------------------------------------------------------------
# LineSet ↔ PyVista
# ------------------------------------------------------------------


def omf_lineset_to_pyvista(
    lineset: omf.LineSetElement,
) -> pv.PolyData:
    """Convert an OMF LineSetElement to a PyVista PolyData with lines.

    Parameters
    ----------
    lineset : omf.LineSetElement
        OMF line-set element.

    Returns
    -------
    pyvista.PolyData
    """
    import pyvista as pv

    from geoh5_bridge.utils import _reconstruct_polylines

    vertices = np.asarray(lineset.geometry.vertices, dtype=float)
    segments = np.asarray(lineset.geometry.segments, dtype=int)

    # Reconstruct polylines and build PyVista lines array
    polylines = _reconstruct_polylines(segments)
    pv_lines: list[int] = []
    for line in polylines:
        pv_lines.append(len(line))
        pv_lines.extend(line)

    polydata = pv.PolyData(vertices, lines=np.array(pv_lines))

    for dname, arr, location in _scalar_data_for_element(lineset):
        if location == "vertices":
            polydata.point_data[dname] = arr
        elif location == "segments":
            polydata.cell_data[dname] = arr

    return polydata


def pyvista_to_omf_lineset(
    polydata: pv.PolyData,
    *,
    name: str = "LineSet",
    data_names: list[str] | None = None,
) -> omf.LineSetElement:
    """Convert a PyVista PolyData with lines to an OMF LineSetElement.

    Parameters
    ----------
    polydata : pyvista.PolyData
        PolyData with line cells.
    name : str, optional
        Element name.
    data_names : list[str], optional
        Point data arrays to include.  When *None*, all are included.

    Returns
    -------
    omf.LineSetElement
    """
    import omf as _omf

    vertices = np.asarray(polydata.points, dtype=float)

    # Parse PyVista lines → segment pairs
    lines_arr = np.asarray(polydata.lines)
    all_segments: list[list[int]] = []
    i = 0
    while i < len(lines_arr):
        n = lines_arr[i]
        indices = lines_arr[i + 1 : i + 1 + n].tolist()
        for j in range(len(indices) - 1):
            all_segments.append([indices[j], indices[j + 1]])
        i += 1 + n

    segments = np.array(all_segments, dtype=int)

    data: list[_omf.ScalarData] = []
    keys = list(polydata.point_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    for k in keys:
        arr = np.asarray(polydata.point_data[k], dtype=float)
        data.append(
            _omf.ScalarData(name=k, array=arr, location="vertices")
        )

    return _omf.LineSetElement(
        name=name,
        geometry=_omf.LineSetGeometry(vertices=vertices, segments=segments),
        data=data,
    )


# ------------------------------------------------------------------
# Surface ↔ PyVista
# ------------------------------------------------------------------


def omf_surface_to_pyvista(
    surface: omf.SurfaceElement,
) -> pv.PolyData:
    """Convert an OMF SurfaceElement to a PyVista PolyData triangle mesh.

    Handles both ``SurfaceGeometry`` (explicit triangles) and
    ``SurfaceGridGeometry`` (structured grid surface).

    Parameters
    ----------
    surface : omf.SurfaceElement
        OMF surface element.

    Returns
    -------
    pyvista.PolyData
    """
    import omf as _omf
    import pyvista as pv

    geom = surface.geometry

    if isinstance(geom, _omf.SurfaceGeometry):
        vertices = np.asarray(geom.vertices, dtype=float)
        triangles = np.asarray(geom.triangles, dtype=np.uint32)
        n_faces = len(triangles)
        pv_faces = np.column_stack(
            [np.full(n_faces, 3, dtype=np.uint32), triangles]
        ).ravel()
        polydata = pv.PolyData(vertices, pv_faces)

    elif isinstance(geom, _omf.SurfaceGridGeometry):
        # Build structured grid node coordinates
        origin = np.asarray(geom.origin, dtype=float)
        tensor_u = np.asarray(geom.tensor_u, dtype=float)
        tensor_v = np.asarray(geom.tensor_v, dtype=float)
        axis_u = np.asarray(geom.axis_u, dtype=float)
        axis_v = np.asarray(geom.axis_v, dtype=float)

        nu = len(tensor_u) + 1  # node count
        nv = len(tensor_v) + 1

        u_edges = np.concatenate([[0], np.cumsum(tensor_u)])
        v_edges = np.concatenate([[0], np.cumsum(tensor_v)])

        # Build grid of points
        vertices_list: list[np.ndarray] = []
        offset_z = np.asarray(geom.offset_w) if geom.offset_w is not None else None
        for j in range(nv):
            for i in range(nu):
                pt = origin + u_edges[i] * axis_u + v_edges[j] * axis_v
                if offset_z is not None:
                    idx = j * nu + i
                    if idx < len(offset_z):
                        pt = pt + np.array([0, 0, float(offset_z[idx])])
                vertices_list.append(pt)

        vertices = np.array(vertices_list, dtype=float)

        # Build triangle faces
        face_list: list[list[int]] = []
        for j in range(nv - 1):
            for i in range(nu - 1):
                v0 = j * nu + i
                v1 = v0 + 1
                v2 = v0 + nu
                v3 = v2 + 1
                face_list.append([3, v0, v1, v3])
                face_list.append([3, v0, v3, v2])

        pv_faces = np.array([x for f in face_list for x in f], dtype=np.uint32)
        polydata = pv.PolyData(vertices, pv_faces)
    else:
        raise TypeError(f"Unsupported surface geometry type: {type(geom)}")

    for dname, arr, location in _scalar_data_for_element(surface):
        if location == "vertices":
            if len(arr) == polydata.n_points:
                polydata.point_data[dname] = arr
        elif location == "faces":
            if len(arr) == polydata.n_cells:
                polydata.cell_data[dname] = arr

    return polydata


def pyvista_to_omf_surface(
    polydata: pv.PolyData,
    *,
    name: str = "Surface",
    data_names: list[str] | None = None,
) -> omf.SurfaceElement:
    """Convert a PyVista PolyData triangle mesh to an OMF SurfaceElement.

    The PolyData must contain only triangular faces.

    Parameters
    ----------
    polydata : pyvista.PolyData
        Triangle mesh.
    name : str, optional
        Element name.
    data_names : list[str], optional
        Point data arrays to include.  When *None*, all are included.

    Returns
    -------
    omf.SurfaceElement

    Raises
    ------
    ValueError
        If the PolyData contains non-triangular faces.
    """
    import omf as _omf

    if not polydata.is_all_triangles:
        raise ValueError(
            "PolyData must contain only triangular faces. "
            "Use polydata.triangulate() first."
        )

    vertices = np.asarray(polydata.points, dtype=float)
    faces = np.asarray(polydata.faces)
    n_faces = polydata.n_cells
    triangles = faces.reshape(n_faces, 4)[:, 1:].astype(int)

    data: list[_omf.ScalarData] = []
    keys = list(polydata.point_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    for k in keys:
        arr = np.asarray(polydata.point_data[k], dtype=float)
        data.append(
            _omf.ScalarData(name=k, array=arr, location="vertices")
        )

    return _omf.SurfaceElement(
        name=name,
        geometry=_omf.SurfaceGeometry(vertices=vertices, triangles=triangles),
        data=data,
    )


# ------------------------------------------------------------------
# Volume ↔ PyVista
# ------------------------------------------------------------------


def omf_volume_to_pyvista(
    volume: omf.VolumeElement,
) -> pv.RectilinearGrid:
    """Convert an OMF VolumeElement to a PyVista RectilinearGrid.

    Parameters
    ----------
    volume : omf.VolumeElement
        OMF volume element with grid geometry.

    Returns
    -------
    pyvista.RectilinearGrid
    """
    import pyvista as pv

    geom = volume.geometry
    origin = np.asarray(geom.origin, dtype=float)
    tensor_u = np.asarray(geom.tensor_u, dtype=float)
    tensor_v = np.asarray(geom.tensor_v, dtype=float)
    tensor_w = np.asarray(geom.tensor_w, dtype=float)

    x_edges = origin[0] + np.concatenate([[0], np.cumsum(tensor_u)])
    y_edges = origin[1] + np.concatenate([[0], np.cumsum(tensor_v)])
    z_edges = origin[2] + np.concatenate([[0], np.cumsum(tensor_w)])

    rg = pv.RectilinearGrid(x_edges, y_edges, z_edges)

    for dname, arr, location in _scalar_data_for_element(volume):
        if location == "cells":
            rg.cell_data[dname] = arr
        elif location == "vertices":
            rg.point_data[dname] = arr

    return rg


def pyvista_to_omf_volume(
    rectilinear_grid: pv.RectilinearGrid,
    *,
    name: str = "Volume",
    data_names: list[str] | None = None,
) -> omf.VolumeElement:
    """Convert a PyVista RectilinearGrid to an OMF VolumeElement.

    Parameters
    ----------
    rectilinear_grid : pyvista.RectilinearGrid
        Rectilinear grid.
    name : str, optional
        Element name.
    data_names : list[str], optional
        Cell data arrays to include.  When *None*, all are included.

    Returns
    -------
    omf.VolumeElement
    """
    import omf as _omf

    x_edges = np.asarray(rectilinear_grid.x, dtype=float)
    y_edges = np.asarray(rectilinear_grid.y, dtype=float)
    z_edges = np.asarray(rectilinear_grid.z, dtype=float)

    origin = [float(x_edges[0]), float(y_edges[0]), float(z_edges[0])]
    tensor_u = np.diff(x_edges)
    tensor_v = np.diff(y_edges)
    tensor_w = np.diff(z_edges)

    data: list[_omf.ScalarData] = []
    keys = list(rectilinear_grid.cell_data.keys())
    if data_names is not None:
        keys = [k for k in keys if k in data_names]

    for k in keys:
        arr = np.asarray(rectilinear_grid.cell_data[k], dtype=float)
        data.append(
            _omf.ScalarData(name=k, array=arr, location="cells")
        )

    return _omf.VolumeElement(
        name=name,
        geometry=_omf.VolumeGridGeometry(
            origin=origin,
            tensor_u=tensor_u,
            tensor_v=tensor_v,
            tensor_w=tensor_w,
            axis_u=[1.0, 0.0, 0.0],
            axis_v=[0.0, 1.0, 0.0],
            axis_w=[0.0, 0.0, 1.0],
        ),
        data=data,
    )


# ------------------------------------------------------------------
# Project ↔ PyVista MultiBlock
# ------------------------------------------------------------------


def omf_project_to_pyvista(
    project: omf.Project,
) -> pv.MultiBlock:
    """Convert an OMF Project to a PyVista MultiBlock.

    Each element is converted to the appropriate PyVista mesh type and
    stored in the MultiBlock under the element's name.

    Parameters
    ----------
    project : omf.Project
        OMF project.

    Returns
    -------
    pyvista.MultiBlock
    """
    import omf as _omf
    import pyvista as pv

    mb = pv.MultiBlock()

    for element in project.elements:
        if isinstance(element, _omf.PointSetElement):
            mb[element.name] = omf_pointset_to_pyvista(element)
        elif isinstance(element, _omf.LineSetElement):
            mb[element.name] = omf_lineset_to_pyvista(element)
        elif isinstance(element, _omf.SurfaceElement):
            mb[element.name] = omf_surface_to_pyvista(element)
        elif isinstance(element, _omf.VolumeElement):
            mb[element.name] = omf_volume_to_pyvista(element)

    return mb


def pyvista_to_omf_project(
    multiblock: pv.MultiBlock,
    *,
    project_name: str = "Project",
) -> omf.Project:
    """Convert a PyVista MultiBlock to an OMF Project.

    Block types are inferred from the PyVista mesh type:
    - ``PolyData`` with only vertices → ``PointSetElement``
    - ``PolyData`` with lines → ``LineSetElement``
    - ``PolyData`` with triangle faces → ``SurfaceElement``
    - ``RectilinearGrid`` → ``VolumeElement``

    Parameters
    ----------
    multiblock : pyvista.MultiBlock
        Container of PyVista meshes.
    project_name : str, optional
        Name for the OMF project.

    Returns
    -------
    omf.Project
    """
    import omf as _omf
    import pyvista as pv

    elements: list = []

    for i in range(multiblock.n_blocks):
        block = multiblock[i]
        block_name = multiblock.keys()[i] or f"Element_{i}"

        if isinstance(block, pv.RectilinearGrid):
            elements.append(
                pyvista_to_omf_volume(block, name=block_name)
            )
        elif isinstance(block, pv.PolyData):
            if block.n_lines > 0:
                elements.append(
                    pyvista_to_omf_lineset(block, name=block_name)
                )
            elif block.n_cells > 0 and block.is_all_triangles:
                elements.append(
                    pyvista_to_omf_surface(block, name=block_name)
                )
            else:
                # Default: point set
                elements.append(
                    pyvista_to_omf_pointset(block, name=block_name)
                )

    return _omf.Project(name=project_name, elements=elements)
