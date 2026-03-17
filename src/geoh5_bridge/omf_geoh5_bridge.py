"""Bidirectional conversions between OMF elements and geoh5py objects.

Provides direct OMF ↔ geoh5 conversion without going through PyVista,
preserving all geometry and scalar data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoh5_bridge.utils import _add_data_columns

if TYPE_CHECKING:
    import omf
    from geoh5py.objects import BlockModel, Curve, Points, Surface
    from geoh5py.workspace import Workspace


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _omf_scalar_data(
    element,
) -> dict[str, tuple[np.ndarray, str]]:
    """Return ``{name: (array, location)}`` for ScalarData children."""
    import omf as _omf

    result: dict[str, tuple[np.ndarray, str]] = {}
    for d in element.data:
        if isinstance(d, _omf.ScalarData):
            result[d.name] = (np.asarray(d.array, dtype=float), d.location)
    return result


# ------------------------------------------------------------------
# PointSet ↔ Points
# ------------------------------------------------------------------


def omf_pointset_to_points(
    pointset: omf.PointSetElement,
    workspace: Workspace,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> Points:
    """Convert an OMF PointSetElement to a geoh5py Points object.

    Parameters
    ----------
    pointset : omf.PointSetElement
        OMF point-set element.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Points object.  Defaults to the OMF element name.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.Points
    """
    from geoh5py.objects import Points

    obj_name = name or pointset.name or "Points"
    vertices = np.asarray(pointset.geometry.vertices, dtype=float)
    pts = Points.create(workspace, vertices=vertices, name=obj_name)

    all_data = _omf_scalar_data(pointset)
    if data_names is not None:
        all_data = {k: v for k, v in all_data.items() if k in data_names}

    vertex_data = {k: arr for k, (arr, loc) in all_data.items() if loc == "vertices"}
    if vertex_data:
        _add_data_columns(pts, vertex_data)

    return pts


def points_to_omf_pointset(
    pts: Points,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> omf.PointSetElement:
    """Convert a geoh5py Points object to an OMF PointSetElement.

    Parameters
    ----------
    pts : geoh5py.objects.Points
        Points object.
    name : str, optional
        Element name.  Defaults to the geoh5 object name.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    omf.PointSetElement
    """
    import omf as _omf

    obj_name = name or pts.name or "PointSet"
    vertices = np.asarray(pts.vertices, dtype=float)

    children = {
        c.name: np.asarray(c.values, dtype=float)
        for c in pts.children
        if hasattr(c, "values")
    }
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    data = [
        _omf.ScalarData(name=k, array=v, location="vertices")
        for k, v in children.items()
    ]

    return _omf.PointSetElement(
        name=obj_name,
        geometry=_omf.PointSetGeometry(vertices=vertices),
        data=data,
    )


# ------------------------------------------------------------------
# LineSet ↔ Curve
# ------------------------------------------------------------------


def omf_lineset_to_curve(
    lineset: omf.LineSetElement,
    workspace: Workspace,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> Curve:
    """Convert an OMF LineSetElement to a geoh5py Curve.

    Parameters
    ----------
    lineset : omf.LineSetElement
        OMF line-set element.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Curve object.  Defaults to the OMF element name.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.Curve
    """
    from geoh5py.objects import Curve

    obj_name = name or lineset.name or "Curve"
    vertices = np.asarray(lineset.geometry.vertices, dtype=float)
    segments = np.asarray(lineset.geometry.segments, dtype=np.uint32)

    curve = Curve.create(
        workspace, vertices=vertices, cells=segments, name=obj_name
    )

    all_data = _omf_scalar_data(lineset)
    if data_names is not None:
        all_data = {k: v for k, v in all_data.items() if k in data_names}

    vertex_data = {k: arr for k, (arr, loc) in all_data.items() if loc == "vertices"}
    if vertex_data:
        _add_data_columns(curve, vertex_data)

    return curve


def curve_to_omf_lineset(
    curve: Curve,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> omf.LineSetElement:
    """Convert a geoh5py Curve to an OMF LineSetElement.

    Parameters
    ----------
    curve : geoh5py.objects.Curve
        Curve object.
    name : str, optional
        Element name.  Defaults to the geoh5 object name.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    omf.LineSetElement
    """
    import omf as _omf

    obj_name = name or curve.name or "LineSet"
    vertices = np.asarray(curve.vertices, dtype=float)
    cells = np.asarray(curve.cells, dtype=int)

    children = {
        c.name: np.asarray(c.values, dtype=float)
        for c in curve.children
        if hasattr(c, "values")
    }
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    data = [
        _omf.ScalarData(name=k, array=v, location="vertices")
        for k, v in children.items()
    ]

    return _omf.LineSetElement(
        name=obj_name,
        geometry=_omf.LineSetGeometry(vertices=vertices, segments=cells),
        data=data,
    )


# ------------------------------------------------------------------
# Surface ↔ Surface
# ------------------------------------------------------------------


def omf_surface_to_surface(
    omf_surface: omf.SurfaceElement,
    workspace: Workspace,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> Surface:
    """Convert an OMF SurfaceElement (triangle mesh) to a geoh5py Surface.

    Only ``SurfaceGeometry`` (explicit triangles) is supported. For
    ``SurfaceGridGeometry`` (structured grid surfaces), convert to
    PyVista first using :func:`omf_surface_to_pyvista`.

    Parameters
    ----------
    omf_surface : omf.SurfaceElement
        OMF surface element with SurfaceGeometry.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Surface object.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.Surface

    Raises
    ------
    TypeError
        If the geometry is not ``SurfaceGeometry``.
    """
    import omf as _omf
    from geoh5py.objects import Surface

    geom = omf_surface.geometry
    if not isinstance(geom, _omf.SurfaceGeometry):
        raise TypeError(
            f"Expected SurfaceGeometry, got {type(geom).__name__}. "
            "Use omf_surface_to_pyvista for grid surfaces."
        )

    obj_name = name or omf_surface.name or "Surface"
    vertices = np.asarray(geom.vertices, dtype=float)
    cells = np.asarray(geom.triangles, dtype=np.uint32)

    surface = Surface.create(
        workspace, vertices=vertices, cells=cells, name=obj_name
    )

    all_data = _omf_scalar_data(omf_surface)
    if data_names is not None:
        all_data = {k: v for k, v in all_data.items() if k in data_names}

    vertex_data = {k: arr for k, (arr, loc) in all_data.items() if loc == "vertices"}
    if vertex_data:
        _add_data_columns(surface, vertex_data)

    return surface


def surface_to_omf_surface(
    surface: Surface,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> omf.SurfaceElement:
    """Convert a geoh5py Surface to an OMF SurfaceElement.

    Parameters
    ----------
    surface : geoh5py.objects.Surface
        Surface object with triangle cells.
    name : str, optional
        Element name.  Defaults to the geoh5 object name.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    omf.SurfaceElement
    """
    import omf as _omf

    obj_name = name or surface.name or "Surface"
    vertices = np.asarray(surface.vertices, dtype=float)
    cells = np.asarray(surface.cells, dtype=int)

    children = {
        c.name: np.asarray(c.values, dtype=float)
        for c in surface.children
        if hasattr(c, "values")
    }
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    data = [
        _omf.ScalarData(name=k, array=v, location="vertices")
        for k, v in children.items()
    ]

    return _omf.SurfaceElement(
        name=obj_name,
        geometry=_omf.SurfaceGeometry(vertices=vertices, triangles=cells),
        data=data,
    )


# ------------------------------------------------------------------
# Volume ↔ BlockModel
# ------------------------------------------------------------------


def omf_volume_to_blockmodel(
    volume: omf.VolumeElement,
    workspace: Workspace,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> BlockModel:
    """Convert an OMF VolumeElement to a geoh5py BlockModel.

    Parameters
    ----------
    volume : omf.VolumeElement
        OMF volume element.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the BlockModel object.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.BlockModel
    """
    from geoh5py.objects import BlockModel

    geom = volume.geometry
    obj_name = name or volume.name or "BlockModel"

    origin = list(np.asarray(geom.origin, dtype=float))
    tensor_u = np.asarray(geom.tensor_u, dtype=float)
    tensor_v = np.asarray(geom.tensor_v, dtype=float)
    tensor_w = np.asarray(geom.tensor_w, dtype=float)

    # geoh5py BlockModel uses cell-edge delimiters relative to origin
    u_delims = np.concatenate([[0], np.cumsum(tensor_u)])
    v_delims = np.concatenate([[0], np.cumsum(tensor_v)])
    z_delims = np.concatenate([[0], np.cumsum(tensor_w)])

    bm = BlockModel.create(
        workspace,
        origin=origin,
        u_cell_delimiters=u_delims,
        v_cell_delimiters=v_delims,
        z_cell_delimiters=z_delims,
        name=obj_name,
    )

    all_data = _omf_scalar_data(volume)
    if data_names is not None:
        all_data = {k: v for k, v in all_data.items() if k in data_names}

    cell_data = {k: arr for k, (arr, loc) in all_data.items() if loc == "cells"}
    if cell_data:
        _add_data_columns(bm, cell_data)

    return bm


def blockmodel_to_omf_volume(
    blockmodel: BlockModel,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> omf.VolumeElement:
    """Convert a geoh5py BlockModel to an OMF VolumeElement.

    Parameters
    ----------
    blockmodel : geoh5py.objects.BlockModel
        BlockModel object.
    name : str, optional
        Element name.  Defaults to the geoh5 object name.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    omf.VolumeElement
    """
    import omf as _omf

    obj_name = name or blockmodel.name or "Volume"

    origin_x = float(blockmodel.origin["x"])
    origin_y = float(blockmodel.origin["y"])
    origin_z = float(blockmodel.origin["z"])

    u_delims = np.asarray(blockmodel.u_cell_delimiters, dtype=float)
    v_delims = np.asarray(blockmodel.v_cell_delimiters, dtype=float)
    z_delims = np.asarray(blockmodel.z_cell_delimiters, dtype=float)

    tensor_u = np.diff(u_delims)
    tensor_v = np.diff(v_delims)
    tensor_w = np.diff(z_delims)

    children = {
        c.name: np.asarray(c.values, dtype=float)
        for c in blockmodel.children
        if hasattr(c, "values")
    }
    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    data = [
        _omf.ScalarData(name=k, array=v, location="cells")
        for k, v in children.items()
    ]

    return _omf.VolumeElement(
        name=obj_name,
        geometry=_omf.VolumeGridGeometry(
            origin=[origin_x, origin_y, origin_z],
            tensor_u=tensor_u,
            tensor_v=tensor_v,
            tensor_w=tensor_w,
            axis_u=[1.0, 0.0, 0.0],
            axis_v=[0.0, 1.0, 0.0],
            axis_w=[0.0, 0.0, 1.0],
        ),
        data=data,
    )
