"""Bidirectional conversions between OMF elements and geoh5py objects.

Provides direct OMF ↔ geoh5 conversion without going through PyVista,
preserving all geometry and data (scalar, integer, string, vector, and
categorical/referenced data).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import omf
    from geoh5py.objects import BlockModel, Curve, Grid2D, Points, Surface
    from geoh5py.workspace import Workspace


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _omf_all_data(element) -> list[dict]:
    """Extract all recognised data from an OMF element.

    Returns a list of dicts, each with at minimum the keys:

    * ``name`` – data channel name
    * ``array`` – ``numpy.ndarray`` (or ``list[str]`` for string data)
    * ``location`` – OMF location string (``"vertices"``, ``"cells"``, …)
    * ``kind`` – one of ``"float"``, ``"integer"``, ``"string"``,
      ``"vector2"``, ``"vector3"``, ``"mapped"``

    Dicts with ``kind == "mapped"`` additionally contain:

    * ``legend`` – ``list[str]`` of category labels (0-based)
    """
    import omf as _omf

    result: list[dict] = []
    for d in element.data:
        item: dict = {"name": d.name, "location": d.location}

        if isinstance(d, _omf.ScalarData):
            arr = np.asarray(d.array)
            if np.issubdtype(arr.dtype, np.integer):
                item["array"] = arr
                item["kind"] = "integer"
            else:
                item["array"] = arr.astype(float)
                item["kind"] = "float"

        elif isinstance(d, _omf.Vector3Data):
            item["array"] = np.asarray(d.array, dtype=float)
            item["kind"] = "vector3"

        elif isinstance(d, _omf.Vector2Data):
            item["array"] = np.asarray(d.array, dtype=float)
            item["kind"] = "vector2"

        elif isinstance(d, _omf.StringData):
            item["array"] = list(d.array)
            item["kind"] = "string"

        elif isinstance(d, _omf.MappedData):
            item["array"] = np.asarray(d.array, dtype=int)
            item["kind"] = "mapped"
            item["legend"] = (
                [str(v) for v in d.legends[0].values] if d.legends else []
            )

        else:
            continue  # skip ColorData, DateTimeData, etc.

        result.append(item)
    return result


def _omf_scalar_data(
    element,
) -> dict[str, tuple[np.ndarray, str]]:
    """Return ``{name: (array, location)}`` for ScalarData children.

    .. deprecated::
        Use :func:`_omf_all_data` for new code.  This function is kept for
        internal compatibility.
    """
    import omf as _omf

    result: dict[str, tuple[np.ndarray, str]] = {}
    for d in element.data:
        if isinstance(d, _omf.ScalarData):
            result[d.name] = (np.asarray(d.array, dtype=float), d.location)
    return result


def _add_omf_data_to_geoh5(
    geoh5_obj,
    data_items: list[dict],
    *,
    data_names: list[str] | None = None,
    location_filter: str | None = None,
) -> None:
    """Add OMF data items to a geoh5py object.

    Parameters
    ----------
    geoh5_obj
        Any geoh5py object that supports ``add_data()``.
    data_items
        List of data dicts as returned by :func:`_omf_all_data`.
    data_names
        Optional allow-list of channel names.  ``None`` means include all.
    location_filter
        If given, only items with ``item["location"] == location_filter``
        are added.
    """
    for item in data_items:
        name = item["name"]
        if data_names is not None and name not in data_names:
            continue
        if location_filter is not None and item["location"] != location_filter:
            continue

        kind = item["kind"]
        arr = item["array"]

        if kind == "float":
            geoh5_obj.add_data({name: {"values": np.asarray(arr, dtype=np.float32)}})

        elif kind == "integer":
            geoh5_obj.add_data({name: {"values": np.asarray(arr, dtype=np.int32)}})

        elif kind == "string":
            geoh5_obj.add_data({name: {"values": np.asarray(arr)}})

        elif kind == "mapped":
            legend = item.get("legend", [])
            # OMF uses 0-based indices; geoh5py ReferencedData uses 1-based.
            arr_1based = np.asarray(arr, dtype=np.int32) + 1
            value_map = {i + 1: str(v) for i, v in enumerate(legend)}
            geoh5_obj.add_data(
                {
                    name: {
                        "type": "referenced",
                        "values": arr_1based,
                        "value_map": value_map,
                    }
                }
            )

        elif kind == "vector3":
            arr2d = np.asarray(arr, dtype=np.float32)
            for i, suffix in enumerate(("_x", "_y", "_z")):
                geoh5_obj.add_data(
                    {name + suffix: {"values": arr2d[:, i]}}
                )

        elif kind == "vector2":
            arr2d = np.asarray(arr, dtype=np.float32)
            for i, suffix in enumerate(("_x", "_y")):
                geoh5_obj.add_data(
                    {name + suffix: {"values": arr2d[:, i]}}
                )


def _geoh5_children_to_omf_data(
    geoh5_obj,
    *,
    data_names: list[str] | None = None,
) -> list:
    """Convert geoh5py children data to a list of OMF data objects.

    Supports :class:`~geoh5py.data.FloatData`,
    :class:`~geoh5py.data.IntegerData`,
    :class:`~geoh5py.data.TextData`, and
    :class:`~geoh5py.data.ReferencedData`.

    Parameters
    ----------
    geoh5_obj
        geoh5py object whose children to convert.
    data_names
        Optional allow-list of channel names.

    Returns
    -------
    list
        List of OMF data objects suitable for an element's ``data`` attribute.
    """
    import omf as _omf
    from geoh5py.data import FloatData, IntegerData, ReferencedData, TextData

    result = []
    for c in geoh5_obj.children:
        if not hasattr(c, "values") or c.values is None:
            continue
        name = c.name
        if data_names is not None and name not in data_names:
            continue

        if isinstance(c, ReferencedData):
            # Rebuild OMF MappedData from ReferencedData
            raw_indices = np.asarray(c.values, dtype=int)
            # geoh5 uses 1-based indices; OMF uses 0-based
            indices_0based = raw_indices - 1
            vm_map = c.entity_type.value_map.map
            # Build ordered legend (sorted by key, excluding key=0 "Unknown")
            vm = {
                int(k): v.decode() if isinstance(v, bytes) else str(v)
                for k, v in vm_map
                if int(k) != 0
            }
            max_key = max(vm.keys()) if vm else 0
            legend_vals = [vm.get(i + 1, "") for i in range(max_key)]
            legend = _omf.Legend(values=legend_vals)
            result.append(
                _omf.MappedData(
                    name=name,
                    array=indices_0based,
                    location="vertices",
                    legends=[legend],
                )
            )

        elif isinstance(c, TextData):
            result.append(
                _omf.StringData(
                    name=name,
                    array=list(c.values),
                    location="vertices",
                )
            )

        elif isinstance(c, FloatData):
            result.append(
                _omf.ScalarData(
                    name=name,
                    array=np.asarray(c.values, dtype=float),
                    location="vertices",
                )
            )

        elif isinstance(c, IntegerData):
            result.append(
                _omf.ScalarData(
                    name=name,
                    array=np.asarray(c.values, dtype=float),
                    location="vertices",
                )
            )

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

    all_data = _omf_all_data(pointset)
    _add_omf_data_to_geoh5(
        pts,
        all_data,
        data_names=data_names,
        location_filter="vertices",
    )

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

    data = _geoh5_children_to_omf_data(pts, data_names=data_names)

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

    all_data = _omf_all_data(lineset)
    _add_omf_data_to_geoh5(
        curve,
        all_data,
        data_names=data_names,
        location_filter="vertices",
    )

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

    data = _geoh5_children_to_omf_data(curve, data_names=data_names)

    return _omf.LineSetElement(
        name=obj_name,
        geometry=_omf.LineSetGeometry(vertices=vertices, segments=cells),
        data=data,
    )


# ------------------------------------------------------------------
# Surface ↔ Surface / Grid2D
# ------------------------------------------------------------------


def _is_uniform(arr: np.ndarray) -> bool:
    """Return True if all elements of *arr* are equal."""
    return bool(np.allclose(arr, arr[0]))


def _grid_geom_to_triangles(
    geom,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a ``SurfaceGridGeometry``.

    Returns
    -------
    vertices : numpy.ndarray, shape (N, 3)
    cells : numpy.ndarray, shape (M, 3)  (triangle indices)
    """
    origin = np.asarray(geom.origin, dtype=float)
    tensor_u = np.asarray(geom.tensor_u, dtype=float)
    tensor_v = np.asarray(geom.tensor_v, dtype=float)
    axis_u = np.asarray(geom.axis_u, dtype=float)
    axis_v = np.asarray(geom.axis_v, dtype=float)

    nu = len(tensor_u) + 1
    nv = len(tensor_v) + 1

    u_edges = np.concatenate([[0], np.cumsum(tensor_u)])
    v_edges = np.concatenate([[0], np.cumsum(tensor_v)])

    offset_w = (
        np.asarray(geom.offset_w) if geom.offset_w is not None else None
    )

    vertices_list: list[np.ndarray] = []
    for j in range(nv):
        for i in range(nu):
            pt = origin + u_edges[i] * axis_u + v_edges[j] * axis_v
            if offset_w is not None:
                idx = j * nu + i
                if idx < len(offset_w):
                    pt = pt + np.array([0, 0, float(offset_w[idx])])
            vertices_list.append(pt)

    vertices = np.array(vertices_list, dtype=float)

    tri_list: list[list[int]] = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            v0 = j * nu + i
            v1 = v0 + 1
            v2 = v0 + nu
            v3 = v2 + 1
            tri_list.append([v0, v1, v3])
            tri_list.append([v0, v3, v2])

    cells = np.array(tri_list, dtype=np.uint32)
    return vertices, cells


def omf_surface_to_grid2d(
    omf_surface: omf.SurfaceElement,
    workspace: Workspace,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> Grid2D:
    """Convert an OMF ``SurfaceGridGeometry`` surface to a geoh5py ``Grid2D``.

    The input *must* have ``SurfaceGridGeometry`` with **uniform** cell
    spacing in both directions and **no** ``offset_w`` elevation offsets.
    For surfaces that do not satisfy these constraints, use
    :func:`omf_surface_to_surface` with ``prefer_grid2d=False``.

    Parameters
    ----------
    omf_surface : omf.SurfaceElement
        OMF surface element with ``SurfaceGridGeometry``.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Grid2D object.  Defaults to the OMF element name.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    geoh5py.objects.Grid2D

    Raises
    ------
    TypeError
        If *omf_surface* does not have ``SurfaceGridGeometry``.
    ValueError
        If the grid has non-uniform cell spacing or uses ``offset_w``.
    """
    import omf as _omf
    from geoh5py.objects import Grid2D

    geom = omf_surface.geometry
    if not isinstance(geom, _omf.SurfaceGridGeometry):
        raise TypeError(
            "omf_surface_to_grid2d requires SurfaceGridGeometry; "
            f"got {type(geom).__name__}. Use omf_surface_to_surface() for "
            "explicit triangle meshes."
        )

    tensor_u = np.asarray(geom.tensor_u, dtype=float)
    tensor_v = np.asarray(geom.tensor_v, dtype=float)

    if not _is_uniform(tensor_u):
        raise ValueError(
            "omf_surface_to_grid2d requires uniform cell spacing in u-direction. "
            "Non-uniform spacing detected. Use omf_surface_to_surface() instead."
        )
    if not _is_uniform(tensor_v):
        raise ValueError(
            "omf_surface_to_grid2d requires uniform cell spacing in v-direction. "
            "Non-uniform spacing detected. Use omf_surface_to_surface() instead."
        )
    if geom.offset_w is not None:
        raise ValueError(
            "omf_surface_to_grid2d cannot represent surfaces with offset_w "
            "(per-node elevation offsets). Use omf_surface_to_surface() instead."
        )

    obj_name = name or omf_surface.name or "Grid2D"

    axis_u = np.asarray(geom.axis_u, dtype=float)
    axis_v = np.asarray(geom.axis_v, dtype=float)

    # rotation: angle of axis_u projected onto horizontal plane (degrees, CCW from East)
    rotation = math.degrees(math.atan2(float(axis_u[1]), float(axis_u[0])))
    # dip: angle of axis_v above horizontal (degrees)
    horiz_v = math.sqrt(float(axis_v[0]) ** 2 + float(axis_v[1]) ** 2)
    dip = math.degrees(math.atan2(float(axis_v[2]), horiz_v))

    origin = np.asarray(geom.origin, dtype=float)

    grid = Grid2D.create(
        workspace,
        origin=[float(origin[0]), float(origin[1]), float(origin[2])],
        u_cell_size=float(tensor_u[0]),
        v_cell_size=float(tensor_v[0]),
        u_count=len(tensor_u),
        v_count=len(tensor_v),
        rotation=rotation,
        dip=dip,
        name=obj_name,
    )

    all_data = _omf_all_data(omf_surface)
    _add_omf_data_to_geoh5(grid, all_data, data_names=data_names)

    return grid


def grid2d_to_omf_surface(
    grid2d: Grid2D,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
) -> omf.SurfaceElement:
    """Convert a geoh5py ``Grid2D`` to an OMF ``SurfaceElement``.

    The grid geometry is converted to :class:`omf.SurfaceGridGeometry` with
    uniform cell spacing derived from the Grid2D cell-size properties.

    The Grid2D ``rotation`` angle (degrees CCW from East) maps to
    ``axis_u``.  The ``dip`` angle (degrees above horizontal) controls
    the tilt of the v-direction: ``axis_v`` is perpendicular to ``axis_u``
    in the horizontal plane then rotated upward by ``dip``, so a ``dip`` of
    0° yields a flat horizontal grid and 90° yields a vertical plane.

    Parameters
    ----------
    grid2d : geoh5py.objects.Grid2D
        Grid2D object.
    name : str, optional
        Element name.  Defaults to the geoh5 object name.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.

    Returns
    -------
    omf.SurfaceElement
    """
    import omf as _omf

    obj_name = name or grid2d.name or "Surface"

    origin_x = float(grid2d.origin["x"])
    origin_y = float(grid2d.origin["y"])
    origin_z = float(grid2d.origin["z"])

    u_cell = float(grid2d.u_cell_size)
    v_cell = float(grid2d.v_cell_size)
    u_count = int(grid2d.u_count)
    v_count = int(grid2d.v_count)

    rot_rad = math.radians(float(grid2d.rotation))
    dip_rad = math.radians(float(grid2d.dip))

    # axis_u: unit vector in the horizontal plane at the given rotation angle
    # (CCW from East / x-axis).
    axis_u = [math.cos(rot_rad), math.sin(rot_rad), 0.0]

    # axis_v: unit vector perpendicular to axis_u, tilted upward by the dip
    # angle.  The horizontal component of axis_v is 90° CCW from axis_u;
    # the vertical component rises at the dip angle above the horizontal:
    #   axis_v = horiz_v * cos(dip) + [0, 0, 1] * sin(dip)
    horiz_v = [-math.sin(rot_rad), math.cos(rot_rad), 0.0]
    axis_v = [
        horiz_v[0] * math.cos(dip_rad),
        horiz_v[1] * math.cos(dip_rad),
        math.sin(dip_rad),
    ]

    tensor_u = np.full(u_count, u_cell)
    tensor_v = np.full(v_count, v_cell)

    data = _geoh5_children_to_omf_data(grid2d, data_names=data_names)

    return _omf.SurfaceElement(
        name=obj_name,
        geometry=_omf.SurfaceGridGeometry(
            origin=[origin_x, origin_y, origin_z],
            tensor_u=tensor_u,
            tensor_v=tensor_v,
            axis_u=axis_u,
            axis_v=axis_v,
        ),
        data=data,
    )


def omf_surface_to_surface(
    omf_surface: omf.SurfaceElement,
    workspace: Workspace,
    *,
    name: str | None = None,
    data_names: list[str] | None = None,
    prefer_grid2d: bool = True,
) -> Union[Surface, Grid2D]:
    """Convert an OMF SurfaceElement to a geoh5py Surface or Grid2D.

    When *prefer_grid2d* is ``True`` (the default) and the input has
    ``SurfaceGridGeometry`` with uniform cell spacing and no ``offset_w``,
    a :class:`~geoh5py.objects.Grid2D` is returned.  In all other cases
    (explicit triangle meshes, non-uniform grids, or grids with elevation
    offsets) a triangulated :class:`~geoh5py.objects.Surface` is returned.

    Parameters
    ----------
    omf_surface : omf.SurfaceElement
        OMF surface element.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the output object.
    data_names : list[str], optional
        Data channels to include.  When *None*, all are included.
    prefer_grid2d : bool, optional
        When ``True`` (default), convert ``SurfaceGridGeometry`` inputs that
        are compatible with Grid2D directly to a
        :class:`~geoh5py.objects.Grid2D` object, preserving the implicit
        grid structure.  Set to ``False`` to always triangulate and return a
        :class:`~geoh5py.objects.Surface`.

    Returns
    -------
    geoh5py.objects.Surface or geoh5py.objects.Grid2D
    """
    import omf as _omf
    from geoh5py.objects import Surface

    geom = omf_surface.geometry
    obj_name = name or omf_surface.name or "Surface"

    if isinstance(geom, _omf.SurfaceGridGeometry) and prefer_grid2d:
        tensor_u = np.asarray(geom.tensor_u, dtype=float)
        tensor_v = np.asarray(geom.tensor_v, dtype=float)
        has_offset = geom.offset_w is not None
        if _is_uniform(tensor_u) and _is_uniform(tensor_v) and not has_offset:
            return omf_surface_to_grid2d(
                omf_surface, workspace, name=name, data_names=data_names
            )
        # Non-uniform or has offset_w: fall through to triangulation

    if isinstance(geom, _omf.SurfaceGeometry):
        vertices = np.asarray(geom.vertices, dtype=float)
        cells = np.asarray(geom.triangles, dtype=np.uint32)

    elif isinstance(geom, _omf.SurfaceGridGeometry):
        vertices, cells = _grid_geom_to_triangles(geom)
    else:
        raise TypeError(f"Unsupported surface geometry type: {type(geom)}")

    surface = Surface.create(
        workspace, vertices=vertices, cells=cells, name=obj_name
    )

    all_data = _omf_all_data(omf_surface)
    _add_omf_data_to_geoh5(
        surface,
        all_data,
        data_names=data_names,
        location_filter="vertices",
    )

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

    data = _geoh5_children_to_omf_data(surface, data_names=data_names)

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

    all_data = _omf_all_data(volume)
    _add_omf_data_to_geoh5(
        bm,
        all_data,
        data_names=data_names,
        location_filter="cells",
    )

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

    # Collect data from children; BlockModel data lives at "cells"
    data_base = _geoh5_children_to_omf_data(blockmodel, data_names=data_names)
    # Override location to "cells" (geoh5 BlockModel children are all cell data)
    data = []
    for d in data_base:
        d.location = "cells"
        data.append(d)

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
