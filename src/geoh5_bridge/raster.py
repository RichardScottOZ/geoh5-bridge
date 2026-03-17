"""Convert rasters (via rioxarray or xarray) to geoh5 Grid2D or Points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoh5_bridge.utils import _add_data_columns

if TYPE_CHECKING:
    import xarray as xr
    from geoh5py.objects import Grid2D, Points
    from geoh5py.workspace import Workspace


def raster_to_grid2d(
    data_array: xr.DataArray,
    workspace: Workspace,
    *,
    name: str = "Raster",
    grid_kwargs: dict | None = None,
) -> Grid2D:
    """Convert an xarray DataArray (2D raster) to a geoh5py Grid2D.

    The DataArray should have exactly two spatial dimensions. When loaded
    via ``rioxarray`` the dimensions are typically ``("y", "x")``. A
    ``"band"`` dimension is also accepted and each band is stored as a
    separate data channel on the resulting Grid2D.

    Parameters
    ----------
    data_array : xarray.DataArray
        2D or multi-band raster. Expected dimension order is
        ``(band, y, x)`` or ``(y, x)``.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace to write into.
    name : str, optional
        Name for the Grid2D object (default ``"Raster"``).
    grid_kwargs : dict, optional
        Extra keyword arguments forwarded to ``Grid2D.create()``.

    Returns
    -------
    geoh5py.objects.Grid2D
        The created Grid2D with data channels attached.

    Examples
    --------
    >>> import rioxarray  # noqa: F401
    >>> import xarray as xr
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import raster_to_grid2d
    >>> da = xr.open_dataarray("elevation.tif", engine="rasterio")
    >>> ws = Workspace.create("output.geoh5")
    >>> grid = raster_to_grid2d(da, ws, name="Elevation")
    """
    from geoh5py.objects import Grid2D

    dims = list(data_array.dims)
    has_band = "band" in dims

    if has_band:
        y_dim, x_dim = [d for d in dims if d != "band"]
    else:
        if len(dims) != 2:
            raise ValueError(
                f"Expected 2 spatial dimensions, got {len(dims)}: {dims}"
            )
        y_dim, x_dim = dims[0], dims[1]

    x_coords = data_array[x_dim].values
    y_coords = data_array[y_dim].values

    nx = len(x_coords)
    ny = len(y_coords)

    dx = float(np.abs(np.diff(x_coords).mean())) if nx > 1 else 1.0
    dy = float(np.abs(np.diff(y_coords).mean())) if ny > 1 else 1.0

    origin = [float(x_coords.min()), float(y_coords.min()), 0.0]

    extra = grid_kwargs or {}
    grid = Grid2D.create(
        workspace,
        origin=origin,
        u_cell_size=dx,
        v_cell_size=dy,
        u_count=nx,
        v_count=ny,
        name=name,
        **extra,
    )

    # Determine whether y-coordinates are descending (typical for rasters)
    y_descending = len(y_coords) > 1 and y_coords[0] > y_coords[-1]

    if has_band:
        bands = data_array["band"].values
        for band_val in bands:
            band_data = data_array.sel(band=band_val).values  # shape (ny, nx)
            if y_descending:
                band_data = band_data[::-1, :]
            band_name = (
                data_array.attrs.get("long_name", f"band_{band_val}")
                if len(bands) > 1
                else name
            )
            _add_data_columns(grid, {str(band_name): band_data.ravel()})
    else:
        values = data_array.values  # shape (ny, nx)
        if y_descending:
            values = values[::-1, :]
        _add_data_columns(grid, {name: values.ravel()})

    return grid


def raster_to_points(
    data_array: xr.DataArray,
    workspace: Workspace,
    *,
    name: str = "RasterPoints",
    nodata: float | None = None,
) -> Points:
    """Convert an xarray DataArray raster to geoh5py Points.

    Every valid raster cell becomes a point with x, y coordinates at the
    cell centre and raster values attached as data channels. Cells that
    match *nodata* are excluded.

    Parameters
    ----------
    data_array : xarray.DataArray
        2D or multi-band raster with ``(band, y, x)`` or ``(y, x)`` dims.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the Points object.
    nodata : float, optional
        Value to treat as missing. Cells with this value in *any* band
        are excluded.  Falls back to ``data_array.rio.nodata`` when
        available.

    Returns
    -------
    geoh5py.objects.Points
    """
    from geoh5py.objects import Points

    dims = list(data_array.dims)
    has_band = "band" in dims

    if has_band:
        y_dim, x_dim = [d for d in dims if d != "band"]
    else:
        if len(dims) != 2:
            raise ValueError(
                f"Expected 2 spatial dimensions, got {len(dims)}: {dims}"
            )
        y_dim, x_dim = dims[0], dims[1]

    x_coords = data_array[x_dim].values
    y_coords = data_array[y_dim].values

    xx, yy = np.meshgrid(x_coords, y_coords)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    if nodata is None:
        try:
            nodata = data_array.rio.nodata
        except AttributeError:
            nodata = None

    if has_band:
        bands = data_array["band"].values
        all_values = {}
        for band_val in bands:
            band_data = data_array.sel(band=band_val).values.ravel()
            band_name = (
                data_array.attrs.get("long_name", f"band_{band_val}")
                if len(bands) > 1
                else name
            )
            all_values[str(band_name)] = band_data
    else:
        all_values = {name: data_array.values.ravel()}

    # Build mask of valid cells
    if nodata is not None:
        mask = np.ones(xx_flat.shape, dtype=bool)
        for vals in all_values.values():
            mask &= ~np.isnan(vals) & (vals != nodata)
    else:
        mask = np.ones(xx_flat.shape, dtype=bool)
        for vals in all_values.values():
            mask &= ~np.isnan(vals)

    vertices = np.column_stack(
        [xx_flat[mask], yy_flat[mask], np.zeros(mask.sum())]
    )

    pts = Points.create(workspace, vertices=vertices, name=name)

    filtered = {k: v[mask] for k, v in all_values.items()}
    _add_data_columns(pts, filtered)

    return pts


def grid2d_to_raster(
    grid: Grid2D,
    *,
    data_names: list[str] | None = None,
) -> xr.DataArray:
    """Convert a geoh5py Grid2D back to an xarray DataArray.

    This is the inverse of :func:`raster_to_grid2d`.  Coordinate arrays
    are reconstructed from the grid origin and cell sizes so that a
    round-trip ``DataArray → Grid2D → DataArray`` preserves coordinate
    values exactly.

    Parameters
    ----------
    grid : geoh5py.objects.Grid2D
        Grid2D object with at least one data channel.
    data_names : list[str], optional
        Names of data channels to include.  When *None* (default) all
        data children that expose a ``values`` attribute are used.

    Returns
    -------
    xarray.DataArray
        2D ``(y, x)`` array when a single channel is present, or 3D
        ``(band, y, x)`` when multiple channels are included.

    Raises
    ------
    ValueError
        If no data channels are found on *grid*.

    Examples
    --------
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import grid2d_to_raster
    >>> ws = Workspace("input.geoh5")
    >>> grid = ws.get_entity("MyGrid")[0]
    >>> da = grid2d_to_raster(grid)
    """
    import xarray as xr

    nx = grid.u_count
    ny = grid.v_count

    origin_x = float(grid.origin["x"])
    origin_y = float(grid.origin["y"])
    dx = float(grid.u_cell_size)
    dy = float(grid.v_cell_size)

    x_coords = origin_x + np.arange(nx) * dx
    y_coords = origin_y + np.arange(ny) * dy

    # Collect data children
    children = {
        c.name: c.values for c in grid.children if hasattr(c, "values")
    }

    if data_names is not None:
        children = {k: v for k, v in children.items() if k in data_names}

    if not children:
        raise ValueError("No data channels found on the Grid2D object.")

    expected = ny * nx
    for ch_name, ch_vals in children.items():
        if np.asarray(ch_vals).size != expected:
            raise ValueError(
                f"Data channel '{ch_name}' has {np.asarray(ch_vals).size} "
                f"values, expected {expected} (ny={ny} × nx={nx})."
            )

    if len(children) == 1:
        name, values = next(iter(children.items()))
        data_2d = np.asarray(values, dtype=float).reshape(ny, nx)
        return xr.DataArray(
            data_2d,
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
            name=name,
        )

    band_names = list(children.keys())
    data_3d = np.stack(
        [
            np.asarray(children[b], dtype=float).reshape(ny, nx)
            for b in band_names
        ],
        axis=0,
    )
    return xr.DataArray(
        data_3d,
        dims=("band", "y", "x"),
        coords={"band": band_names, "x": x_coords, "y": y_coords},
        name="data",
    )
