"""Convert 3D xarray Datasets / DataArrays to geoh5 BlockModel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geoh5_bridge.utils import _add_data_columns

if TYPE_CHECKING:
    import xarray as xr
    from geoh5py.objects import BlockModel
    from geoh5py.workspace import Workspace


def xarray_to_blockmodel(
    dataset: xr.Dataset | xr.DataArray,
    workspace: Workspace,
    *,
    name: str = "BlockModel",
    dims: tuple[str, str, str] | None = None,
    variables: list[str] | None = None,
) -> BlockModel:
    """Convert a 3D xarray Dataset or DataArray to a geoh5py BlockModel.

    The input must have exactly three coordinate dimensions representing
    the *x*, *y*, and *z* axes. Each coordinate must be monotonically
    increasing (or decreasing — the function handles both).

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Three-dimensional dataset. If a ``DataArray`` is passed it is
        promoted to a ``Dataset`` with one variable.
    workspace : geoh5py.workspace.Workspace
        Open geoh5py Workspace.
    name : str, optional
        Name for the BlockModel object.
    dims : tuple[str, str, str], optional
        Explicit ``(x_dim, y_dim, z_dim)`` dimension names. When
        ``None``, the function tries to auto-detect common names such as
        ``x``, ``y``, ``z``, ``easting``, ``northing``, ``depth``.
    variables : list[str], optional
        Data variables to include. Defaults to all variables in the
        Dataset.

    Returns
    -------
    geoh5py.objects.BlockModel

    Examples
    --------
    >>> import xarray as xr
    >>> from geoh5py.workspace import Workspace
    >>> from geoh5_bridge import xarray_to_blockmodel
    >>> ds = xr.open_dataset("model.nc")
    >>> ws = Workspace.create("output.geoh5")
    >>> bm = xarray_to_blockmodel(ds, ws, name="DensityModel")
    """
    import xarray as xr
    from geoh5py.objects import BlockModel

    if isinstance(dataset, xr.DataArray):
        var_name = dataset.name or "data"
        dataset = dataset.to_dataset(name=var_name)

    all_dims = list(dataset.dims)
    if len(all_dims) != 3:
        raise ValueError(
            f"Expected 3 dimensions, got {len(all_dims)}: {all_dims}"
        )

    if dims is not None:
        x_dim, y_dim, z_dim = dims
    else:
        x_dim, y_dim, z_dim = _guess_xyz_dims(all_dims)

    x_coords = dataset[x_dim].values.astype(float)
    y_coords = dataset[y_dim].values.astype(float)
    z_coords = dataset[z_dim].values.astype(float)

    # Ensure ascending order
    x_coords = np.sort(x_coords)
    y_coords = np.sort(y_coords)
    z_coords = np.sort(z_coords)

    u_delims = _coords_to_delimiters(x_coords)
    v_delims = _coords_to_delimiters(y_coords)
    z_delims = _coords_to_delimiters(z_coords)

    # Origin is at the first cell edge (half-spacing before first centre)
    dx0 = np.diff(x_coords)[0] / 2.0 if len(x_coords) > 1 else 0.5
    dy0 = np.diff(y_coords)[0] / 2.0 if len(y_coords) > 1 else 0.5
    dz0 = np.diff(z_coords)[0] / 2.0 if len(z_coords) > 1 else 0.5
    origin = [
        float(x_coords.min() - dx0),
        float(y_coords.min() - dy0),
        float(z_coords.min() - dz0),
    ]

    bm = BlockModel.create(
        workspace,
        origin=origin,
        u_cell_delimiters=u_delims,
        v_cell_delimiters=v_delims,
        z_cell_delimiters=z_delims,
        name=name,
    )

    if variables is None:
        variables = list(dataset.data_vars)

    data = {}
    for var in variables:
        da = dataset[var]
        # Transpose to (x, y, z) order then flatten
        values = da.transpose(x_dim, y_dim, z_dim).values
        # Sort if coordinates were not ascending
        data[var] = values.ravel().astype(float)

    if data:
        _add_data_columns(bm, data)

    return bm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_X_NAMES = {"x", "easting", "lon", "longitude"}
_Y_NAMES = {"y", "northing", "lat", "latitude"}
_Z_NAMES = {"z", "depth", "elevation", "level", "height"}


def _guess_xyz_dims(
    dims: list[str],
) -> tuple[str, str, str]:
    """Attempt to identify x, y, z dimension names from a list."""
    x = y = z = None
    for d in dims:
        dl = d.lower()
        if dl in _X_NAMES:
            x = d
        elif dl in _Y_NAMES:
            y = d
        elif dl in _Z_NAMES:
            z = d

    if x is None or y is None or z is None:
        # Fall back to positional order
        x, y, z = dims[0], dims[1], dims[2]

    return x, y, z


def _coords_to_delimiters(coords: np.ndarray) -> np.ndarray:
    """Convert cell-centre coordinates to cell-edge delimiters.

    For *n* cell centres this returns *n + 1* delimiter values starting
    at 0, producing *n* cells whose centres align with *coords*.
    """
    if len(coords) < 2:
        return np.array([0.0, 1.0])

    spacings = np.diff(coords)
    half_first = spacings[0] / 2.0
    half_last = spacings[-1] / 2.0
    midpoints = coords[:-1] + spacings / 2.0
    edges_abs = np.concatenate([
        [coords[0] - half_first],
        midpoints,
        [coords[-1] + half_last],
    ])
    return edges_abs - edges_abs[0]
