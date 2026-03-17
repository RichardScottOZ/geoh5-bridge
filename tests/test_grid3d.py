"""Tests for geoh5_bridge.grid3d module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from geoh5py.workspace import Workspace

from geoh5_bridge.grid3d import blockmodel_to_xarray, xarray_to_blockmodel


@pytest.fixture()
def workspace(tmp_path):
    ws = Workspace.create(str(tmp_path / "test.geoh5"))
    yield ws
    ws.close()


@pytest.fixture()
def sample_dataset():
    """Simple 3D xarray Dataset."""
    x = np.arange(0, 5, dtype=float)
    y = np.arange(0, 4, dtype=float)
    z = np.arange(0, 3, dtype=float)
    rng = np.random.default_rng(42)
    density = rng.random((5, 4, 3))
    return xr.Dataset(
        {"density": (("x", "y", "z"), density)},
        coords={"x": x, "y": y, "z": z},
    )


@pytest.fixture()
def dataarray_3d():
    """3D xarray DataArray."""
    x = np.arange(0, 3, dtype=float)
    y = np.arange(0, 4, dtype=float)
    z = np.arange(0, 2, dtype=float)
    data = np.arange(24, dtype=float).reshape(3, 4, 2)
    return xr.DataArray(
        data, dims=("x", "y", "z"), coords={"x": x, "y": y, "z": z}
    )


class TestXarrayToBlockModel:
    def test_basic_conversion(self, workspace, sample_dataset):
        bm = xarray_to_blockmodel(
            sample_dataset, workspace, name="TestBM"
        )
        assert bm is not None
        assert bm.shape == (5, 4, 3)

    def test_data_attached(self, workspace, sample_dataset):
        bm = xarray_to_blockmodel(
            sample_dataset, workspace, name="TestBM"
        )
        children = [c for c in bm.children if hasattr(c, "values")]
        assert len(children) == 1
        assert children[0].values.shape == (60,)  # 5*4*3

    def test_dataarray_input(self, workspace, dataarray_3d):
        bm = xarray_to_blockmodel(
            dataarray_3d, workspace, name="DataArrayBM"
        )
        assert bm.shape == (3, 4, 2)

    def test_origin(self, workspace, sample_dataset):
        bm = xarray_to_blockmodel(
            sample_dataset, workspace, name="Origin"
        )
        # For coords starting at 0 with spacing 1, origin is -0.5
        assert bm.origin["x"] == pytest.approx(-0.5)
        assert bm.origin["y"] == pytest.approx(-0.5)
        assert bm.origin["z"] == pytest.approx(-0.5)

    def test_custom_dims(self, workspace):
        ds = xr.Dataset(
            {"temp": (("easting", "northing", "depth"), np.ones((2, 3, 4)))},
            coords={
                "easting": [0.0, 1.0],
                "northing": [0.0, 1.0, 2.0],
                "depth": [0.0, 1.0, 2.0, 3.0],
            },
        )
        bm = xarray_to_blockmodel(
            ds,
            workspace,
            dims=("easting", "northing", "depth"),
            name="Custom",
        )
        assert bm.shape == (2, 3, 4)

    def test_auto_detect_dims(self, workspace):
        ds = xr.Dataset(
            {"val": (("easting", "northing", "depth"), np.ones((2, 3, 4)))},
            coords={
                "easting": [0.0, 1.0],
                "northing": [0.0, 1.0, 2.0],
                "depth": [0.0, 1.0, 2.0, 3.0],
            },
        )
        bm = xarray_to_blockmodel(ds, workspace, name="Auto")
        assert bm.shape == (2, 3, 4)

    def test_wrong_ndim(self, workspace):
        ds = xr.Dataset(
            {"val": (("x", "y"), np.ones((2, 3)))},
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
        )
        with pytest.raises(ValueError, match="Expected 3 dimensions"):
            xarray_to_blockmodel(ds, workspace)

    def test_select_variables(self, workspace):
        ds = xr.Dataset(
            {
                "a": (("x", "y", "z"), np.ones((2, 2, 2))),
                "b": (("x", "y", "z"), np.zeros((2, 2, 2))),
            },
            coords={
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "z": [0.0, 1.0],
            },
        )
        bm = xarray_to_blockmodel(
            ds, workspace, variables=["a"], name="Selective"
        )
        children = [c for c in bm.children if hasattr(c, "values")]
        assert len(children) == 1


class TestBlockModelToXarray:
    def test_basic_roundtrip(self, workspace, sample_dataset):
        """Round-trip: Dataset → BlockModel → Dataset preserves shape."""
        bm = xarray_to_blockmodel(sample_dataset, workspace, name="RT")
        result = blockmodel_to_xarray(bm)

        assert "density" in result.data_vars
        assert result["density"].shape == (5, 4, 3)

    def test_coords_roundtrip(self, workspace):
        """Coordinates survive the round-trip."""
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([100.0, 200.0])
        z = np.array([0.0, 5.0])
        ds = xr.Dataset(
            {"val": (("x", "y", "z"), np.ones((3, 2, 2)))},
            coords={"x": x, "y": y, "z": z},
        )
        bm = xarray_to_blockmodel(ds, workspace, name="Coords")
        result = blockmodel_to_xarray(bm)

        np.testing.assert_array_almost_equal(result["x"].values, x)
        np.testing.assert_array_almost_equal(result["y"].values, y)
        np.testing.assert_array_almost_equal(result["z"].values, z)

    def test_data_values_preserved(self, workspace):
        """Data values survive the round-trip."""
        rng = np.random.default_rng(99)
        values = rng.random((3, 4, 2))
        ds = xr.Dataset(
            {"temp": (("x", "y", "z"), values)},
            coords={
                "x": np.arange(3, dtype=float),
                "y": np.arange(4, dtype=float),
                "z": np.arange(2, dtype=float),
            },
        )
        bm = xarray_to_blockmodel(ds, workspace, name="DataRT")
        result = blockmodel_to_xarray(bm)

        np.testing.assert_array_almost_equal(
            result["temp"].values, values, decimal=4
        )

    def test_variable_filter(self, workspace):
        """Selecting specific variables works."""
        ds = xr.Dataset(
            {
                "a": (("x", "y", "z"), np.ones((2, 2, 2))),
                "b": (("x", "y", "z"), np.zeros((2, 2, 2))),
            },
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
        )
        bm = xarray_to_blockmodel(ds, workspace, name="VarFilt")
        result = blockmodel_to_xarray(bm, variables=["a"])

        assert "a" in result.data_vars
        assert "b" not in result.data_vars

    def test_custom_dims(self, workspace):
        """Custom dimension names work."""
        ds = xr.Dataset(
            {"val": (("x", "y", "z"), np.ones((2, 3, 4)))},
            coords={
                "x": [0.0, 1.0],
                "y": [0.0, 1.0, 2.0],
                "z": [0.0, 1.0, 2.0, 3.0],
            },
        )
        bm = xarray_to_blockmodel(ds, workspace, name="CustomDim")
        result = blockmodel_to_xarray(
            bm, dims=("easting", "northing", "depth")
        )

        assert "easting" in result.dims
        assert "northing" in result.dims
        assert "depth" in result.dims

    def test_no_data_raises(self, workspace):
        """Requesting non-existent channels raises ValueError."""
        ds = xr.Dataset(
            {"val": (("x", "y", "z"), np.ones((2, 2, 2)))},
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]},
        )
        bm = xarray_to_blockmodel(ds, workspace, name="NoData")
        with pytest.raises(ValueError, match="No data channels"):
            blockmodel_to_xarray(bm, variables=["nonexistent"])
