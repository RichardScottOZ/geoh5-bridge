"""Tests for geoh5_bridge.raster module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from geoh5py.workspace import Workspace

from geoh5_bridge.raster import grid2d_to_raster, raster_to_grid2d, raster_to_points


@pytest.fixture()
def workspace(tmp_path):
    """Create a temporary geoh5 Workspace."""
    ws = Workspace.create(str(tmp_path / "test.geoh5"))
    yield ws
    ws.close()


@pytest.fixture()
def sample_raster():
    """Create a simple 2D raster DataArray."""
    x = np.arange(100, 105, dtype=float)
    y = np.arange(200, 208, dtype=float)[::-1]  # descending (standard raster)
    data = np.arange(len(y) * len(x), dtype=float).reshape(len(y), len(x))
    return xr.DataArray(data, dims=("y", "x"), coords={"x": x, "y": y})


@pytest.fixture()
def multiband_raster():
    """Create a multi-band raster DataArray."""
    x = np.arange(0, 4, dtype=float)
    y = np.arange(0, 3, dtype=float)[::-1]
    bands = [1, 2]
    data = np.random.default_rng(42).random((len(bands), len(y), len(x)))
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": bands, "x": x, "y": y},
    )


class TestRasterToGrid2D:
    def test_basic_conversion(self, workspace, sample_raster):
        grid = raster_to_grid2d(sample_raster, workspace, name="TestGrid")

        assert grid is not None
        assert grid.shape == (5, 8)  # (u_count=nx, v_count=ny)
        assert grid.n_cells == 40
        # Origin should be at min x, min y
        assert grid.origin["x"] == pytest.approx(100.0)
        assert grid.origin["y"] == pytest.approx(200.0)

    def test_data_attached(self, workspace, sample_raster):
        grid = raster_to_grid2d(sample_raster, workspace, name="Elev")
        children = [c for c in grid.children if hasattr(c, "values")]
        assert len(children) == 1
        assert len(children[0].values) == grid.n_cells

    def test_multiband(self, workspace, multiband_raster):
        grid = raster_to_grid2d(
            multiband_raster, workspace, name="MultiBand"
        )
        assert grid.shape == (4, 3)
        children = [c for c in grid.children if hasattr(c, "values")]
        assert len(children) == 2  # Two bands

    def test_ascending_y(self, workspace):
        """Ascending y coordinates should also work."""
        x = np.arange(0, 3, dtype=float)
        y = np.arange(0, 4, dtype=float)  # ascending
        data = np.arange(12, dtype=float).reshape(4, 3)
        da = xr.DataArray(data, dims=("y", "x"), coords={"x": x, "y": y})
        grid = raster_to_grid2d(da, workspace, name="AscY")
        assert grid.shape == (3, 4)

    def test_wrong_ndim_raises(self, workspace):
        da = xr.DataArray(np.ones((2, 3, 4)), dims=("a", "b", "c"))
        with pytest.raises(ValueError, match="Expected 2 spatial dimensions"):
            raster_to_grid2d(da, workspace)


class TestRasterToPoints:
    def test_basic_conversion(self, workspace, sample_raster):
        pts = raster_to_points(sample_raster, workspace, name="TestPts")
        assert pts is not None
        # 5 x 8 = 40 points (no nodata)
        assert len(pts.vertices) == 40

    def test_nodata_excluded(self, workspace):
        x = np.arange(0, 3, dtype=float)
        y = np.arange(0, 2, dtype=float)
        data = np.array([[1.0, 2.0, -999.0], [4.0, 5.0, 6.0]])
        da = xr.DataArray(data, dims=("y", "x"), coords={"x": x, "y": y})
        pts = raster_to_points(da, workspace, name="Filtered", nodata=-999.0)
        assert len(pts.vertices) == 5  # one cell excluded

    def test_nan_excluded(self, workspace):
        x = np.arange(0, 3, dtype=float)
        y = np.arange(0, 2, dtype=float)
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        da = xr.DataArray(data, dims=("y", "x"), coords={"x": x, "y": y})
        pts = raster_to_points(da, workspace, name="NanPts")
        assert len(pts.vertices) == 5

    def test_multiband_points(self, workspace, multiband_raster):
        pts = raster_to_points(
            multiband_raster, workspace, name="BandPts"
        )
        assert pts is not None
        assert len(pts.vertices) == 12  # 4x3 grid, no nodata


class TestGrid2dToRaster:
    def test_basic_roundtrip(self, workspace):
        """Round-trip: DataArray → Grid2D → DataArray preserves coords."""
        x = np.arange(0, 5, dtype=float)
        y = np.arange(0, 4, dtype=float)
        data = np.arange(20, dtype=float).reshape(4, 5)
        da = xr.DataArray(data, dims=("y", "x"), coords={"x": x, "y": y})

        grid = raster_to_grid2d(da, workspace, name="RT")
        result = grid2d_to_raster(grid)

        np.testing.assert_array_almost_equal(result["x"].values, x)
        np.testing.assert_array_almost_equal(result["y"].values, y)

    def test_data_values_preserved(self, workspace):
        """Values survive the round-trip."""
        x = np.arange(0, 3, dtype=float)
        y = np.arange(0, 2, dtype=float)
        data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        da = xr.DataArray(data, dims=("y", "x"), coords={"x": x, "y": y})

        grid = raster_to_grid2d(da, workspace, name="Vals")
        result = grid2d_to_raster(grid)

        np.testing.assert_array_almost_equal(result.values, data, decimal=4)

    def test_multiband_roundtrip(self, workspace, multiband_raster):
        """Multi-band raster round-trips with band dimension."""
        grid = raster_to_grid2d(multiband_raster, workspace, name="MB")
        result = grid2d_to_raster(grid)

        assert "band" in result.dims
        assert result.shape[0] == 2  # 2 bands

    def test_data_names_filter(self, workspace):
        """Selecting specific data channels works."""
        x = np.arange(0, 3, dtype=float)
        y = np.arange(0, 2, dtype=float)
        da = xr.DataArray(
            np.ones((2, 3)), dims=("y", "x"), coords={"x": x, "y": y}
        )
        grid = raster_to_grid2d(da, workspace, name="Filter")
        result = grid2d_to_raster(grid, data_names=["Filter"])
        assert result.name == "Filter"

    def test_no_data_raises(self, workspace):
        """Requesting non-existent channels raises ValueError."""
        x = np.arange(0, 3, dtype=float)
        y = np.arange(0, 2, dtype=float)
        da = xr.DataArray(
            np.ones((2, 3)), dims=("y", "x"), coords={"x": x, "y": y}
        )
        grid = raster_to_grid2d(da, workspace, name="NoData")
        with pytest.raises(ValueError, match="No data channels"):
            grid2d_to_raster(grid, data_names=["nonexistent"])
