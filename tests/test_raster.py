"""Tests for geoh5_bridge.raster module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from geoh5py.workspace import Workspace

from geoh5_bridge.raster import raster_to_grid2d, raster_to_points


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
