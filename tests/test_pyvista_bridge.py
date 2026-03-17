"""Tests for geoh5_bridge.pyvista_bridge module."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
from geoh5py.objects import BlockModel, Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace

from geoh5_bridge.pyvista_bridge import (
    blockmodel_to_pyvista,
    curve_to_pyvista,
    grid2d_to_pyvista,
    points_to_pyvista,
    pyvista_to_blockmodel,
    pyvista_to_curve,
    pyvista_to_grid2d,
    pyvista_to_points,
    pyvista_to_surface,
    surface_to_pyvista,
)


@pytest.fixture()
def workspace(tmp_path):
    ws = Workspace.create(str(tmp_path / "test.geoh5"))
    yield ws
    ws.close()


# ------------------------------------------------------------------
# Sample geoh5 objects
# ------------------------------------------------------------------


@pytest.fixture()
def sample_points(workspace):
    """Points with data."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    )
    pts = Points.create(workspace, vertices=vertices, name="SamplePts")
    pts.add_data(
        {"temperature": {"values": np.array([10.0, 20.0, 30.0], dtype=np.float32)}}
    )
    return pts


@pytest.fixture()
def sample_grid2d(workspace):
    """Grid2D with data."""
    grid = Grid2D.create(
        workspace,
        origin=[100.0, 200.0, 0.0],
        u_cell_size=1.0,
        v_cell_size=1.0,
        u_count=5,
        v_count=4,
        name="SampleGrid",
    )
    grid.add_data(
        {"elevation": {"values": np.arange(20, dtype=np.float32)}}
    )
    return grid


@pytest.fixture()
def sample_curve(workspace):
    """Curve with two lines and data."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [4.0, 4.0, 0.0],
        ]
    )
    cells = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.uint32)
    curve = Curve.create(
        workspace, vertices=vertices, cells=cells, name="SampleCurve"
    )
    curve.add_data(
        {"speed": {"values": np.array([50, 50, 50, 80, 80], dtype=np.float32)}}
    )
    return curve


@pytest.fixture()
def sample_surface(workspace):
    """Surface with two triangles and data."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    surf = Surface.create(
        workspace, vertices=vertices, cells=cells, name="SampleSurf"
    )
    surf.add_data(
        {"area": {"values": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)}}
    )
    return surf


@pytest.fixture()
def sample_blockmodel(workspace):
    """BlockModel with data."""
    bm = BlockModel.create(
        workspace,
        origin=[-0.5, -0.5, -0.5],
        u_cell_delimiters=np.array([0.0, 1.0, 2.0, 3.0]),
        v_cell_delimiters=np.array([0.0, 1.0, 2.0]),
        z_cell_delimiters=np.array([0.0, 1.0]),
        name="SampleBM",
    )
    bm.add_data(
        {"density": {"values": np.arange(6, dtype=np.float32)}}
    )
    return bm


# ------------------------------------------------------------------
# Sample PyVista objects
# ------------------------------------------------------------------


@pytest.fixture()
def pv_points():
    """PyVista PolyData point cloud."""
    pts = pv.PolyData(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float))
    pts.point_data["temperature"] = np.array([10.0, 20.0, 30.0])
    return pts


@pytest.fixture()
def pv_structured():
    """PyVista StructuredGrid (flat 2D)."""
    x_nodes = np.arange(6, dtype=float)  # 5 cells
    y_nodes = np.arange(5, dtype=float)  # 4 cells
    xx, yy = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    zz = np.zeros_like(xx)
    sg = pv.StructuredGrid(xx, yy, zz)
    sg.cell_data["elevation"] = np.arange(20, dtype=float)
    return sg


@pytest.fixture()
def pv_lines():
    """PyVista PolyData with lines."""
    vertices = np.array(
        [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 3, 0], [4, 4, 0]], dtype=float
    )
    lines = np.array([3, 0, 1, 2, 2, 3, 4])
    pd = pv.PolyData(vertices, lines=lines)
    pd.point_data["speed"] = np.array([50.0, 50.0, 50.0, 80.0, 80.0])
    return pd


@pytest.fixture()
def pv_triangles():
    """PyVista PolyData triangle mesh."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    faces = np.array([3, 0, 1, 2, 3, 0, 2, 3])
    pd = pv.PolyData(vertices, faces)
    pd.point_data["area"] = np.array([1.0, 1.0, 1.0, 1.0])
    return pd


@pytest.fixture()
def pv_rectilinear():
    """PyVista RectilinearGrid."""
    rg = pv.RectilinearGrid(
        np.array([-0.5, 0.5, 1.5, 2.5]),
        np.array([-0.5, 0.5, 1.5]),
        np.array([-0.5, 0.5]),
    )
    rg.cell_data["density"] = np.arange(6, dtype=float)
    return rg


# ==================================================================
# Points ↔ PyVista
# ==================================================================


class TestPointsToPyvista:
    def test_basic_conversion(self, sample_points):
        pd = points_to_pyvista(sample_points)
        assert pd.n_points == 3
        np.testing.assert_array_almost_equal(
            pd.points, sample_points.vertices
        )

    def test_data_attached(self, sample_points):
        pd = points_to_pyvista(sample_points)
        assert "temperature" in pd.point_data
        np.testing.assert_array_almost_equal(
            pd.point_data["temperature"], [10.0, 20.0, 30.0]
        )

    def test_data_names_filter(self, sample_points):
        pd = points_to_pyvista(sample_points, data_names=["nonexistent"])
        assert "temperature" not in pd.point_data


class TestPyvistaToPoints:
    def test_basic_conversion(self, workspace, pv_points):
        pts = pyvista_to_points(pv_points, workspace, name="FromPV")
        assert len(pts.vertices) == 3
        np.testing.assert_array_almost_equal(pts.vertices, pv_points.points)

    def test_data_attached(self, workspace, pv_points):
        pts = pyvista_to_points(pv_points, workspace, name="FromPV")
        children = {c.name: c.values for c in pts.children if hasattr(c, "values")}
        assert "temperature" in children
        np.testing.assert_array_almost_equal(
            children["temperature"], [10.0, 20.0, 30.0], decimal=4
        )

    def test_roundtrip(self, workspace, sample_points):
        """Points → PyVista → Points preserves data."""
        pd = points_to_pyvista(sample_points)
        pts2 = pyvista_to_points(pd, workspace, name="RT")
        np.testing.assert_array_almost_equal(
            pts2.vertices, sample_points.vertices
        )


# ==================================================================
# Grid2D ↔ PyVista
# ==================================================================


class TestGrid2dToPyvista:
    def test_basic_conversion(self, sample_grid2d):
        sg = grid2d_to_pyvista(sample_grid2d)
        assert isinstance(sg, pv.StructuredGrid)
        assert sg.n_cells == 20

    def test_data_attached(self, sample_grid2d):
        sg = grid2d_to_pyvista(sample_grid2d)
        assert "elevation" in sg.cell_data
        assert len(sg.cell_data["elevation"]) == 20


class TestPyvistaToGrid2d:
    def test_basic_conversion(self, workspace, pv_structured):
        grid = pyvista_to_grid2d(pv_structured, workspace, name="FromPV")
        assert grid.u_count == 5
        assert grid.v_count == 4

    def test_data_attached(self, workspace, pv_structured):
        grid = pyvista_to_grid2d(pv_structured, workspace, name="FromPV")
        children = {c.name: c.values for c in grid.children if hasattr(c, "values")}
        assert "elevation" in children

    def test_roundtrip(self, workspace, sample_grid2d):
        """Grid2D → PyVista → Grid2D preserves shape."""
        sg = grid2d_to_pyvista(sample_grid2d)
        grid2 = pyvista_to_grid2d(sg, workspace, name="RT")
        assert grid2.u_count == sample_grid2d.u_count
        assert grid2.v_count == sample_grid2d.v_count


# ==================================================================
# Curve ↔ PyVista
# ==================================================================


class TestCurveToPyvista:
    def test_basic_conversion(self, sample_curve):
        pd = curve_to_pyvista(sample_curve)
        assert pd.n_points == 5
        assert pd.n_lines == 2

    def test_data_attached(self, sample_curve):
        pd = curve_to_pyvista(sample_curve)
        assert "speed" in pd.point_data
        np.testing.assert_array_almost_equal(
            pd.point_data["speed"], [50, 50, 50, 80, 80]
        )


class TestPyvistaToCurve:
    def test_basic_conversion(self, workspace, pv_lines):
        curve = pyvista_to_curve(pv_lines, workspace, name="FromPV")
        assert len(curve.vertices) == 5
        assert curve.cells.shape == (3, 2)

    def test_data_attached(self, workspace, pv_lines):
        curve = pyvista_to_curve(pv_lines, workspace, name="FromPV")
        children = {c.name: c.values for c in curve.children if hasattr(c, "values")}
        assert "speed" in children

    def test_roundtrip(self, workspace, sample_curve):
        """Curve → PyVista → Curve preserves vertices and cells."""
        pd = curve_to_pyvista(sample_curve)
        curve2 = pyvista_to_curve(pd, workspace, name="RT")
        np.testing.assert_array_almost_equal(
            curve2.vertices, sample_curve.vertices
        )
        np.testing.assert_array_equal(curve2.cells, sample_curve.cells)


# ==================================================================
# Surface ↔ PyVista
# ==================================================================


class TestSurfaceToPyvista:
    def test_basic_conversion(self, sample_surface):
        pd = surface_to_pyvista(sample_surface)
        assert pd.n_points == 4
        assert pd.n_cells == 2
        assert pd.is_all_triangles

    def test_data_attached(self, sample_surface):
        pd = surface_to_pyvista(sample_surface)
        assert "area" in pd.point_data
        np.testing.assert_array_almost_equal(
            pd.point_data["area"], [1.0, 1.0, 1.0, 1.0]
        )


class TestPyvistaToSurface:
    def test_basic_conversion(self, workspace, pv_triangles):
        surf = pyvista_to_surface(pv_triangles, workspace, name="FromPV")
        assert len(surf.vertices) == 4
        assert surf.cells.shape == (2, 3)

    def test_data_attached(self, workspace, pv_triangles):
        surf = pyvista_to_surface(pv_triangles, workspace, name="FromPV")
        children = {c.name: c.values for c in surf.children if hasattr(c, "values")}
        assert "area" in children

    def test_roundtrip(self, workspace, sample_surface):
        """Surface → PyVista → Surface preserves vertices and cells."""
        pd = surface_to_pyvista(sample_surface)
        surf2 = pyvista_to_surface(pd, workspace, name="RT")
        np.testing.assert_array_almost_equal(
            surf2.vertices, sample_surface.vertices
        )
        np.testing.assert_array_equal(surf2.cells, sample_surface.cells)

    def test_non_triangles_raises(self, workspace):
        """Non-triangular faces should raise ValueError."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
        )
        faces = np.array([4, 0, 1, 2, 3])  # Quad face
        pd = pv.PolyData(vertices, faces)
        with pytest.raises(ValueError, match="triangular faces"):
            pyvista_to_surface(pd, workspace)


# ==================================================================
# BlockModel ↔ PyVista
# ==================================================================


class TestBlockModelToPyvista:
    def test_basic_conversion(self, sample_blockmodel):
        rg = blockmodel_to_pyvista(sample_blockmodel)
        assert isinstance(rg, pv.RectilinearGrid)
        assert rg.n_cells == 6

    def test_data_attached(self, sample_blockmodel):
        rg = blockmodel_to_pyvista(sample_blockmodel)
        assert "density" in rg.cell_data
        np.testing.assert_array_almost_equal(
            rg.cell_data["density"], np.arange(6, dtype=float)
        )

    def test_edges(self, sample_blockmodel):
        """Cell edges should match origin + delimiters."""
        rg = blockmodel_to_pyvista(sample_blockmodel)
        np.testing.assert_array_almost_equal(
            rg.x, [-0.5, 0.5, 1.5, 2.5]
        )
        np.testing.assert_array_almost_equal(
            rg.y, [-0.5, 0.5, 1.5]
        )
        np.testing.assert_array_almost_equal(rg.z, [-0.5, 0.5])


class TestPyvistaToBlockModel:
    def test_basic_conversion(self, workspace, pv_rectilinear):
        bm = pyvista_to_blockmodel(
            pv_rectilinear, workspace, name="FromPV"
        )
        assert bm.shape == (3, 2, 1)

    def test_data_attached(self, workspace, pv_rectilinear):
        bm = pyvista_to_blockmodel(
            pv_rectilinear, workspace, name="FromPV"
        )
        children = {c.name: c.values for c in bm.children if hasattr(c, "values")}
        assert "density" in children

    def test_roundtrip(self, workspace, sample_blockmodel):
        """BlockModel → PyVista → BlockModel preserves shape and data."""
        rg = blockmodel_to_pyvista(sample_blockmodel)
        bm2 = pyvista_to_blockmodel(rg, workspace, name="RT")
        assert bm2.shape == sample_blockmodel.shape

        children_orig = {
            c.name: c.values
            for c in sample_blockmodel.children
            if hasattr(c, "values")
        }
        children_new = {
            c.name: c.values
            for c in bm2.children
            if hasattr(c, "values")
        }
        np.testing.assert_array_almost_equal(
            children_new["density"],
            children_orig["density"],
            decimal=4,
        )
