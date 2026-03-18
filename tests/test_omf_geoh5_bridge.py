"""Tests for geoh5_bridge.omf_geoh5_bridge module (OMF ↔ geoh5)."""

from __future__ import annotations

import numpy as np
import omf
import pytest
from geoh5py.objects import BlockModel, Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace

from geoh5_bridge import (
    BlockModel as PublicBlockModel,
    Curve as PublicCurve,
    Points as PublicPoints,
    Surface as PublicSurface,
)
from geoh5_bridge.omf_geoh5_bridge import (
    blockmodel_to_omf_volume,
    curve_to_omf_lineset,
    grid2d_to_omf_surface,
    omf_lineset_to_curve,
    omf_pointset_to_points,
    omf_surface_to_grid2d,
    omf_surface_to_surface,
    omf_volume_to_blockmodel,
    points_to_omf_pointset,
    surface_to_omf_surface,
)


@pytest.fixture()
def workspace(tmp_path):
    ws = Workspace.create(str(tmp_path / "test.geoh5"))
    yield ws
    ws.close()


# ------------------------------------------------------------------
# Sample OMF elements
# ------------------------------------------------------------------


@pytest.fixture()
def omf_pointset():
    return omf.PointSetElement(
        name="TestPoints",
        geometry=omf.PointSetGeometry(
            vertices=np.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
            )
        ),
        data=[
            omf.ScalarData(
                name="temperature",
                array=np.array([10.0, 20.0, 30.0]),
                location="vertices",
            )
        ],
    )


@pytest.fixture()
def omf_lineset():
    return omf.LineSetElement(
        name="TestLines",
        geometry=omf.LineSetGeometry(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 3.0, 0.0],
                    [4.0, 4.0, 0.0],
                ]
            ),
            segments=np.array([[0, 1], [1, 2], [3, 4]]),
        ),
        data=[
            omf.ScalarData(
                name="speed",
                array=np.array([50.0, 50.0, 50.0, 80.0, 80.0]),
                location="vertices",
            )
        ],
    )


@pytest.fixture()
def omf_surface():
    return omf.SurfaceElement(
        name="TestSurface",
        geometry=omf.SurfaceGeometry(
            vertices=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            triangles=np.array([[0, 1, 2], [0, 2, 3]]),
        ),
        data=[
            omf.ScalarData(
                name="value",
                array=np.array([1.0, 2.0, 3.0, 4.0]),
                location="vertices",
            )
        ],
    )


@pytest.fixture()
def omf_volume():
    return omf.VolumeElement(
        name="TestVolume",
        geometry=omf.VolumeGridGeometry(
            origin=[-0.5, -0.5, -0.5],
            tensor_u=np.array([1.0, 1.0, 1.0]),
            tensor_v=np.array([1.0, 1.0]),
            tensor_w=np.array([1.0]),
            axis_u=[1.0, 0.0, 0.0],
            axis_v=[0.0, 1.0, 0.0],
            axis_w=[0.0, 0.0, 1.0],
        ),
        data=[
            omf.ScalarData(
                name="density",
                array=np.arange(6, dtype=float),
                location="cells",
            )
        ],
    )


# ------------------------------------------------------------------
# Sample geoh5 objects
# ------------------------------------------------------------------


@pytest.fixture()
def sample_points(workspace):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    )
    pts = Points.create(workspace, vertices=vertices, name="SamplePts")
    pts.add_data(
        {"temperature": {"values": np.array([10.0, 20.0, 30.0], dtype=np.float32)}}
    )
    return pts


@pytest.fixture()
def sample_curve(workspace):
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
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    surf = Surface.create(
        workspace, vertices=vertices, cells=cells, name="SampleSurf"
    )
    surf.add_data(
        {"value": {"values": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}}
    )
    return surf


@pytest.fixture()
def sample_blockmodel(workspace):
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


# ==================================================================
# PointSet ↔ Points
# ==================================================================


class TestOmfPointsetToPoints:
    def test_basic_conversion(self, workspace, omf_pointset):
        pts = omf_pointset_to_points(omf_pointset, workspace)
        assert len(pts.vertices) == 3
        np.testing.assert_array_almost_equal(
            pts.vertices, [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        )

    def test_name_inherited(self, workspace, omf_pointset):
        pts = omf_pointset_to_points(omf_pointset, workspace)
        assert pts.name == "TestPoints"

    def test_name_override(self, workspace, omf_pointset):
        pts = omf_pointset_to_points(
            omf_pointset, workspace, name="CustomName"
        )
        assert pts.name == "CustomName"

    def test_data_attached(self, workspace, omf_pointset):
        pts = omf_pointset_to_points(omf_pointset, workspace)
        children = {
            c.name: c.values for c in pts.children if hasattr(c, "values")
        }
        assert "temperature" in children
        np.testing.assert_array_almost_equal(
            children["temperature"], [10, 20, 30], decimal=4
        )

    def test_data_names_filter(self, workspace, omf_pointset):
        pts = omf_pointset_to_points(
            omf_pointset, workspace, data_names=["nonexistent"]
        )
        children = [c for c in pts.children if hasattr(c, "values")]
        assert len(children) == 0


class TestPointsToOmfPointset:
    def test_basic_conversion(self, sample_points):
        elem = points_to_omf_pointset(sample_points)
        verts = np.asarray(elem.geometry.vertices)
        assert verts.shape == (3, 3)
        assert elem.name == "SamplePts"

    def test_data_attached(self, sample_points):
        elem = points_to_omf_pointset(sample_points)
        assert len(elem.data) == 1
        assert elem.data[0].name == "temperature"

    def test_roundtrip(self, workspace, omf_pointset):
        """OMF → geoh5 → OMF preserves geometry and data."""
        pts = omf_pointset_to_points(omf_pointset, workspace)
        elem2 = points_to_omf_pointset(pts)
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.geometry.vertices),
            np.asarray(omf_pointset.geometry.vertices),
        )
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.data[0].array),
            np.asarray(omf_pointset.data[0].array),
            decimal=4,
        )


# ==================================================================
# LineSet ↔ Curve
# ==================================================================


class TestOmfLinesetToCurve:
    def test_basic_conversion(self, workspace, omf_lineset):
        curve = omf_lineset_to_curve(omf_lineset, workspace)
        assert len(curve.vertices) == 5
        assert curve.cells.shape == (3, 2)

    def test_name_inherited(self, workspace, omf_lineset):
        curve = omf_lineset_to_curve(omf_lineset, workspace)
        assert curve.name == "TestLines"

    def test_data_attached(self, workspace, omf_lineset):
        curve = omf_lineset_to_curve(omf_lineset, workspace)
        children = {
            c.name: c.values for c in curve.children if hasattr(c, "values")
        }
        assert "speed" in children


class TestCurveToOmfLineset:
    def test_basic_conversion(self, sample_curve):
        elem = curve_to_omf_lineset(sample_curve)
        segs = np.asarray(elem.geometry.segments)
        assert segs.shape == (3, 2)
        assert elem.name == "SampleCurve"

    def test_data_attached(self, sample_curve):
        elem = curve_to_omf_lineset(sample_curve)
        assert len(elem.data) == 1
        assert elem.data[0].name == "speed"

    def test_roundtrip(self, workspace, omf_lineset):
        """OMF → geoh5 → OMF preserves segments and data."""
        curve = omf_lineset_to_curve(omf_lineset, workspace)
        elem2 = curve_to_omf_lineset(curve)
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.geometry.vertices),
            np.asarray(omf_lineset.geometry.vertices),
        )
        np.testing.assert_array_equal(
            np.asarray(elem2.geometry.segments),
            np.asarray(omf_lineset.geometry.segments),
        )


# ==================================================================
# Surface ↔ Surface
# ==================================================================


class TestOmfSurfaceToSurface:
    def test_basic_conversion(self, workspace, omf_surface):
        surf = omf_surface_to_surface(omf_surface, workspace)
        assert len(surf.vertices) == 4
        assert surf.cells.shape == (2, 3)

    def test_name_inherited(self, workspace, omf_surface):
        surf = omf_surface_to_surface(omf_surface, workspace)
        assert surf.name == "TestSurface"

    def test_data_attached(self, workspace, omf_surface):
        surf = omf_surface_to_surface(omf_surface, workspace)
        children = {
            c.name: c.values for c in surf.children if hasattr(c, "values")
        }
        assert "value" in children

    def test_grid_surface_basic(self, workspace):
        """SurfaceGridGeometry with uniform spacing → Grid2D by default."""
        grid_surf = omf.SurfaceElement(
            name="GridSurf",
            geometry=omf.SurfaceGridGeometry(
                origin=[0.0, 0.0, 0.0],
                tensor_u=np.array([1.0, 1.0]),
                tensor_v=np.array([1.0]),
                axis_u=[1.0, 0.0, 0.0],
                axis_v=[0.0, 1.0, 0.0],
            ),
        )
        result = omf_surface_to_surface(grid_surf, workspace)
        # With prefer_grid2d=True (default), uniform grid → Grid2D
        assert isinstance(result, Grid2D)
        assert result.name == "GridSurf"
        # 2 u-cells × 1 v-cell
        assert result.u_count == 2
        assert result.v_count == 1

    def test_grid_surface_basic_triangulated(self, workspace):
        """SurfaceGridGeometry with prefer_grid2d=False → triangulated Surface."""
        grid_surf = omf.SurfaceElement(
            name="GridSurf",
            geometry=omf.SurfaceGridGeometry(
                origin=[0.0, 0.0, 0.0],
                tensor_u=np.array([1.0, 1.0]),
                tensor_v=np.array([1.0]),
                axis_u=[1.0, 0.0, 0.0],
                axis_v=[0.0, 1.0, 0.0],
            ),
        )
        surf = omf_surface_to_surface(grid_surf, workspace, prefer_grid2d=False)
        assert isinstance(surf, Surface)
        # 3 u-nodes × 2 v-nodes = 6 vertices
        assert len(surf.vertices) == 6
        # 2 u-cells × 1 v-cell × 2 triangles = 4 triangles
        assert surf.cells.shape == (4, 3)

    def test_grid_surface_with_data(self, workspace):
        """SurfaceGridGeometry with vertex data."""
        grid_surf = omf.SurfaceElement(
            name="GridData",
            geometry=omf.SurfaceGridGeometry(
                origin=[0.0, 0.0, 0.0],
                tensor_u=np.array([1.0, 1.0]),
                tensor_v=np.array([1.0]),
                axis_u=[1.0, 0.0, 0.0],
                axis_v=[0.0, 1.0, 0.0],
            ),
            data=[
                omf.ScalarData(
                    name="elev",
                    array=np.arange(6, dtype=float),
                    location="vertices",
                )
            ],
        )
        surf = omf_surface_to_surface(grid_surf, workspace)
        children = {
            c.name: c.values for c in surf.children if hasattr(c, "values")
        }
        assert "elev" in children
        np.testing.assert_array_almost_equal(
            children["elev"], np.arange(6, dtype=float), decimal=4
        )

    def test_grid_surface_with_offset(self, workspace):
        """SurfaceGridGeometry with offset_w."""
        offsets = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        grid_surf = omf.SurfaceElement(
            name="GridOffset",
            geometry=omf.SurfaceGridGeometry(
                origin=[0.0, 0.0, 0.0],
                tensor_u=np.array([1.0, 1.0]),
                tensor_v=np.array([1.0]),
                axis_u=[1.0, 0.0, 0.0],
                axis_v=[0.0, 1.0, 0.0],
                offset_w=offsets,
            ),
        )
        surf = omf_surface_to_surface(grid_surf, workspace)
        assert len(surf.vertices) == 6
        # Verify z-values match offsets
        np.testing.assert_array_almost_equal(
            surf.vertices[:, 2], offsets
        )


class TestSurfaceToOmfSurface:
    def test_basic_conversion(self, sample_surface):
        elem = surface_to_omf_surface(sample_surface)
        tris = np.asarray(elem.geometry.triangles)
        assert tris.shape == (2, 3)
        assert elem.name == "SampleSurf"

    def test_data_attached(self, sample_surface):
        elem = surface_to_omf_surface(sample_surface)
        assert len(elem.data) == 1
        assert elem.data[0].name == "value"

    def test_roundtrip(self, workspace, omf_surface):
        """OMF → geoh5 → OMF preserves triangles and data."""
        surf = omf_surface_to_surface(omf_surface, workspace)
        elem2 = surface_to_omf_surface(surf)
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.geometry.vertices),
            np.asarray(omf_surface.geometry.vertices),
        )
        np.testing.assert_array_equal(
            np.asarray(elem2.geometry.triangles),
            np.asarray(omf_surface.geometry.triangles),
        )


# ==================================================================
# SurfaceGridGeometry ↔ Grid2D
# ==================================================================


@pytest.fixture()
def omf_grid_surface():
    return omf.SurfaceElement(
        name="GridSurf",
        geometry=omf.SurfaceGridGeometry(
            origin=[1.0, 2.0, 0.0],
            tensor_u=np.array([1.0, 1.0, 1.0]),
            tensor_v=np.array([2.0, 2.0]),
            axis_u=[1.0, 0.0, 0.0],
            axis_v=[0.0, 1.0, 0.0],
        ),
        data=[
            omf.ScalarData(
                name="elev",
                array=np.arange(6, dtype=float),
                location="cells",
            )
        ],
    )


class TestOmfSurfaceToGrid2D:
    def test_returns_grid2d(self, workspace, omf_grid_surface):
        result = omf_surface_to_grid2d(omf_grid_surface, workspace)
        assert isinstance(result, Grid2D)

    def test_shape(self, workspace, omf_grid_surface):
        result = omf_surface_to_grid2d(omf_grid_surface, workspace)
        # tensor_u has 3 cells, tensor_v has 2 cells
        assert result.u_count == 3
        assert result.v_count == 2

    def test_cell_size(self, workspace, omf_grid_surface):
        result = omf_surface_to_grid2d(omf_grid_surface, workspace)
        assert result.u_cell_size == pytest.approx(1.0)
        assert result.v_cell_size == pytest.approx(2.0)

    def test_origin(self, workspace, omf_grid_surface):
        result = omf_surface_to_grid2d(omf_grid_surface, workspace)
        assert float(result.origin["x"]) == pytest.approx(1.0)
        assert float(result.origin["y"]) == pytest.approx(2.0)
        assert float(result.origin["z"]) == pytest.approx(0.0)

    def test_rotation(self, workspace, omf_grid_surface):
        result = omf_surface_to_grid2d(omf_grid_surface, workspace)
        assert result.rotation == pytest.approx(0.0)

    def test_name_inherited(self, workspace, omf_grid_surface):
        result = omf_surface_to_grid2d(omf_grid_surface, workspace)
        assert result.name == "GridSurf"

    def test_name_override(self, workspace, omf_grid_surface):
        result = omf_surface_to_grid2d(omf_grid_surface, workspace, name="Custom")
        assert result.name == "Custom"

    def test_data_attached(self, workspace, omf_grid_surface):
        result = omf_surface_to_grid2d(omf_grid_surface, workspace)
        names = [c.name for c in result.children if hasattr(c, "values")]
        assert "elev" in names

    def test_non_grid_raises_type_error(self, workspace, omf_surface):
        """SurfaceGeometry (not a grid) should raise TypeError."""
        with pytest.raises(TypeError, match="SurfaceGridGeometry"):
            omf_surface_to_grid2d(omf_surface, workspace)

    def test_non_uniform_u_raises(self, workspace):
        surf = omf.SurfaceElement(
            name="NonUniform",
            geometry=omf.SurfaceGridGeometry(
                origin=[0, 0, 0],
                tensor_u=np.array([1.0, 2.0]),  # non-uniform
                tensor_v=np.array([1.0]),
                axis_u=[1, 0, 0],
                axis_v=[0, 1, 0],
            ),
        )
        with pytest.raises(ValueError, match="uniform"):
            omf_surface_to_grid2d(surf, workspace)

    def test_offset_w_raises(self, workspace):
        surf = omf.SurfaceElement(
            name="WithOffset",
            geometry=omf.SurfaceGridGeometry(
                origin=[0, 0, 0],
                tensor_u=np.array([1.0, 1.0]),
                tensor_v=np.array([1.0]),
                axis_u=[1, 0, 0],
                axis_v=[0, 1, 0],
                offset_w=np.array([0.0, 0.1, 0.2, 0.0, 0.1, 0.2]),
            ),
        )
        with pytest.raises(ValueError, match="offset_w"):
            omf_surface_to_grid2d(surf, workspace)

    def test_prefer_grid2d_default_returns_grid2d(self, workspace, omf_grid_surface):
        """omf_surface_to_surface prefer_grid2d=True (default) → Grid2D."""
        result = omf_surface_to_surface(omf_grid_surface, workspace)
        assert isinstance(result, Grid2D)

    def test_prefer_grid2d_false_returns_surface(self, workspace, omf_grid_surface):
        """omf_surface_to_surface prefer_grid2d=False → triangulated Surface."""
        result = omf_surface_to_surface(omf_grid_surface, workspace, prefer_grid2d=False)
        assert isinstance(result, Surface)

    def test_prefer_grid2d_with_offset_falls_back(self, workspace):
        """Offset-w grid with prefer_grid2d=True falls back to Surface."""
        surf = omf.SurfaceElement(
            name="WithOffset",
            geometry=omf.SurfaceGridGeometry(
                origin=[0, 0, 0],
                tensor_u=np.array([1.0, 1.0]),
                tensor_v=np.array([1.0]),
                axis_u=[1, 0, 0],
                axis_v=[0, 1, 0],
                offset_w=np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]),
            ),
        )
        # prefer_grid2d=True but has offset_w → falls back to Surface
        result = omf_surface_to_surface(surf, workspace)
        assert isinstance(result, Surface)

    def test_rotated_grid(self, workspace):
        """SurfaceGridGeometry with 45° rotation."""
        import math
        surf = omf.SurfaceElement(
            name="Rotated",
            geometry=omf.SurfaceGridGeometry(
                origin=[0, 0, 0],
                tensor_u=np.array([1.0, 1.0]),
                tensor_v=np.array([1.0]),
                axis_u=[math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0],
                axis_v=[-math.sin(math.radians(45)), math.cos(math.radians(45)), 0.0],
            ),
        )
        result = omf_surface_to_grid2d(surf, workspace)
        assert isinstance(result, Grid2D)
        assert result.rotation == pytest.approx(45.0, abs=1e-4)


class TestGrid2DToOmfSurface:
    @pytest.fixture()
    def sample_grid2d(self, workspace):
        g = Grid2D.create(
            workspace,
            origin=[1.0, 2.0, 3.0],
            u_cell_size=1.0,
            v_cell_size=2.0,
            u_count=3,
            v_count=2,
            rotation=0.0,
            dip=0.0,
            name="SampleGrid",
        )
        g.add_data({"elevation": {"values": np.arange(6, dtype=np.float32)}})
        return g

    def test_returns_surface_element(self, sample_grid2d):
        elem = grid2d_to_omf_surface(sample_grid2d)
        import omf as _omf
        assert isinstance(elem, _omf.SurfaceElement)

    def test_geometry_type(self, sample_grid2d):
        elem = grid2d_to_omf_surface(sample_grid2d)
        import omf as _omf
        assert isinstance(elem.geometry, _omf.SurfaceGridGeometry)

    def test_tensor_u(self, sample_grid2d):
        elem = grid2d_to_omf_surface(sample_grid2d)
        tu = np.asarray(elem.geometry.tensor_u)
        assert len(tu) == 3
        np.testing.assert_array_almost_equal(tu, [1.0, 1.0, 1.0])

    def test_tensor_v(self, sample_grid2d):
        elem = grid2d_to_omf_surface(sample_grid2d)
        tv = np.asarray(elem.geometry.tensor_v)
        assert len(tv) == 2
        np.testing.assert_array_almost_equal(tv, [2.0, 2.0])

    def test_origin(self, sample_grid2d):
        elem = grid2d_to_omf_surface(sample_grid2d)
        np.testing.assert_array_almost_equal(elem.geometry.origin, [1.0, 2.0, 3.0])

    def test_name_inherited(self, sample_grid2d):
        elem = grid2d_to_omf_surface(sample_grid2d)
        assert elem.name == "SampleGrid"

    def test_data_attached(self, sample_grid2d):
        elem = grid2d_to_omf_surface(sample_grid2d)
        assert len(elem.data) >= 1
        names = [d.name for d in elem.data]
        assert "elevation" in names

    def test_roundtrip(self, workspace):
        """OMF SurfaceGridGeometry → Grid2D → OMF SurfaceGridGeometry preserves shape."""
        orig = omf.SurfaceElement(
            name="RT",
            geometry=omf.SurfaceGridGeometry(
                origin=[0.0, 0.0, 0.0],
                tensor_u=np.array([2.0, 2.0]),
                tensor_v=np.array([3.0, 3.0, 3.0]),
                axis_u=[1.0, 0.0, 0.0],
                axis_v=[0.0, 1.0, 0.0],
            ),
        )
        grid = omf_surface_to_grid2d(orig, workspace)
        back = grid2d_to_omf_surface(grid)
        np.testing.assert_array_almost_equal(
            np.asarray(back.geometry.tensor_u), [2.0, 2.0]
        )
        np.testing.assert_array_almost_equal(
            np.asarray(back.geometry.tensor_v), [3.0, 3.0, 3.0]
        )


# ==================================================================
# Volume ↔ BlockModel
# ==================================================================


class TestOmfVolumeToBlockmodel:
    def test_basic_conversion(self, workspace, omf_volume):
        bm = omf_volume_to_blockmodel(omf_volume, workspace)
        assert bm.shape == (3, 2, 1)

    def test_origin(self, workspace, omf_volume):
        bm = omf_volume_to_blockmodel(omf_volume, workspace)
        assert bm.origin["x"] == pytest.approx(-0.5)
        assert bm.origin["y"] == pytest.approx(-0.5)
        assert bm.origin["z"] == pytest.approx(-0.5)

    def test_name_inherited(self, workspace, omf_volume):
        bm = omf_volume_to_blockmodel(omf_volume, workspace)
        assert bm.name == "TestVolume"

    def test_data_attached(self, workspace, omf_volume):
        bm = omf_volume_to_blockmodel(omf_volume, workspace)
        children = {
            c.name: c.values for c in bm.children if hasattr(c, "values")
        }
        assert "density" in children
        np.testing.assert_array_almost_equal(
            children["density"], np.arange(6, dtype=float), decimal=4
        )


class TestBlockmodelToOmfVolume:
    def test_basic_conversion(self, sample_blockmodel):
        elem = blockmodel_to_omf_volume(sample_blockmodel)
        assert elem.name == "SampleBM"
        np.testing.assert_array_almost_equal(
            np.asarray(elem.geometry.tensor_u), [1, 1, 1]
        )
        np.testing.assert_array_almost_equal(
            np.asarray(elem.geometry.tensor_v), [1, 1]
        )
        np.testing.assert_array_almost_equal(
            np.asarray(elem.geometry.tensor_w), [1]
        )

    def test_origin(self, sample_blockmodel):
        elem = blockmodel_to_omf_volume(sample_blockmodel)
        np.testing.assert_array_almost_equal(
            np.asarray(elem.geometry.origin), [-0.5, -0.5, -0.5]
        )

    def test_data_attached(self, sample_blockmodel):
        elem = blockmodel_to_omf_volume(sample_blockmodel)
        assert len(elem.data) == 1
        assert elem.data[0].name == "density"
        assert elem.data[0].location == "cells"

    def test_roundtrip(self, workspace, omf_volume):
        """OMF → geoh5 → OMF preserves geometry and data."""
        bm = omf_volume_to_blockmodel(omf_volume, workspace)
        elem2 = blockmodel_to_omf_volume(bm)
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.geometry.origin),
            np.asarray(omf_volume.geometry.origin),
        )
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.geometry.tensor_u),
            np.asarray(omf_volume.geometry.tensor_u),
        )
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.data[0].array),
            np.asarray(omf_volume.data[0].array),
            decimal=4,
        )


# ==================================================================
# Extended data-type tests (OMF ↔ geoh5)
# ==================================================================


class TestOmfExtendedDataTypes:
    """Test that non-scalar OMF data types survive OMF → geoh5 conversion."""

    def test_string_data_omf_to_geoh5(self, workspace):
        """StringData → geoh5 TextData."""
        from geoh5py.data import TextData

        elem = omf.PointSetElement(
            name="P",
            geometry=omf.PointSetGeometry(
                vertices=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
            ),
            data=[
                omf.StringData(
                    name="rock_type",
                    array=["granite", "basalt", "granite"],
                    location="vertices",
                )
            ],
        )
        pts = omf_pointset_to_points(elem, workspace)
        children = {c.name: c for c in pts.children if hasattr(c, "values")}
        assert "rock_type" in children
        assert isinstance(children["rock_type"], TextData)
        assert list(children["rock_type"].values) == ["granite", "basalt", "granite"]

    def test_integer_data_omf_to_geoh5(self, workspace):
        """ScalarData with integer array → geoh5 IntegerData."""
        from geoh5py.data import IntegerData

        elem = omf.PointSetElement(
            name="P",
            geometry=omf.PointSetGeometry(
                vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
            ),
            data=[
                omf.ScalarData(
                    name="id",
                    array=np.array([1, 2], dtype=np.int32),
                    location="vertices",
                )
            ],
        )
        pts = omf_pointset_to_points(elem, workspace)
        children = {c.name: c for c in pts.children if hasattr(c, "values")}
        assert "id" in children
        assert isinstance(children["id"], IntegerData)

    def test_vector3_data_omf_to_geoh5(self, workspace):
        """Vector3Data → three component FloatData channels."""
        from geoh5py.data import FloatData

        elem = omf.PointSetElement(
            name="P",
            geometry=omf.PointSetGeometry(
                vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
            ),
            data=[
                omf.Vector3Data(
                    name="velocity",
                    array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    location="vertices",
                )
            ],
        )
        pts = omf_pointset_to_points(elem, workspace)
        children = {c.name: c for c in pts.children if hasattr(c, "values")}
        for suffix in ("_x", "_y", "_z"):
            assert "velocity" + suffix in children
            assert isinstance(children["velocity" + suffix], FloatData)
        np.testing.assert_array_almost_equal(
            children["velocity_x"].values, [1.0, 4.0], decimal=4
        )

    def test_mapped_data_omf_to_geoh5(self, workspace):
        """MappedData → geoh5 ReferencedData with value_map."""
        from geoh5py.data import ReferencedData

        leg = omf.Legend(values=["rock", "soil", "sand"])
        elem = omf.PointSetElement(
            name="P",
            geometry=omf.PointSetGeometry(
                vertices=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
            ),
            data=[
                omf.MappedData(
                    name="lithology",
                    array=np.array([0, 1, 2]),
                    location="vertices",
                    legends=[leg],
                )
            ],
        )
        pts = omf_pointset_to_points(elem, workspace)
        children = {c.name: c for c in pts.children if hasattr(c, "values")}
        assert "lithology" in children
        assert isinstance(children["lithology"], ReferencedData)
        vm = children["lithology"].entity_type.value_map.map
        labels = {int(k): (v.decode() if isinstance(v, bytes) else v) for k, v in vm if int(k) != 0}
        assert labels[1] == "rock"
        assert labels[2] == "soil"
        assert labels[3] == "sand"


class TestGeoh5ExtendedDataTypesRoundtrip:
    """Test that non-float geoh5 data types round-trip back to OMF."""

    def test_text_data_roundtrip(self, workspace):
        """TextData → OMF StringData."""
        from geoh5py.objects import Points

        pts = Points.create(
            workspace,
            vertices=np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
            name="P",
        )
        pts.add_data({"label": {"values": np.array(["a", "b"])}})
        elem = points_to_omf_pointset(pts)
        types = {d.name: type(d).__name__ for d in elem.data}
        assert "label" in types
        assert types["label"] == "StringData"
        assert list(elem.data[0].array) == ["a", "b"]

    def test_referenced_data_roundtrip(self, workspace):
        """ReferencedData → OMF MappedData."""
        from geoh5py.objects import Points

        pts = Points.create(
            workspace,
            vertices=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float),
            name="P",
        )
        pts.add_data(
            {
                "lith": {
                    "type": "referenced",
                    "values": np.array([1, 2, 1], dtype=np.int32),
                    "value_map": {1: "rock", 2: "soil"},
                }
            }
        )
        elem = points_to_omf_pointset(pts)
        types = {d.name: type(d).__name__ for d in elem.data}
        assert "lith" in types
        assert types["lith"] == "MappedData"
        indices = np.asarray(elem.data[0].array)
        legend = list(elem.data[0].legends[0].values)
        # 1-based geoh5 [1,2,1] → 0-based OMF [0,1,0]
        assert legend[indices[0]] == "rock"
        assert legend[indices[1]] == "soil"


def test_readme_reverse_example_public_types_and_dispatch(
    sample_points, sample_curve, sample_surface, sample_blockmodel
):
    assert PublicPoints is Points
    assert PublicCurve is Curve
    assert PublicSurface is Surface
    assert PublicBlockModel is BlockModel

    elements = []
    for obj in [sample_points, sample_curve, sample_surface, sample_blockmodel]:
        if isinstance(obj, PublicBlockModel):
            elem = blockmodel_to_omf_volume(obj)
        elif isinstance(obj, PublicSurface):
            elem = surface_to_omf_surface(obj)
        elif isinstance(obj, PublicCurve):
            elem = curve_to_omf_lineset(obj)
        elif isinstance(obj, PublicPoints):
            elem = points_to_omf_pointset(obj)
        elements.append(elem)

    assert isinstance(elements[0], omf.PointSetElement)
    assert isinstance(elements[1], omf.LineSetElement)
    assert isinstance(elements[2], omf.SurfaceElement)
    assert isinstance(elements[3], omf.VolumeElement)


class TestOmfGeoh5FileRoundtrip:
    """Test full OMF file → geoh5 workspace → OMF file round-trip."""

    def test_file_roundtrip(
        self, tmp_path, omf_pointset, omf_lineset, omf_surface, omf_volume
    ):
        # Save OMF project
        proj = omf.Project(
            name="FileRT",
            elements=[omf_pointset, omf_lineset, omf_surface, omf_volume],
        )
        omf_path = str(tmp_path / "original.omf")
        omf.OMFWriter(proj, omf_path)

        # Read OMF → geoh5
        reader = omf.OMFReader(omf_path)
        proj_in = reader.get_project()

        ws = Workspace.create(str(tmp_path / "converted.geoh5"))
        geoh5_objects = []
        for elem in proj_in.elements:
            if isinstance(elem, omf.PointSetElement):
                geoh5_objects.append(
                    omf_pointset_to_points(elem, ws)
                )
            elif isinstance(elem, omf.LineSetElement):
                geoh5_objects.append(
                    omf_lineset_to_curve(elem, ws)
                )
            elif isinstance(elem, omf.SurfaceElement):
                geoh5_objects.append(
                    omf_surface_to_surface(elem, ws)
                )
            elif isinstance(elem, omf.VolumeElement):
                geoh5_objects.append(
                    omf_volume_to_blockmodel(elem, ws)
                )

        assert len(geoh5_objects) == 4

        # geoh5 → OMF
        elements_back = []
        for obj in geoh5_objects:
            if isinstance(obj, Points):
                elements_back.append(points_to_omf_pointset(obj))
            elif isinstance(obj, Curve):
                elements_back.append(curve_to_omf_lineset(obj))
            elif isinstance(obj, Surface):
                elements_back.append(surface_to_omf_surface(obj))
            elif isinstance(obj, BlockModel):
                elements_back.append(blockmodel_to_omf_volume(obj))

        proj_out = omf.Project(name="FileRT2", elements=elements_back)
        omf_path2 = str(tmp_path / "roundtrip.omf")
        omf.OMFWriter(proj_out, omf_path2)

        # Verify
        reader2 = omf.OMFReader(omf_path2)
        proj_check = reader2.get_project()
        assert len(proj_check.elements) == 4
        names = sorted(e.name for e in proj_check.elements)
        expected = sorted(e.name for e in proj.elements)
        assert names == expected

        ws.close()
