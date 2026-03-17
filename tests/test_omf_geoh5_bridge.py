"""Tests for geoh5_bridge.omf_geoh5_bridge module (OMF ↔ geoh5)."""

from __future__ import annotations

import numpy as np
import omf
import pytest
from geoh5py.objects import BlockModel, Curve, Points, Surface
from geoh5py.workspace import Workspace

from geoh5_bridge.omf_geoh5_bridge import (
    blockmodel_to_omf_volume,
    curve_to_omf_lineset,
    omf_lineset_to_curve,
    omf_pointset_to_points,
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

    def test_grid_surface_raises(self, workspace):
        """SurfaceGridGeometry should raise TypeError."""
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
        with pytest.raises(TypeError, match="SurfaceGeometry"):
            omf_surface_to_surface(grid_surf, workspace)


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
