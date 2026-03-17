"""Tests for geoh5_bridge.omf_bridge module (OMF ↔ PyVista)."""

from __future__ import annotations

import numpy as np
import omf
import pytest
import pyvista as pv

from geoh5_bridge.omf_bridge import (
    omf_lineset_to_pyvista,
    omf_pointset_to_pyvista,
    omf_project_to_pyvista,
    omf_surface_to_pyvista,
    omf_volume_to_pyvista,
    pyvista_to_omf_lineset,
    pyvista_to_omf_pointset,
    pyvista_to_omf_project,
    pyvista_to_omf_surface,
    pyvista_to_omf_volume,
)


# ------------------------------------------------------------------
# Sample OMF elements
# ------------------------------------------------------------------


@pytest.fixture()
def omf_pointset():
    """OMF PointSetElement with data."""
    return omf.PointSetElement(
        name="SamplePoints",
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
    """OMF LineSetElement with two lines and data."""
    return omf.LineSetElement(
        name="SampleLines",
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
    """OMF SurfaceElement (triangle mesh) with data."""
    return omf.SurfaceElement(
        name="SampleSurface",
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
    """OMF VolumeElement with data."""
    return omf.VolumeElement(
        name="SampleVolume",
        geometry=omf.VolumeGridGeometry(
            origin=[0.0, 0.0, 0.0],
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
# Sample PyVista objects
# ------------------------------------------------------------------


@pytest.fixture()
def pv_points():
    """PyVista PolyData point cloud."""
    pts = pv.PolyData(
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    )
    pts.point_data["temperature"] = np.array([10.0, 20.0, 30.0])
    return pts


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
    pd.point_data["value"] = np.array([1.0, 2.0, 3.0, 4.0])
    return pd


@pytest.fixture()
def pv_rectilinear():
    """PyVista RectilinearGrid."""
    rg = pv.RectilinearGrid(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0]),
    )
    rg.cell_data["density"] = np.arange(6, dtype=float)
    return rg


# ==================================================================
# PointSet ↔ PyVista
# ==================================================================


class TestOmfPointsetToPyvista:
    def test_basic_conversion(self, omf_pointset):
        pd = omf_pointset_to_pyvista(omf_pointset)
        assert pd.n_points == 3
        np.testing.assert_array_almost_equal(
            pd.points,
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        )

    def test_data_attached(self, omf_pointset):
        pd = omf_pointset_to_pyvista(omf_pointset)
        assert "temperature" in pd.point_data
        np.testing.assert_array_almost_equal(
            pd.point_data["temperature"], [10, 20, 30]
        )


class TestPyvistaToOmfPointset:
    def test_basic_conversion(self, pv_points):
        elem = pyvista_to_omf_pointset(pv_points, name="TestPts")
        assert elem.name == "TestPts"
        verts = np.asarray(elem.geometry.vertices)
        assert verts.shape == (3, 3)

    def test_data_attached(self, pv_points):
        elem = pyvista_to_omf_pointset(pv_points)
        assert len(elem.data) == 1
        assert elem.data[0].name == "temperature"

    def test_roundtrip(self, omf_pointset):
        """OMF → PyVista → OMF preserves geometry and data."""
        pd = omf_pointset_to_pyvista(omf_pointset)
        elem2 = pyvista_to_omf_pointset(pd, name="RT")
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.geometry.vertices),
            np.asarray(omf_pointset.geometry.vertices),
        )
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.data[0].array),
            np.asarray(omf_pointset.data[0].array),
        )

    def test_data_names_filter(self, pv_points):
        elem = pyvista_to_omf_pointset(
            pv_points, data_names=["nonexistent"]
        )
        assert len(elem.data) == 0


# ==================================================================
# LineSet ↔ PyVista
# ==================================================================


class TestOmfLinesetToPyvista:
    def test_basic_conversion(self, omf_lineset):
        pd = omf_lineset_to_pyvista(omf_lineset)
        assert pd.n_points == 5
        assert pd.n_lines == 2  # two polylines

    def test_data_attached(self, omf_lineset):
        pd = omf_lineset_to_pyvista(omf_lineset)
        assert "speed" in pd.point_data
        np.testing.assert_array_almost_equal(
            pd.point_data["speed"], [50, 50, 50, 80, 80]
        )


class TestPyvistaToOmfLineset:
    def test_basic_conversion(self, pv_lines):
        elem = pyvista_to_omf_lineset(pv_lines, name="TestLines")
        assert elem.name == "TestLines"
        segs = np.asarray(elem.geometry.segments)
        assert segs.shape == (3, 2)

    def test_data_attached(self, pv_lines):
        elem = pyvista_to_omf_lineset(pv_lines)
        assert len(elem.data) == 1
        assert elem.data[0].name == "speed"

    def test_roundtrip(self, omf_lineset):
        """OMF → PyVista → OMF preserves segments and data."""
        pd = omf_lineset_to_pyvista(omf_lineset)
        elem2 = pyvista_to_omf_lineset(pd, name="RT")
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.geometry.vertices),
            np.asarray(omf_lineset.geometry.vertices),
        )
        np.testing.assert_array_equal(
            np.asarray(elem2.geometry.segments),
            np.asarray(omf_lineset.geometry.segments),
        )

    def test_data_names_filter(self, pv_lines):
        elem = pyvista_to_omf_lineset(
            pv_lines, data_names=["nonexistent"]
        )
        assert len(elem.data) == 0


# ==================================================================
# Surface ↔ PyVista
# ==================================================================


class TestOmfSurfaceToPyvista:
    def test_basic_conversion(self, omf_surface):
        pd = omf_surface_to_pyvista(omf_surface)
        assert pd.n_points == 4
        assert pd.n_cells == 2
        assert pd.is_all_triangles

    def test_data_attached(self, omf_surface):
        pd = omf_surface_to_pyvista(omf_surface)
        assert "value" in pd.point_data
        np.testing.assert_array_almost_equal(
            pd.point_data["value"], [1, 2, 3, 4]
        )


class TestPyvistaToOmfSurface:
    def test_basic_conversion(self, pv_triangles):
        elem = pyvista_to_omf_surface(pv_triangles, name="TestSurf")
        assert elem.name == "TestSurf"
        tris = np.asarray(elem.geometry.triangles)
        assert tris.shape == (2, 3)

    def test_data_attached(self, pv_triangles):
        elem = pyvista_to_omf_surface(pv_triangles)
        assert len(elem.data) == 1
        assert elem.data[0].name == "value"

    def test_roundtrip(self, omf_surface):
        """OMF → PyVista → OMF preserves triangles and data."""
        pd = omf_surface_to_pyvista(omf_surface)
        elem2 = pyvista_to_omf_surface(pd, name="RT")
        np.testing.assert_array_almost_equal(
            np.asarray(elem2.geometry.vertices),
            np.asarray(omf_surface.geometry.vertices),
        )
        np.testing.assert_array_equal(
            np.asarray(elem2.geometry.triangles),
            np.asarray(omf_surface.geometry.triangles),
        )

    def test_non_triangles_raises(self):
        """Non-triangular faces should raise ValueError."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
        )
        faces = np.array([4, 0, 1, 2, 3])  # Quad
        pd = pv.PolyData(vertices, faces)
        with pytest.raises(ValueError, match="triangular faces"):
            pyvista_to_omf_surface(pd)


# ==================================================================
# Volume ↔ PyVista
# ==================================================================


class TestOmfVolumeToPyvista:
    def test_basic_conversion(self, omf_volume):
        rg = omf_volume_to_pyvista(omf_volume)
        assert isinstance(rg, pv.RectilinearGrid)
        assert rg.n_cells == 6

    def test_data_attached(self, omf_volume):
        rg = omf_volume_to_pyvista(omf_volume)
        assert "density" in rg.cell_data
        np.testing.assert_array_almost_equal(
            rg.cell_data["density"], np.arange(6, dtype=float)
        )

    def test_edges(self, omf_volume):
        """Cell edges match origin + cumulative tensor."""
        rg = omf_volume_to_pyvista(omf_volume)
        np.testing.assert_array_almost_equal(rg.x, [0, 1, 2, 3])
        np.testing.assert_array_almost_equal(rg.y, [0, 1, 2])
        np.testing.assert_array_almost_equal(rg.z, [0, 1])


class TestPyvistaToOmfVolume:
    def test_basic_conversion(self, pv_rectilinear):
        elem = pyvista_to_omf_volume(pv_rectilinear, name="TestVol")
        assert elem.name == "TestVol"
        np.testing.assert_array_almost_equal(
            np.asarray(elem.geometry.tensor_u), [1, 1, 1]
        )
        np.testing.assert_array_almost_equal(
            np.asarray(elem.geometry.tensor_v), [1, 1]
        )
        np.testing.assert_array_almost_equal(
            np.asarray(elem.geometry.tensor_w), [1]
        )

    def test_data_attached(self, pv_rectilinear):
        elem = pyvista_to_omf_volume(pv_rectilinear)
        assert len(elem.data) == 1
        assert elem.data[0].name == "density"
        assert elem.data[0].location == "cells"

    def test_roundtrip(self, omf_volume):
        """OMF → PyVista → OMF preserves geometry and data."""
        rg = omf_volume_to_pyvista(omf_volume)
        elem2 = pyvista_to_omf_volume(rg, name="RT")
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
        )


# ==================================================================
# Project ↔ PyVista MultiBlock
# ==================================================================


class TestOmfProjectToPyvista:
    def test_basic_conversion(
        self, omf_pointset, omf_lineset, omf_surface, omf_volume
    ):
        proj = omf.Project(
            name="TestProject",
            elements=[omf_pointset, omf_lineset, omf_surface, omf_volume],
        )
        mb = omf_project_to_pyvista(proj)
        assert isinstance(mb, pv.MultiBlock)
        assert mb.n_blocks == 4
        assert "SamplePoints" in mb.keys()
        assert "SampleLines" in mb.keys()
        assert "SampleSurface" in mb.keys()
        assert "SampleVolume" in mb.keys()


class TestPyvistaToOmfProject:
    def test_basic_conversion(
        self, pv_points, pv_lines, pv_triangles, pv_rectilinear
    ):
        mb = pv.MultiBlock(
            {
                "Points": pv_points,
                "Lines": pv_lines,
                "Surface": pv_triangles,
                "Volume": pv_rectilinear,
            }
        )
        proj = pyvista_to_omf_project(mb, project_name="TestProject")
        assert proj.name == "TestProject"
        assert len(proj.elements) == 4

        names = [e.name for e in proj.elements]
        assert "Points" in names
        assert "Lines" in names
        assert "Surface" in names
        assert "Volume" in names

    def test_roundtrip(
        self, omf_pointset, omf_lineset, omf_surface, omf_volume
    ):
        """Project → PyVista → Project preserves element count and names."""
        proj = omf.Project(
            name="RT",
            elements=[omf_pointset, omf_lineset, omf_surface, omf_volume],
        )
        mb = omf_project_to_pyvista(proj)
        proj2 = pyvista_to_omf_project(mb, project_name="RT2")
        assert len(proj2.elements) == len(proj.elements)
        orig_names = sorted(e.name for e in proj.elements)
        new_names = sorted(e.name for e in proj2.elements)
        assert orig_names == new_names


class TestOmfPyvistaFileRoundtrip:
    """Test full file-based round-trip: OMF file → PyVista → OMF file."""

    def test_file_roundtrip(self, tmp_path, omf_pointset, omf_volume):
        proj = omf.Project(
            name="FileRT",
            elements=[omf_pointset, omf_volume],
        )

        # Save OMF
        path1 = str(tmp_path / "original.omf")
        omf.OMFWriter(proj, path1)

        # Read → PyVista
        reader = omf.OMFReader(path1)
        proj_read = reader.get_project()
        mb = omf_project_to_pyvista(proj_read)

        # PyVista → OMF
        proj2 = pyvista_to_omf_project(mb, project_name="FileRT2")

        # Save again
        path2 = str(tmp_path / "roundtrip.omf")
        omf.OMFWriter(proj2, path2)

        # Read back and verify
        reader2 = omf.OMFReader(path2)
        proj3 = reader2.get_project()
        assert len(proj3.elements) == 2
        names = sorted(e.name for e in proj3.elements)
        assert names == ["SamplePoints", "SampleVolume"]
