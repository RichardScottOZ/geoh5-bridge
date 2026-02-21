"""Tests for geoh5_bridge.vector module."""

from __future__ import annotations

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from geoh5py.workspace import Workspace

from geoh5_bridge.vector import (
    geodataframe_to_curve,
    geodataframe_to_points,
    geodataframe_to_surface,
)


@pytest.fixture()
def workspace(tmp_path):
    ws = Workspace.create(str(tmp_path / "test.geoh5"))
    yield ws
    ws.close()


@pytest.fixture()
def point_gdf():
    """Simple GeoDataFrame with Point geometries."""
    return gpd.GeoDataFrame(
        {"value": [10.0, 20.0, 30.0], "label": ["a", "b", "c"]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )


@pytest.fixture()
def point_3d_gdf():
    """GeoDataFrame with 3D Point geometries."""
    return gpd.GeoDataFrame(
        {"elevation": [100.0, 200.0]},
        geometry=[Point(0, 0, 10), Point(1, 1, 20)],
    )


@pytest.fixture()
def line_gdf():
    """GeoDataFrame with LineString geometries."""
    return gpd.GeoDataFrame(
        {"speed": [50.0, 80.0]},
        geometry=[
            LineString([(0, 0), (1, 1), (2, 0)]),
            LineString([(3, 3), (4, 4)]),
        ],
    )


class TestGeoDataFrameToPoints:
    def test_basic_conversion(self, workspace, point_gdf):
        pts = geodataframe_to_points(point_gdf, workspace, name="TestPts")
        assert pts is not None
        assert len(pts.vertices) == 3
        # Check z is 0 for 2D points
        np.testing.assert_array_equal(pts.vertices[:, 2], 0.0)

    def test_data_attached(self, workspace, point_gdf):
        pts = geodataframe_to_points(point_gdf, workspace, name="TestPts")
        children = [c for c in pts.children if hasattr(c, "values")]
        # Only numeric column ("value") should be attached
        assert len(children) == 1
        np.testing.assert_array_almost_equal(
            children[0].values, [10.0, 20.0, 30.0]
        )

    def test_3d_geometry(self, workspace, point_3d_gdf):
        pts = geodataframe_to_points(point_3d_gdf, workspace, name="Pts3D")
        assert pts.vertices[0, 2] == pytest.approx(10.0)
        assert pts.vertices[1, 2] == pytest.approx(20.0)

    def test_z_column(self, workspace, point_gdf):
        pts = geodataframe_to_points(
            point_gdf, workspace, z_column="value", name="ZCol"
        )
        np.testing.assert_array_almost_equal(
            pts.vertices[:, 2], [10.0, 20.0, 30.0]
        )

    def test_explicit_data_columns(self, workspace, point_gdf):
        pts = geodataframe_to_points(
            point_gdf, workspace, data_columns=["value"], name="Explicit"
        )
        children = [c for c in pts.children if hasattr(c, "values")]
        assert len(children) == 1


class TestGeoDataFrameToCurve:
    def test_basic_conversion(self, workspace, line_gdf):
        curve = geodataframe_to_curve(line_gdf, workspace, name="TestCurve")
        assert curve is not None
        # First line: 3 vertices, second: 2 vertices = 5 total
        assert len(curve.vertices) == 5

    def test_cells(self, workspace, line_gdf):
        curve = geodataframe_to_curve(line_gdf, workspace, name="TestCurve")
        # First line: 2 cells, second: 1 cell = 3 cells total
        assert curve.cells.shape == (3, 2)

    def test_data_attached(self, workspace, line_gdf):
        curve = geodataframe_to_curve(line_gdf, workspace, name="TestCurve")
        children = [c for c in curve.children if hasattr(c, "values")]
        assert len(children) == 1
        # Speed values replicated per vertex: [50,50,50, 80,80]
        np.testing.assert_array_almost_equal(
            children[0].values, [50.0, 50.0, 50.0, 80.0, 80.0]
        )

    def test_multilinestring(self, workspace):
        gdf = gpd.GeoDataFrame(
            {"id": [1.0]},
            geometry=[
                MultiLineString(
                    [[(0, 0), (1, 1)], [(2, 2), (3, 3), (4, 4)]]
                )
            ],
        )
        curve = geodataframe_to_curve(gdf, workspace, name="Multi")
        assert len(curve.vertices) == 5

    def test_empty_raises(self, workspace):
        gdf = gpd.GeoDataFrame(
            {"val": [1.0]}, geometry=[Point(0, 0)]
        )
        with pytest.raises(ValueError, match="No LineString"):
            geodataframe_to_curve(gdf, workspace)


class TestGeoDataFrameToSurface:
    def test_basic_conversion(self, workspace):
        gdf = gpd.GeoDataFrame(
            {"area": [1.0]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        )
        surf = geodataframe_to_surface(gdf, workspace, name="TestSurf")
        assert surf is not None
        # Square: 4 vertices, 2 triangles
        assert len(surf.vertices) == 4
        assert surf.cells.shape == (2, 3)
        # All z values should be 0 for 2D polygons
        np.testing.assert_array_equal(surf.vertices[:, 2], 0.0)

    def test_data_attached(self, workspace):
        gdf = gpd.GeoDataFrame(
            {"area": [1.0], "label": ["a"]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        )
        surf = geodataframe_to_surface(gdf, workspace, name="DataSurf")
        children = [c for c in surf.children if hasattr(c, "values")]
        # Only numeric column ("area") should be attached
        assert len(children) == 1
        # Per-vertex values: 4 vertices all from feature 0
        np.testing.assert_array_almost_equal(
            children[0].values, [1.0, 1.0, 1.0, 1.0]
        )

    def test_concave_polygon(self, workspace):
        # L-shaped polygon: concave, should produce correct triangulation
        gdf = gpd.GeoDataFrame(
            {"id": [1.0]},
            geometry=[
                Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
            ],
        )
        surf = geodataframe_to_surface(gdf, workspace, name="Concave")
        assert surf is not None
        assert len(surf.vertices) == 6
        # All triangle cells should have 3 vertex indices
        assert surf.cells.shape[1] == 3

    def test_multipolygon(self, workspace):
        gdf = gpd.GeoDataFrame(
            {"id": [1.0]},
            geometry=[
                MultiPolygon(
                    [
                        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                    ]
                )
            ],
        )
        surf = geodataframe_to_surface(gdf, workspace, name="Multi")
        # Two squares: 4+4=8 vertices, 2+2=4 triangles
        assert len(surf.vertices) == 8
        assert surf.cells.shape == (4, 3)

    def test_multiple_features(self, workspace):
        gdf = gpd.GeoDataFrame(
            {"area": [1.0, 4.0]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
            ],
        )
        surf = geodataframe_to_surface(gdf, workspace, name="TwoPolys")
        assert len(surf.vertices) == 8
        assert surf.cells.shape == (4, 3)

    def test_empty_raises(self, workspace):
        gdf = gpd.GeoDataFrame(
            {"val": [1.0]}, geometry=[Point(0, 0)]
        )
        with pytest.raises(ValueError, match="No Polygon"):
            geodataframe_to_surface(gdf, workspace)
