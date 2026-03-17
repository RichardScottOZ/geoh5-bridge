"""geoh5-bridge: Convert pydata formats to geoh5."""

from geoh5_bridge.raster import grid2d_to_raster, raster_to_grid2d, raster_to_points
from geoh5_bridge.vector import (
    geodataframe_to_points,
    geodataframe_to_curve,
    geodataframe_to_surface,
    points_to_geodataframe,
    curve_to_geodataframe,
    surface_to_geodataframe,
)
from geoh5_bridge.grid3d import blockmodel_to_xarray, xarray_to_blockmodel

__all__ = [
    "grid2d_to_raster",
    "raster_to_grid2d",
    "raster_to_points",
    "geodataframe_to_points",
    "geodataframe_to_curve",
    "geodataframe_to_surface",
    "points_to_geodataframe",
    "curve_to_geodataframe",
    "surface_to_geodataframe",
    "blockmodel_to_xarray",
    "xarray_to_blockmodel",
]

__version__ = "0.1.0"
