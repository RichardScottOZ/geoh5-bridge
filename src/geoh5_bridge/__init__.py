"""geoh5-bridge: Bidirectional conversions between pydata/PyVista and geoh5."""

from geoh5_bridge.raster import grid2d_to_raster, raster_to_grid2d, raster_to_points
from geoh5_bridge.vector import (
    curve_to_geodataframe,
    geodataframe_to_curve,
    geodataframe_to_points,
    geodataframe_to_surface,
    points_to_geodataframe,
    surface_to_geodataframe,
)
from geoh5_bridge.grid3d import blockmodel_to_xarray, xarray_to_blockmodel
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

__all__ = [
    # Raster conversions
    "grid2d_to_raster",
    "raster_to_grid2d",
    "raster_to_points",
    # Vector conversions
    "geodataframe_to_points",
    "geodataframe_to_curve",
    "geodataframe_to_surface",
    "points_to_geodataframe",
    "curve_to_geodataframe",
    "surface_to_geodataframe",
    # 3D grid conversions
    "blockmodel_to_xarray",
    "xarray_to_blockmodel",
    # PyVista conversions
    "points_to_pyvista",
    "pyvista_to_points",
    "grid2d_to_pyvista",
    "pyvista_to_grid2d",
    "curve_to_pyvista",
    "pyvista_to_curve",
    "surface_to_pyvista",
    "pyvista_to_surface",
    "blockmodel_to_pyvista",
    "pyvista_to_blockmodel",
]

__version__ = "0.1.0"
