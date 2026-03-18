"""geoh5-bridge: Bidirectional conversions between pydata/PyVista/OMF and geoh5."""

from geoh5py.objects import BlockModel, Curve, Points, Surface

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

__all__ = [
    # geoh5py object types used in examples
    "Points",
    "Curve",
    "Surface",
    "BlockModel",
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
    # OMF ↔ PyVista conversions
    "omf_pointset_to_pyvista",
    "pyvista_to_omf_pointset",
    "omf_lineset_to_pyvista",
    "pyvista_to_omf_lineset",
    "omf_surface_to_pyvista",
    "pyvista_to_omf_surface",
    "omf_volume_to_pyvista",
    "pyvista_to_omf_volume",
    "omf_project_to_pyvista",
    "pyvista_to_omf_project",
    # OMF ↔ geoh5 conversions
    "omf_pointset_to_points",
    "points_to_omf_pointset",
    "omf_lineset_to_curve",
    "curve_to_omf_lineset",
    "omf_surface_to_surface",
    "surface_to_omf_surface",
    "omf_volume_to_blockmodel",
    "blockmodel_to_omf_volume",
]

__version__ = "0.1.0"
