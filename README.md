# geoh5-bridge

Convert common Python geospatial data structures to the
[geoh5 format](https://mirageoscience-geoh5py.readthedocs-hosted.com/en/stable/content/geoh5_format/)
used by [ANALYST](https://www.mirageoscience.com/mining-industry-software/geoscience-analyst/)
and other Mira Geoscience tools.

`geoh5-bridge` wraps [geoh5py](https://github.com/MiraGeoscience/geoh5py) and
provides a simple, Pythonic API for common conversion tasks.

## Features

| Source format | Target geoh5 object | Function |
|---|---|---|
| `xarray.DataArray` (raster / GeoTIFF via [rioxarray](https://corteva.github.io/rioxarray/)) | `Grid2D` | `raster_to_grid2d()` |
| `xarray.DataArray` (raster) | `Points` | `raster_to_points()` |
| `geopandas.GeoDataFrame` (Point geometries) | `Points` | `geodataframe_to_points()` |
| `geopandas.GeoDataFrame` (LineString / MultiLineString) | `Curve` | `geodataframe_to_curve()` |
| `geopandas.GeoDataFrame` (Polygon / MultiPolygon) | `Surface` | `geodataframe_to_surface()` |
| `xarray.Dataset` / `DataArray` (3-D) | `BlockModel` | `xarray_to_blockmodel()` |

## Installation

```bash
pip install geoh5-bridge          # core (geoh5py + numpy)
pip install geoh5-bridge[raster]  # + rioxarray, xarray, rasterio
pip install geoh5-bridge[vector]  # + geopandas
pip install geoh5-bridge[grid3d]  # + xarray, netcdf4
pip install geoh5-bridge[all]     # everything
pip install geoh5-bridge[dev]     # all + pytest
```

## Quick start

### Raster (GeoTIFF) → Grid2D

```python
import rioxarray                       # registers the "rasterio" engine
import xarray as xr
from geoh5py.workspace import Workspace
from geoh5_bridge import raster_to_grid2d

da = xr.open_dataarray("elevation.tif", engine="rasterio")

with Workspace.create("output.geoh5") as ws:
    grid = raster_to_grid2d(da, ws, name="Elevation")
```

Multi-band rasters are supported — each band is stored as a separate
data channel on the Grid2D.

### Raster → Points (with nodata filtering)

```python
from geoh5_bridge import raster_to_points

with Workspace.create("output.geoh5") as ws:
    pts = raster_to_points(da, ws, name="ElevPoints", nodata=-9999)
```

### GeoDataFrame → Points

```python
import geopandas as gpd
from geoh5_bridge import geodataframe_to_points

gdf = gpd.read_file("sample_points.geojson")

with Workspace.create("output.geoh5") as ws:
    pts = geodataframe_to_points(gdf, ws, name="SamplePoints")
```

All numeric attribute columns are attached as data channels
automatically. Use `data_columns=["col1", "col2"]` to select specific
columns, or `z_column="elevation"` to use a column for the z-coordinate.

### GeoDataFrame (lines) → Curve

```python
from geoh5_bridge import geodataframe_to_curve

gdf = gpd.read_file("roads.geojson")

with Workspace.create("output.geoh5") as ws:
    curve = geodataframe_to_curve(gdf, ws, name="Roads")
```

### GeoDataFrame (polygons) → Surface

```python
from geoh5_bridge import geodataframe_to_surface

gdf = gpd.read_file("parcels.geojson")

with Workspace.create("output.geoh5") as ws:
    surf = geodataframe_to_surface(gdf, ws, name="Parcels")
```

Polygon and MultiPolygon geometries are triangulated and stored as a
Surface mesh. Concave polygons and polygons with holes are handled
correctly.

### 3-D xarray / NetCDF → BlockModel

```python
import xarray as xr
from geoh5_bridge import xarray_to_blockmodel

ds = xr.open_dataset("model.nc")

with Workspace.create("output.geoh5") as ws:
    bm = xarray_to_blockmodel(ds, ws, name="DensityModel")
```

Dimension names are auto-detected from common conventions (`x`/`easting`,
`y`/`northing`, `z`/`depth`, etc.) or can be specified explicitly with the
`dims=("easting", "northing", "depth")` parameter.

## Related packages and resources

### Core

| Package | Description |
|---|---|
| [geoh5py](https://github.com/MiraGeoscience/geoh5py) | Official Python API for the geoh5 file format |
| [rioxarray](https://corteva.github.io/rioxarray/) | Rasterio xarray extension for reading/writing geospatial rasters |
| [geopandas](https://geopandas.org/) | Spatial operations on geometric types built on pandas |
| [xarray](https://xarray.dev/) | N-D labelled arrays and datasets |

### Visualisation

| Package | Description |
|---|---|
| [PyVista](https://docs.pyvista.org/) | 3-D plotting and mesh analysis (VTK wrapper) — useful for visualising BlockModels and Surfaces |
| [GeoVista](https://github.com/bjlittle/geovista) | Cartographic rendering and mesh analytics powered by PyVista |
| [geoh5vista](https://github.com/MiraGeoscience/geoh5vista) | PyVista interface for geoh5 objects — read geoh5 directly into PyVista meshes |
| [pyvista-xarray](https://github.com/pyvista/pyvista-xarray) | xarray accessor for PyVista — plot xarray data in 3-D with PyVista |

### Drillhole / subsurface data

| Package | Description |
|---|---|
| [lasio](https://github.com/kinverarity1/lasio) | Read/write Log ASCII Standard (LAS) well-log files |
| [welly](https://github.com/agilescientific/welly) | Manage well data with an easy-to-use Python interface |
| [striplog](https://github.com/agilescientific/striplog) | Lithology and stratigraphic logs |
| [geoh5py Drillhole](https://mirageoscience-geoh5py.readthedocs-hosted.com/en/stable/content/user_guide/objects.html) | geoh5py natively supports `Drillhole` objects for collar + survey + interval data |

### Other useful geoscience I/O

| Package | Description |
|---|---|
| [SimPEG](https://simpeg.xyz/) | Simulation and parameter estimation for geophysics — can export to geoh5 |
| [discretize](https://discretize.simpeg.xyz/) | Meshing utilities for SimPEG (TensorMesh ↔ BlockModel) |
| [verde](https://www.fatiando.org/verde/) | Spatial data processing and gridding |
| [rasterio](https://rasterio.readthedocs.io/) | Low-level raster I/O |

## Development

```bash
git clone https://github.com/RichardScottOZ/geoh5-bridge.git
cd geoh5-bridge
pip install -e ".[dev]"
pytest
```

## License

MIT
