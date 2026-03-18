# geoh5-bridge

Convert common Python geospatial data structures to and from the
[geoh5 format](https://mirageoscience-geoh5py.readthedocs-hosted.com/en/stable/content/geoh5_format/)
used by [ANALYST](https://www.mirageoscience.com/mining-industry-software/geoscience-analyst/)
and other Mira Geoscience tools.

`geoh5-bridge` wraps [geoh5py](https://github.com/MiraGeoscience/geoh5py) and
provides a simple, Pythonic API for common conversion tasks — including
bidirectional bridges to [PyVista](https://docs.pyvista.org/) and the
[Open Mining Format (OMF)](https://omf.readthedocs.io/).

## Features

### Raster / xarray bridge

| Source | Target | Function |
|---|---|---|
| `xarray.DataArray` (raster / GeoTIFF via [rioxarray](https://corteva.github.io/rioxarray/)) | `Grid2D` | `raster_to_grid2d()` |
| `xarray.DataArray` (raster) | `Points` | `raster_to_points()` |
| `Grid2D` | `xarray.DataArray` | `grid2d_to_raster()` |
| `xarray.Dataset` / `DataArray` (3-D) | `BlockModel` | `xarray_to_blockmodel()` |
| `BlockModel` | `xarray.Dataset` | `blockmodel_to_xarray()` |

### Vector / GeoDataFrame bridge

| Source | Target | Function |
|---|---|---|
| `geopandas.GeoDataFrame` (Point geometries) | `Points` | `geodataframe_to_points()` |
| `Points` | `geopandas.GeoDataFrame` | `points_to_geodataframe()` |
| `geopandas.GeoDataFrame` (LineString / MultiLineString) | `Curve` | `geodataframe_to_curve()` |
| `Curve` | `geopandas.GeoDataFrame` | `curve_to_geodataframe()` |
| `geopandas.GeoDataFrame` (Polygon / MultiPolygon) | `Surface` | `geodataframe_to_surface()` |
| `Surface` | `geopandas.GeoDataFrame` | `surface_to_geodataframe()` |

### PyVista bridge

| Source | Target | Function |
|---|---|---|
| `Points` | `pyvista.PolyData` | `points_to_pyvista()` |
| `pyvista.PolyData` | `Points` | `pyvista_to_points()` |
| `Grid2D` | `pyvista.StructuredGrid` | `grid2d_to_pyvista()` |
| `pyvista.StructuredGrid` | `Grid2D` | `pyvista_to_grid2d()` |
| `Curve` | `pyvista.PolyData` (lines) | `curve_to_pyvista()` |
| `pyvista.PolyData` (lines) | `Curve` | `pyvista_to_curve()` |
| `Surface` | `pyvista.PolyData` (mesh) | `surface_to_pyvista()` |
| `pyvista.PolyData` (mesh) | `Surface` | `pyvista_to_surface()` |
| `BlockModel` | `pyvista.RectilinearGrid` | `blockmodel_to_pyvista()` |
| `pyvista.RectilinearGrid` | `BlockModel` | `pyvista_to_blockmodel()` |

### OMF ↔ PyVista bridge

| Source | Target | Function |
|---|---|---|
| `omf.PointSetElement` | `pyvista.PolyData` | `omf_pointset_to_pyvista()` |
| `pyvista.PolyData` | `omf.PointSetElement` | `pyvista_to_omf_pointset()` |
| `omf.LineSetElement` | `pyvista.PolyData` (lines) | `omf_lineset_to_pyvista()` |
| `pyvista.PolyData` (lines) | `omf.LineSetElement` | `pyvista_to_omf_lineset()` |
| `omf.SurfaceElement` (`SurfaceGeometry` or `SurfaceGridGeometry`) | `pyvista.PolyData` (mesh) | `omf_surface_to_pyvista()` |
| `pyvista.PolyData` (mesh) | `omf.SurfaceElement` (`SurfaceGeometry`) | `pyvista_to_omf_surface()` |
| `omf.VolumeElement` | `pyvista.RectilinearGrid` | `omf_volume_to_pyvista()` |
| `pyvista.RectilinearGrid` | `omf.VolumeElement` | `pyvista_to_omf_volume()` |
| `omf.Project` | `pyvista.MultiBlock` | `omf_project_to_pyvista()` |
| `pyvista.MultiBlock` | `omf.Project` | `pyvista_to_omf_project()` |

### OMF ↔ geoh5 bridge (direct, no PyVista required)

| Source | Target | Function |
|---|---|---|
| `omf.PointSetElement` | `Points` | `omf_pointset_to_points()` |
| `Points` | `omf.PointSetElement` | `points_to_omf_pointset()` |
| `omf.LineSetElement` | `Curve` | `omf_lineset_to_curve()` |
| `Curve` | `omf.LineSetElement` | `curve_to_omf_lineset()` |
| `omf.SurfaceElement` (`SurfaceGeometry` or `SurfaceGridGeometry`) | `Surface` or `Grid2D` | `omf_surface_to_surface()` |
| `omf.SurfaceElement` (`SurfaceGridGeometry`, uniform spacing) | `Grid2D` | `omf_surface_to_grid2d()` |
| `Surface` | `omf.SurfaceElement` (`SurfaceGeometry`) | `surface_to_omf_surface()` |
| `Grid2D` | `omf.SurfaceElement` (`SurfaceGridGeometry`) | `grid2d_to_omf_surface()` |
| `omf.VolumeElement` | `BlockModel` | `omf_volume_to_blockmodel()` |
| `BlockModel` | `omf.VolumeElement` | `blockmodel_to_omf_volume()` |

## Installation

```bash
pip install geoh5-bridge            # core (geoh5py + numpy)
pip install geoh5-bridge[raster]    # + rioxarray, xarray, rasterio
pip install geoh5-bridge[vector]    # + geopandas
pip install geoh5-bridge[grid3d]    # + xarray, netcdf4
pip install geoh5-bridge[pyvista]   # + pyvista
pip install geoh5-bridge[omf]       # + omf, pyvista
pip install geoh5-bridge[all]       # everything
pip install geoh5-bridge[dev]       # all + pytest
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

### GeoDataFrame → Points (and back)

```python
import geopandas as gpd
from geoh5_bridge import geodataframe_to_points, points_to_geodataframe

gdf = gpd.read_file("sample_points.geojson")

with Workspace.create("output.geoh5") as ws:
    pts = geodataframe_to_points(gdf, ws, name="SamplePoints")

    # Round-trip back to a GeoDataFrame
    gdf_out = points_to_geodataframe(pts)
```

All numeric attribute columns are attached as data channels
automatically. Use `data_columns=["col1", "col2"]` to select specific
columns, or `z_column="elevation"` to use a column for the z-coordinate.

### GeoDataFrame (lines) → Curve (and back)

```python
from geoh5_bridge import geodataframe_to_curve, curve_to_geodataframe

gdf = gpd.read_file("roads.geojson")

with Workspace.create("output.geoh5") as ws:
    curve = geodataframe_to_curve(gdf, ws, name="Roads")
    gdf_out = curve_to_geodataframe(curve)
```

### GeoDataFrame (polygons) → Surface (and back)

```python
from geoh5_bridge import geodataframe_to_surface, surface_to_geodataframe

gdf = gpd.read_file("parcels.geojson")

with Workspace.create("output.geoh5") as ws:
    surf = geodataframe_to_surface(gdf, ws, name="Parcels")
    gdf_out = surface_to_geodataframe(surf)
```

Polygon and MultiPolygon geometries are triangulated and stored as a
Surface mesh. Concave polygons and polygons with holes are handled
correctly.

### 3-D xarray / NetCDF → BlockModel (and back)

```python
import xarray as xr
from geoh5_bridge import xarray_to_blockmodel, blockmodel_to_xarray

ds = xr.open_dataset("model.nc")

with Workspace.create("output.geoh5") as ws:
    bm = xarray_to_blockmodel(ds, ws, name="DensityModel")

    # Round-trip back to xarray
    ds_out = blockmodel_to_xarray(bm)
```

Dimension names are auto-detected from common conventions (`x`/`easting`,
`y`/`northing`, `z`/`depth`, etc.) or can be specified explicitly with the
`dims=("easting", "northing", "depth")` parameter.

---

### PyVista bridge

`geoh5-bridge` provides **bidirectional** conversions between all major
geoh5py object types and their PyVista equivalents.  Requires
`pip install geoh5-bridge[pyvista]`.

#### Points ↔ PyVista PolyData

```python
import pyvista as pv
from geoh5_bridge import points_to_pyvista, pyvista_to_points

with Workspace.create("output.geoh5") as ws:
    # geoh5 → PyVista
    cloud = points_to_pyvista(pts)

    # PyVista → geoh5
    pts2 = pyvista_to_points(cloud, ws, name="CloudPoints")
```

#### Grid2D ↔ PyVista StructuredGrid

```python
from geoh5_bridge import grid2d_to_pyvista, pyvista_to_grid2d

with Workspace.create("output.geoh5") as ws:
    sg = grid2d_to_pyvista(grid)
    grid2 = pyvista_to_grid2d(sg, ws, name="Grid")
```

#### Curve ↔ PyVista PolyData (lines)

```python
from geoh5_bridge import curve_to_pyvista, pyvista_to_curve

with Workspace.create("output.geoh5") as ws:
    pd_lines = curve_to_pyvista(curve)
    curve2 = pyvista_to_curve(pd_lines, ws, name="Lines")
```

#### Surface ↔ PyVista PolyData (triangle mesh)

```python
from geoh5_bridge import surface_to_pyvista, pyvista_to_surface

with Workspace.create("output.geoh5") as ws:
    mesh = surface_to_pyvista(surf)
    surf2 = pyvista_to_surface(mesh, ws, name="Mesh")
```

#### BlockModel ↔ PyVista RectilinearGrid

```python
from geoh5_bridge import blockmodel_to_pyvista, pyvista_to_blockmodel

with Workspace.create("output.geoh5") as ws:
    rg = blockmodel_to_pyvista(bm)
    bm2 = pyvista_to_blockmodel(rg, ws, name="Model")
```

All PyVista bridge functions accept an optional `data_names` list to
select which data channels to include.

---

### OMF ↔ PyVista bridge

Bidirectional conversions between [Open Mining Format (OMF)](https://omf.readthedocs.io/)
elements and PyVista meshes.  Unlike `omfvista`, this bridge also supports
the **reverse direction** (PyVista → OMF).  Requires
`pip install geoh5-bridge[omf]`.

```python
import omf
from geoh5_bridge import (
    omf_pointset_to_pyvista, pyvista_to_omf_pointset,
    omf_lineset_to_pyvista,  pyvista_to_omf_lineset,
    omf_surface_to_pyvista,  pyvista_to_omf_surface,
    omf_volume_to_pyvista,   pyvista_to_omf_volume,
    omf_project_to_pyvista,  pyvista_to_omf_project,
)

# Load an OMF project and convert every element to PyVista
reader = omf.OMFReader("model.omf")
project = reader.get_project()
multiblock = omf_project_to_pyvista(project)   # pyvista.MultiBlock

# Modify in PyVista and write back to OMF
project2 = pyvista_to_omf_project(multiblock, project_name="Updated")
omf.OMFWriter(project2, "model_updated.omf")
```

Individual element conversions follow the same pattern:

```python
# OMF → PyVista
pd = omf_pointset_to_pyvista(pointset)
pd = omf_lineset_to_pyvista(lineset)
pd = omf_surface_to_pyvista(surface_elem)   # handles both SurfaceGeometry
                                             # and SurfaceGridGeometry (see below)
rg = omf_volume_to_pyvista(volume)

# PyVista → OMF
pointset  = pyvista_to_omf_pointset(pd, name="Samples")
lineset   = pyvista_to_omf_lineset(pd, name="Drillholes")
surf_elem = pyvista_to_omf_surface(mesh, name="Fault")
volume    = pyvista_to_omf_volume(rg, name="Density")
```

#### OMF surface geometry types

OMF `SurfaceElement` objects can use one of two geometry representations:

**`SurfaceGeometry`** — explicit triangle mesh

The most common type.  Stores a list of 3-D vertices and a list of
triangle index triples (exactly like a geoh5py `Surface`).

```python
import omf
import numpy as np

# Build a simple triangle
geom = omf.SurfaceGeometry(
    vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
    triangles=np.array([[0, 1, 2]], dtype=int),
)
elem = omf.SurfaceElement(name="Fault", geometry=geom)

# Convert to PyVista PolyData
mesh = omf_surface_to_pyvista(elem)
```

**`SurfaceGridGeometry`** — structured (regular) grid surface

Used for DEM-like or map-view surfaces where the XY layout is a
regular grid and each node may have an independent Z elevation
(`offset_w`).  Defined by:

| Attribute | Type | Description |
|---|---|---|
| `origin` | `[x, y, z]` | Grid origin (corner node) |
| `tensor_u` | `float[]` | Cell spacings along the U axis (len = number of columns) |
| `tensor_v` | `float[]` | Cell spacings along the V axis (len = number of rows) |
| `axis_u` | `[dx, dy, dz]` | Unit vector for the U direction |
| `axis_v` | `[dx, dy, dz]` | Unit vector for the V direction |
| `offset_w` | `float[]` or `None` | Per-node Z displacement, row-major order `j*(nu+1)+i` |

The node count is `(len(tensor_u)+1) × (len(tensor_v)+1)`.  Each grid
cell is split into two triangles, so `omf_surface_to_pyvista()` returns
a `PolyData` with `2 × len(tensor_u) × len(tensor_v)` triangles.

```python
import omf
import numpy as np

# 3×2 cell grid (4 nodes wide, 3 nodes tall) in the XY plane
geom = omf.SurfaceGridGeometry(
    origin=np.array([0.0, 0.0, 0.0]),
    tensor_u=np.array([10.0, 10.0, 10.0]),   # 3 columns → 4 node columns
    tensor_v=np.array([10.0, 10.0]),           # 2 rows    → 3 node rows
    axis_u=np.array([1.0, 0.0, 0.0]),
    axis_v=np.array([0.0, 1.0, 0.0]),
    # Optional: per-node elevation offsets (4×3 = 12 nodes, row-major)
    offset_w=np.array([0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5], dtype=float),
)
elem = omf.SurfaceElement(name="Topography", geometry=geom)

# Grid is triangulated automatically → pyvista.PolyData (12 triangles)
mesh = omf_surface_to_pyvista(elem)
```

> **Tip:** `pyvista_to_omf_surface()` always writes back as
> `SurfaceGeometry` (explicit triangles), which is universally compatible
> with all OMF readers.

---

### OMF ↔ geoh5 bridge (direct, no PyVista required)

Convert OMF elements directly to/from geoh5py objects without an
intermediate PyVista step.  Requires `pip install geoh5-bridge[omf]`.

Both `SurfaceGeometry` and `SurfaceGridGeometry` are fully supported.
When a grid surface has **uniform cell spacing** and **no per-node
elevation offsets** (`offset_w`), `omf_surface_to_surface()` returns a
`Grid2D` by default (controlled by the `prefer_grid2d` parameter).  In
all other cases — explicit triangle meshes, non-uniform spacing, or grids
with elevation offsets — a triangulated `Surface` is returned.

```python
import omf
from geoh5py.workspace import Workspace
from geoh5_bridge import (
    Points, Curve, Surface, BlockModel,
    omf_pointset_to_points, points_to_omf_pointset,
    omf_lineset_to_curve,   curve_to_omf_lineset,
    omf_surface_to_surface, surface_to_omf_surface,
    omf_volume_to_blockmodel, blockmodel_to_omf_volume,
)

reader = omf.OMFReader("model.omf")
project = reader.get_project()

with Workspace.create("output.geoh5") as ws:
    for element in project.elements:
        if isinstance(element, omf.PointSetElement):
            omf_pointset_to_points(element, ws)
        elif isinstance(element, omf.LineSetElement):
            omf_lineset_to_curve(element, ws)
        elif isinstance(element, omf.SurfaceElement):
            # SurfaceGridGeometry with uniform spacing → Grid2D (default)
            # SurfaceGeometry or non-uniform/offset grids  → Surface
            omf_surface_to_surface(element, ws)
        elif isinstance(element, omf.VolumeElement):
            omf_volume_to_blockmodel(element, ws)
```

To force all surfaces to be triangulated as `Surface` regardless of
geometry type, pass `prefer_grid2d=False`:

```python
with Workspace.create("output.geoh5") as ws:
    for element in project.elements:
        if isinstance(element, omf.SurfaceElement):
            # Always returns a Surface (triangulated)
            omf_surface_to_surface(element, ws, prefer_grid2d=False)
```

Reverse direction — export geoh5 objects to OMF:

```python
from geoh5py.objects import Grid2D
from geoh5_bridge import grid2d_to_omf_surface

with Workspace("existing.geoh5") as ws:
    for obj in ws.objects:
        if isinstance(obj, BlockModel):
            elem = blockmodel_to_omf_volume(obj)
        elif isinstance(obj, Grid2D):
            # Grid2D → SurfaceGridGeometry (preserves implicit grid structure)
            elem = grid2d_to_omf_surface(obj)
        elif isinstance(obj, Surface):
            # Surface → SurfaceGeometry (explicit triangles)
            elem = surface_to_omf_surface(obj)
        elif isinstance(obj, Curve):
            elem = curve_to_omf_lineset(obj)
        elif isinstance(obj, Points):
            elem = points_to_omf_pointset(obj)
```

#### `SurfaceGridGeometry` → `Grid2D` (direct conversion)

For OMF grid surfaces that map cleanly to a geoh5py `Grid2D` (uniform
cell spacing, no per-node elevation offsets, axes aligned to the
horizontal plane), use `omf_surface_to_grid2d()` directly:

```python
import omf
import numpy as np
from geoh5py.workspace import Workspace
from geoh5_bridge import omf_surface_to_grid2d

# Build a 4×3 cell uniform grid surface
geom = omf.SurfaceGridGeometry(
    origin=np.array([500000.0, 7000000.0, 350.0]),
    tensor_u=np.full(4, 25.0),   # 4 columns, 25 m spacing
    tensor_v=np.full(3, 25.0),   # 3 rows,    25 m spacing
    axis_u=np.array([1.0, 0.0, 0.0]),
    axis_v=np.array([0.0, 1.0, 0.0]),
    # No offset_w → flat surface at z = 350 m
)
elem = omf.SurfaceElement(name="Topography", geometry=geom)

with Workspace.create("output.geoh5") as ws:
    grid = omf_surface_to_grid2d(elem, ws, name="Topography")
    # grid is a geoh5py Grid2D:
    #   origin      = (500000, 7000000, 350)
    #   u_cell_size = 25.0,  u_count = 4
    #   v_cell_size = 25.0,  v_count = 3
    #   rotation    = 0°,    dip     = 0°
```

`omf_surface_to_grid2d()` raises `TypeError` if the element does not use
`SurfaceGridGeometry`, and `ValueError` if the spacing is non-uniform or
`offset_w` is present (use `omf_surface_to_surface()` for those cases).

#### `Grid2D` → `SurfaceGridGeometry` (round-trip)

`grid2d_to_omf_surface()` converts a geoh5py `Grid2D` back to an OMF
`SurfaceElement` with `SurfaceGridGeometry`, preserving the implicit grid
structure (uniform spacing, rotation, and dip):

```python
import omf
from geoh5py.workspace import Workspace
from geoh5_bridge import grid2d_to_omf_surface

with Workspace("existing.geoh5") as ws:
    grid = ws.get_entity("Topography")[0]   # a Grid2D object
    elem = grid2d_to_omf_surface(grid, name="Topography")
    # elem.geometry is SurfaceGridGeometry — compact, no triangulation needed

    project = omf.Project(name="Export", elements=[elem])
    omf.OMFWriter(project, "topography.omf")
```

The `rotation` angle (degrees CCW from East) maps to `axis_u`, and the
`dip` angle (degrees above horizontal) controls the tilt of `axis_v`.  A
`dip` of `0°` yields a flat horizontal grid; `90°` yields a vertical
plane.

#### How `omf_surface_to_surface()` handles grid surfaces

When the OMF surface uses `SurfaceGridGeometry`, the function first checks
whether the grid is compatible with `Grid2D` (uniform spacing, no
`offset_w`).  If `prefer_grid2d=True` (the default) and the check passes,
it delegates to `omf_surface_to_grid2d()` and returns a `Grid2D`.
Otherwise it reconstructs the grid node positions from `origin`,
`tensor_u`/`tensor_v`, and `axis_u`/`axis_v`, applies any per-node Z
offsets from `offset_w`, and stores the result as a triangulated `Surface`.

| OMF geometry | `prefer_grid2d` | `offset_w` | uniform spacing | Result |
|---|---|---|---|---|
| `SurfaceGeometry` | any | N/A | N/A | `Surface` (explicit triangles) |
| `SurfaceGridGeometry` | `True` (default) | absent | yes | `Grid2D` |
| `SurfaceGridGeometry` | `True` (default) | present | any | `Surface` (triangulated) |
| `SurfaceGridGeometry` | `True` (default) | absent | no | `Surface` (triangulated) |
| `SurfaceGridGeometry` | `False` | any | any | `Surface` (triangulated) |

where `nu = len(tensor_u)` and `nv = len(tensor_v)`.  When triangulated,
the node count is `(nu+1) × (nv+1)` and the triangle count is
`2 × nu × nv`.

> **Note:** `surface_to_omf_surface()` always exports a geoh5py `Surface`
> as `SurfaceGeometry` (explicit triangles), preserving the mesh exactly.
> For `Grid2D` objects, use `grid2d_to_omf_surface()` to preserve the
> compact grid representation.  A grid surface imported from OMF via
> `omf_surface_to_surface()` with `prefer_grid2d=False` and then exported
> back to OMF will be stored as explicit triangles rather than as a grid,
> which is compatible with all OMF readers but uses more storage.

## Related packages and resources

### Core

| Package | Description |
|---|---|
| [geoh5py](https://github.com/MiraGeoscience/geoh5py) | Official Python API for the geoh5 file format |
| [rioxarray](https://corteva.github.io/rioxarray/) | Rasterio xarray extension for reading/writing geospatial rasters |
| [geopandas](https://geopandas.org/) | Spatial operations on geometric types built on pandas |
| [xarray](https://xarray.dev/) | N-D labelled arrays and datasets |

### Open Mining Format (OMF)

| Package | Description |
|---|---|
| [omf](https://omf.readthedocs.io/) | Official Python library for reading and writing OMF files |
| [omfvista](https://github.com/OpenGeoVis/omfvista) | PyVista interface for OMF (OMF → PyVista only; `geoh5-bridge` adds the reverse) |

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
