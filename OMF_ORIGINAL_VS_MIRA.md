# OMF Handling: geoh5-bridge vs Original omf-python vs Mira Geoscience Fork

This document provides a detailed comparison of three different approaches to
working with the Open Mining Format (OMF) in the context of OMF ↔ geoh5
conversion:

1. **Original omf-python** – [`gmggroup/omf-python`](https://github.com/gmggroup/omf-python)
   / PyPI `omf`
2. **Mira Geoscience fork** – [`MiraGeoscience/omf`](https://github.com/MiraGeoscience/omf)
   / PyPI `mira-omf`
3. **This repository** – [`geoh5-bridge`](https://github.com/RichardScottOZ/geoh5-bridge)
   (uses the original `omf` library as a dependency)

---

## 1. Original omf-python (`gmggroup/omf-python`)

### Purpose

A pure data-model and serialisation library for the Open Mining Format
standard, maintained by the Global Mining Standards & Guidelines Group (GMG).
It is **not** coupled to any specific visualisation or workflow tool.

### Architecture

- Provides Python classes for each OMF element type (`PointSetElement`,
  `LineSetElement`, `SurfaceElement`, `VolumeElement`) and their associated
  geometry and data sub-objects.
- Relies on the [`properties`](https://github.com/seequent/properties) library
  (open-sourced by Seequent) for attribute validation and serialisation.
- File I/O through `OMFReader` / `OMFWriter`:

  ```python
  import omf

  reader = omf.OMFReader("model.omf")
  project = reader.get_project()   # → omf.Project

  writer = omf.OMFWriter(project, "output.omf")
  ```

### Versions

| Branch / Tag | Status | Notes |
|---|---|---|
| `1.0.x` | **Stable** | Used by geoh5-bridge (`omf>=1.0`). API documented at [omf.readthedocs.io](https://omf.readthedocs.io). |
| `2.0.0a0` | Pre-release alpha | Backwards-incompatible rewrite; not yet stable. |

The `geoh5-bridge` repository targets `omf>=1.0` (the 1.x stable API). Mira's
fork is also based on the 1.x API.

### Supported Element Types (1.x)

| OMF Class | Geometry Class | Data Locations |
|---|---|---|
| `PointSetElement` | `PointSetGeometry` | `vertices` |
| `LineSetElement` | `LineSetGeometry` | `vertices`, `segments` |
| `SurfaceElement` | `SurfaceGeometry` (triangles) or `SurfaceGridGeometry` (structured grid) | `vertices`, `faces` |
| `VolumeElement` | `VolumeGridGeometry` (rectilinear 3-D block) | `cells` |

### Data Types (1.x)

`ScalarData`, `VectorData`, `MappedData`, `StringData`, `ColorArray`, etc.
The library itself does not restrict which data types are attached to which
geometry type.

### geoh5 Integration

**None.** The original library has no knowledge of geoh5; geoh5 support must
be added externally.

The companion visualisation package
[`omfvista`](https://github.com/OpenGeoVis/omfvista) provides one-way
OMF → PyVista conversion but is **not** maintained by GMG.

---

## 2. Mira Geoscience Fork (`MiraGeoscience/omf` → `mira-omf`)

### Purpose

A fork of the original omf-python library, maintained by
[Mira Geoscience](https://mirageoscience.com/), with **bidirectional OMF ↔
geoh5 conversion built directly into the `omf` package** as a new
`omf.fileio.geoh5` sub-module.

> **Note:** This fork is *not* maintained by GMG. It is intended for Mira's
> own interoperability needs with the `geoh5` file format used by Geoscience
> ANALYST.

### Architecture

The fork adds two new modules on top of the original library:

```
omf/
  fileio/
    __init__.py     # re-exports OMFReader, OMFWriter, GeoH5Writer, compare_elements
    fileio.py       # original OMFReader / OMFWriter unchanged
    geoh5.py        # NEW: class-based OMF ↔ geoh5 conversion (≈ 1 000 lines)
  scripts/
    omf_to_geoh5.py # NEW: CLI wrapper for GeoH5Writer
    geoh5_to_omf.py # NEW: CLI wrapper for GeoH5Reader + OMFWriter
```

### Key Class: `GeoH5Writer`

`GeoH5Writer` is a class that accepts an OMF `Project` (or any `UidModel`)
and a geoh5 file path, performs the conversion immediately during
construction, and stores the resulting geoh5 entity:

```python
from omf.fileio import OMFReader
from omf.fileio.geoh5 import GeoH5Writer

reader = OMFReader("model.omf")
project = reader.get_project()

writer = GeoH5Writer(project, "output.geoh5", compression=5)
workspace = writer()   # returns the Workspace object
```

Internally, `GeoH5Writer.__init__` calls `get_conversion_map(element, ...)` to
select the appropriate concrete `BaseConversion` subclass, then invokes
`converter.from_omf(element)`. All conversion logic is defined through an
inheritance hierarchy:

```
BaseConversion (ABC)
├── ProjectConversion     — omf.Project  ↔  geoh5py RootGroup
├── ElementConversion     — PointSetElement / LineSetElement / SurfaceElement / VolumeElement
│   └── SurfaceGridConversion  — SurfaceElement with SurfaceGridGeometry → Grid2D
├── DataConversion        — ScalarData / MappedData / StringData / etc. ↔ geoh5 Data
├── ArrayConversion       — omf arrays ↔ numpy arrays
├── ContainerGroupConversion — geoh5 groups (geoh5 → OMF only, flattened)
└── KnownUnsupported      — silently skips unsupported element types
```

A `_CONVERSION_MAP` dictionary maps each OMF / geoh5 type to the correct
conversion class at runtime.

### Key Class: `GeoH5Reader`

`GeoH5Reader` does the reverse: reads a geoh5 workspace and reconstructs an
`omf.Project`:

```python
from omf.fileio.geoh5 import GeoH5Reader
from omf.fileio import OMFWriter

reader = GeoH5Reader("workspace.geoh5")
project = reader()   # → omf.Project

OMFWriter(project, "output.omf")
```

### Supported Element Types (Mira fork)

Same four as the original library, with geoh5 equivalents:

| OMF Class | geoh5 Class | Special handling |
|---|---|---|
| `PointSetElement` | `geoh5py.objects.Points` | — |
| `LineSetElement` | `geoh5py.objects.Curve` | — |
| `SurfaceElement` (triangles, `SurfaceGeometry`) | `geoh5py.objects.Surface` | — |
| `SurfaceElement` (grid, `SurfaceGridGeometry`) | `geoh5py.objects.Grid2D` | Converted to a structured grid object (not triangulated) |
| `VolumeElement` | `geoh5py.objects.BlockModel` | — |

### Supported Data Types (Mira fork)

Per the fork README, the following data types are supported for OMF ↔ geoh5
conversion (significantly broader than geoh5-bridge):

| OMF Data Type | geoh5 Data Type | Notes |
|---|---|---|
| `ScalarData` (float) | `FloatData` | ✅ Full support |
| `ScalarData` (integer-like) | `IntegerData` | ✅ Supported; **no-data-values not in OMF standard** — integer arrays with NDV are converted to float on export |
| `StringData` | `TextData` | ✅ Full support |
| `MappedData` + `Legend` | `ReferencedData` + colormap | ✅ Full support including colour maps |

### CLI Script: `omf_to_geoh5.py`

Full source (64 lines):

```python
# omf/scripts/omf_to_geoh5.py
import argparse, logging, sys
from pathlib import Path
from omf.fileio import OMFReader
from omf.fileio.geoh5 import GeoH5Writer

_logger = logging.getLogger(__package__ + "." + Path(__file__).stem)

def main():
    parser = argparse.ArgumentParser(
        prog="omf_to_geoh5",
        description="Converts an OMF file to a new geoh5 file.",
    )
    parser.add_argument("omf_file", type=Path, help="Path to the OMF file to convert.")
    parser.add_argument("-o", "--out", type=Path, required=False, default=None,
        help="Path to the output geoh5 file. Defaults to same location with .geoh5 extension.")
    parser.add_argument("--gzip", type=int, choices=range(0, 10), default=5,
        help="Gzip compression level (0-9) for h5 data.")
    args = parser.parse_args()

    omf_filepath = args.omf_file
    if args.out is None:
        output_filepath = omf_filepath.with_suffix(".geoh5")
    else:
        output_filepath = args.out
        if not output_filepath.suffix:
            output_filepath = output_filepath.with_suffix(".geoh5")

    if output_filepath.exists():
        _logger.error("Cowardly refuses to overwrite existing file '%s'.", output_filepath)
        sys.exit(1)

    reader = OMFReader(str(omf_filepath.absolute()))
    GeoH5Writer(reader.get_project(), output_filepath, compression=args.gzip)
    _logger.info("geoh5 file created: %s", output_filepath)

if __name__ == "__main__":
    main()
```

Key observations about this script:

1. **Thin wrapper** — the entire conversion logic is inside `GeoH5Writer`. The
   script itself is purely argument parsing + file-existence guard.
2. **One-way** — only OMF → geoh5. The reverse direction has a separate script
   `geoh5_to_omf.py`.
3. **Compression control** — the `--gzip` option (default 5) passes a
   compression level to the HDF5 writer.
4. **Safety check** — refuses to overwrite an existing output file.
5. **Default output path** — if no `-o` argument is given, the output is
   placed next to the input file with a `.geoh5` extension.
6. **Installed as an entry-point** — registered as `omf_to_geoh5` (and
   `geoh5_to_omf`) in the `mira-omf` package so it is available directly
   from the command line after `pip install mira-omf`.

### CLI Script: `geoh5_to_omf.py`

The reverse direction. Equally thin:

```python
from omf.fileio import OMFWriter
from omf.fileio.geoh5 import GeoH5Reader

reader = GeoH5Reader(geoh5_filepath)
OMFWriter(reader(), str(output_filepath.absolute()))
```

---

## 3. geoh5-bridge (This Repository)

### Purpose

An **independent** Python library that provides bidirectional bridges between
OMF, PyVista, and geoh5 — all without being embedded inside either the `omf`
or `geoh5py` packages. It depends on the *original* `omf>=1.0` library
(not `mira-omf`).

### Architecture

Two conversion modules, both part of the `geoh5_bridge` package:

```
src/geoh5_bridge/
  omf_bridge.py          # OMF ↔ PyVista (via pyvista)
  omf_geoh5_bridge.py    # OMF ↔ geoh5 (direct, no PyVista required)
  utils.py               # shared helpers: _add_data_columns, _reconstruct_polylines
```

### Two Conversion Paths

#### Path A — OMF ↔ PyVista (`omf_bridge.py`)

Converts OMF elements to `pyvista` mesh objects and back. This path is useful
for 3-D visualisation, manipulation, and interoperability with the PyVista
ecosystem.

| Function | Direction | Input | Output |
|---|---|---|---|
| `omf_pointset_to_pyvista` | OMF → PV | `PointSetElement` | `pv.PolyData` (points) |
| `pyvista_to_omf_pointset` | PV → OMF | `pv.PolyData` | `PointSetElement` |
| `omf_lineset_to_pyvista` | OMF → PV | `LineSetElement` | `pv.PolyData` (lines) |
| `pyvista_to_omf_lineset` | PV → OMF | `pv.PolyData` | `LineSetElement` |
| `omf_surface_to_pyvista` | OMF → PV | `SurfaceElement` | `pv.PolyData` (triangles) |
| `pyvista_to_omf_surface` | PV → OMF | `pv.PolyData` | `SurfaceElement` |
| `omf_volume_to_pyvista` | OMF → PV | `VolumeElement` | `pv.RectilinearGrid` |
| `pyvista_to_omf_volume` | PV → OMF | `pv.RectilinearGrid` | `VolumeElement` |
| `omf_project_to_pyvista` | OMF → PV | `omf.Project` | `pv.MultiBlock` |
| `pyvista_to_omf_project` | PV → OMF | `pv.MultiBlock` | `omf.Project` |

PyVista is an *optional* extra (`pip install geoh5-bridge[omf]`), so none of
these functions are available unless it is installed.

#### Path B — OMF ↔ geoh5 Direct (`omf_geoh5_bridge.py`)

Converts OMF elements directly to `geoh5py` objects. PyVista is *not* required
for this path.

| Function | Direction | Input | Output |
|---|---|---|---|
| `omf_pointset_to_points` | OMF → geoh5 | `PointSetElement`, `Workspace` | `geoh5py.Points` |
| `points_to_omf_pointset` | geoh5 → OMF | `geoh5py.Points` | `PointSetElement` |
| `omf_lineset_to_curve` | OMF → geoh5 | `LineSetElement`, `Workspace` | `geoh5py.Curve` |
| `curve_to_omf_lineset` | geoh5 → OMF | `geoh5py.Curve` | `LineSetElement` |
| `omf_surface_to_surface` | OMF → geoh5 | `SurfaceElement`, `Workspace` | `geoh5py.Surface` |
| `surface_to_omf_surface` | geoh5 → OMF | `geoh5py.Surface` | `SurfaceElement` |
| `omf_volume_to_blockmodel` | OMF → geoh5 | `VolumeElement`, `Workspace` | `geoh5py.BlockModel` |
| `blockmodel_to_omf_volume` | geoh5 → OMF | `geoh5py.BlockModel` | `VolumeElement` |

### Supported Data Types

geoh5-bridge currently processes only **`ScalarData`** (floating-point
arrays). All other OMF data types (`MappedData`, `StringData`, `VectorData`,
etc.) are silently ignored during conversion.

The `_omf_scalar_data()` helper extracts only `ScalarData` children:

```python
def _omf_scalar_data(element) -> dict[str, tuple[np.ndarray, str]]:
    result = {}
    for d in element.data:
        if isinstance(d, _omf.ScalarData):       # ← only scalar data
            result[d.name] = (np.asarray(d.array, dtype=float), d.location)
    return result
```

An optional `data_names` keyword argument on every conversion function lets
callers select a subset of data channels.

### Grid Surface Handling

Both `omf_surface_to_pyvista()` and `omf_surface_to_surface()` handle **both**
`SurfaceGeometry` (explicit triangles) and `SurfaceGridGeometry` (structured
grid) inputs. For `SurfaceGridGeometry`, the library **triangulates the grid**
at conversion time: each grid cell is split into two triangles.

The resulting geoh5 `Surface` object stores the triangulated mesh as explicit
vertices + triangles (no implicit grid structure is preserved). This is
consistent and lossless for visualisation, but uses more storage than a
structured grid representation would.

In contrast, the Mira fork converts `SurfaceGridGeometry` to a `geoh5py.Grid2D`
object, which preserves the implicit grid structure.

### No CLI Scripts

geoh5-bridge is a pure Python library; it provides no command-line entry-points
for file conversion. Users call the conversion functions directly from Python:

```python
import omf
from geoh5py.workspace import Workspace
from geoh5_bridge import omf_pointset_to_points, omf_surface_to_surface

reader = omf.OMFReader("model.omf")
project = reader.get_project()

with Workspace.create("output.geoh5") as ws:
    for elem in project.elements:
        if isinstance(elem, omf.PointSetElement):
            omf_pointset_to_points(elem, ws)
        elif isinstance(elem, omf.SurfaceElement):
            omf_surface_to_surface(elem, ws)
```

---

## 4. Side-by-Side Comparison

### High-Level Summary

| Aspect | Original omf-python | Mira fork (`mira-omf`) | geoh5-bridge |
|---|---|---|---|
| **PyPI package** | `omf` | `mira-omf` | `geoh5-bridge` |
| **OMF model library** | ✅ (is the library) | ✅ (extends it) | Uses `omf>=1.0` as dep |
| **OMF → geoh5** | ❌ | ✅ (built-in) | ✅ (library functions) |
| **geoh5 → OMF** | ❌ | ✅ (built-in) | ✅ (library functions) |
| **OMF → PyVista** | ❌ | ❌ | ✅ |
| **PyVista → OMF** | ❌ | ❌ | ✅ |
| **CLI scripts** | ❌ | ✅ `omf_to_geoh5`, `geoh5_to_omf` | ❌ |
| **Float data** | n/a | ✅ | ✅ |
| **Integer data** | n/a | ✅ | ❌ |
| **String data** | n/a | ✅ | ❌ |
| **Referenced / colormap data** | n/a | ✅ | ❌ |
| **SurfaceGridGeometry** | n/a | → `Grid2D` (preserves structure) | Triangulated → `Surface` |
| **Compression control** | n/a | ✅ (gzip 0–9) | ❌ (geoh5py default) |
| **Overwrite protection** | n/a | ✅ (refuses) | ❌ (up to caller) |
| **Requires PyVista** | ❌ | ❌ | Optional (for PV path) |
| **Dependency on `omf`** | is `omf` | is `mira-omf` | `omf>=1.0` |

### Element-Level Conversion Comparison

| OMF Element | Mira → geoh5 | geoh5-bridge → geoh5 |
|---|---|---|
| `PointSetElement` | `geoh5py.Points` | `geoh5py.Points` |
| `LineSetElement` | `geoh5py.Curve` | `geoh5py.Curve` |
| `SurfaceElement` (triangles) | `geoh5py.Surface` | `geoh5py.Surface` |
| `SurfaceElement` (grid) | **`geoh5py.Grid2D`** (structured) | `geoh5py.Surface` (triangulated) |
| `VolumeElement` | `geoh5py.BlockModel` | `geoh5py.BlockModel` |

### Data-Level Conversion Comparison

| Data Type | Mira OMF → geoh5 | geoh5-bridge OMF → geoh5 |
|---|---|---|
| `ScalarData` (float) | ✅ `FloatData` | ✅ float64 array |
| `ScalarData` (integer) | ✅ `IntegerData` | ❌ (skipped) |
| `MappedData` + `Legend` | ✅ `ReferencedData` + colormap | ❌ (skipped) |
| `StringData` | ✅ Text | ❌ (skipped) |
| `VectorData` | ❌ (not listed) | ❌ (skipped) |

### Approach and Design Philosophy

| Aspect | Mira fork | geoh5-bridge |
|---|---|---|
| **Where conversion lives** | Inside the `omf` package itself | In a separate bridging package |
| **Conversion model** | Class hierarchy with `BaseConversion`, reflection-based `_CONVERSION_MAP` | Individual functions per element type |
| **Entry point** | `GeoH5Writer(project, path)` constructor | `omf_surface_to_surface(elem, ws)` etc. |
| **PyVista intermediate** | Not used | Optional intermediate layer for visualisation |
| **Bidirectional API** | Symmetric: `from_omf` / `from_geoh5` on each class | Separate function pairs |
| **Error handling** | Logs warnings and continues for unsupported types | Silently filters unsupported data types |
| **Workspace management** | `GeoH5Writer` creates and manages the workspace | Caller creates `Workspace`, passes it in |

---

## 5. Detailed Analysis of `omf_to_geoh5.py`

The [Mira script](https://github.com/MiraGeoscience/omf/blob/develop/omf/scripts/omf_to_geoh5.py)
is deliberately minimal (≈ 64 lines). Its entire job is to:

1. Parse CLI arguments (`omf_file`, `--out`, `--gzip`).
2. Determine the output file path (default: same directory, `.geoh5` extension).
3. Guard against accidental overwrites with an existence check.
4. Call `OMFReader(path).get_project()` to load the OMF project.
5. Call `GeoH5Writer(project, output_filepath, compression=gzip_level)` to
   perform the conversion.
6. Log the output path.

All the actual conversion work — element type dispatch, geometry extraction,
data type mapping, geoh5 entity creation — is in `omf/fileio/geoh5.py`, which
is roughly 1 000 lines of class-based Python.

The script becomes available as `omf_to_geoh5` on the command line after
installing `mira-omf`:

```bash
pip install mira-omf

# Convert an OMF file to geoh5 (same directory, default compression=5):
omf_to_geoh5 my_model.omf

# Specify output path and compression:
omf_to_geoh5 my_model.omf --out /data/output.geoh5 --gzip 9
```

Equivalent code using geoh5-bridge (no CLI, all in Python):

```python
import omf
from geoh5py.workspace import Workspace
from geoh5_bridge import (
    omf_pointset_to_points,
    omf_lineset_to_curve,
    omf_surface_to_surface,
    omf_volume_to_blockmodel,
)

reader = omf.OMFReader("my_model.omf")
project = reader.get_project()

with Workspace.create("my_model.geoh5") as ws:
    for elem in project.elements:
        if isinstance(elem, omf.PointSetElement):
            omf_pointset_to_points(elem, ws)
        elif isinstance(elem, omf.LineSetElement):
            omf_lineset_to_curve(elem, ws)
        elif isinstance(elem, omf.SurfaceElement):
            omf_surface_to_surface(elem, ws)
        elif isinstance(elem, omf.VolumeElement):
            omf_volume_to_blockmodel(elem, ws)
```

---

## 6. Notable Differences in `SurfaceGridGeometry` Handling

This is the most significant technical divergence between the two approaches.

### Mira fork

`SurfaceGridGeometry` is converted to a `geoh5py.Grid2D` object, which is a
native geoh5 structured-grid type. The implicit grid structure (origin,
spacing, orientation) is preserved:

```python
# Mira: SurfaceGridGeometry → Grid2D (preserves grid structure)
if isinstance(element, SurfaceElement) and isinstance(element.geometry, SurfaceGridGeometry):
    return SurfaceGridConversion(element, workspace, ...)
```

### geoh5-bridge

`SurfaceGridGeometry` is **triangulated** and stored as an explicit triangle
mesh in a `geoh5py.Surface` object. The triangulation logic computes node
positions from the grid parameters, then splits each cell into two triangles:

```python
# geoh5-bridge: SurfaceGridGeometry → triangulated Surface
nu = len(tensor_u) + 1   # node columns
nv = len(tensor_v) + 1   # node rows

for j in range(nv):
    for i in range(nu):
        pt = origin + u_edges[i]*axis_u + v_edges[j]*axis_v
        if offset_w is not None:
            pt[2] += offset_w[j * nu + i]
        vertices.append(pt)

for j in range(nv - 1):
    for i in range(nu - 1):
        v0, v1 = j*nu + i,   j*nu + (i+1)
        v2, v3 = (j+1)*nu+i, (j+1)*nu+(i+1)
        triangles.append([v0, v1, v3])
        triangles.append([v0, v3, v2])
```

**Trade-offs:**

| | Mira (`Grid2D`) | geoh5-bridge (`Surface` triangulated) |
|---|---|---|
| Preserves grid structure | ✅ | ❌ |
| Round-trip back to `SurfaceGridGeometry` | ✅ | ❌ (becomes explicit triangles) |
| Compatible with all triangle-mesh tools | ❌ (Grid2D only) | ✅ |
| Storage efficiency | ✅ (compact grid params) | ❌ (stores all vertices) |
| Z-offset (`offset_w`) support | ✅ | ✅ |

---

## 7. Summary

**Original `omf-python`** is a pure data-model library with no geoh5
integration. It is the foundation that both subsequent approaches build upon.

**Mira's `mira-omf` fork** embeds geoh5 conversion directly inside the
`omf` package, providing a comprehensive, bidirectional solution with support
for all major data types (float, integer, string, referenced/colormap). The
`omf_to_geoh5.py` script is a thin CLI wrapper that makes one-line batch
conversion possible from the command line. `SurfaceGridGeometry` is handled
correctly as a structured `Grid2D` object.

**`geoh5-bridge`** is an independent bridging library that adds value through
its **PyVista integration** (missing from both other approaches). It enables
OMF data to be visualised, manipulated, and exported back to OMF via PyVista
meshes — a workflow not available in the Mira fork. The direct OMF ↔ geoh5
path avoids the PyVista dependency for pure conversion workflows. The main
current limitations compared to Mira are the restriction to scalar (float)
data only and the lack of CLI entry-points. Grid surfaces are triangulated
rather than preserved as structured objects.

The two conversion approaches are largely complementary: geoh5-bridge excels
at PyVista-mediated workflows and is suitable when you need to manipulate
geometry before writing; the Mira fork excels at comprehensive, lossless
batch conversion with full data type support and a simple command-line
interface.
