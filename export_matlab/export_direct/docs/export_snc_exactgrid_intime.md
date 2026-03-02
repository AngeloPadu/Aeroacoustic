# `export_snc_exactgrid_intime.py`

Export a PowerFLOW surface NetCDF (`.snc`) file to a MATLAB-friendly **HDF5-based** `.mat` file (MATLAB v7.3 style), streaming data frame-by-frame to handle large datasets.

This script is designed for *surface* data stored as “surfels” (polygons) with per-surfel / per-point measurements across time “sets”.

---

## What it exports

### Geometry
- **`V_m`**: `(nV, 3)` surface vertices in **meters**
- **`F`**: `(nTri, 3)` triangles (fan-triangulated from polygons), **1-based indexing** (MATLAB-friendly)
- **`tri_owner`**: `(nTri,)` integer surfel index for each triangle (**0-based**)
- **`centroids_m`**: `(nPoints, 3)` surfel centroids in **meters** (derived from vertices + polygon connectivity)

### Time / set selection
- **`set_indices`**: `(Nt,)` exported set indices

### Variables (time series)
For each selected variable `vname`:
- **`/fields/<vname>`**: `(nPoints, Nt)` values across exported sets

Optional (large output):
- **`/C/<vname>`**: `(nTri, Nt)` values mapped from points to triangles using `tri_owner` (`field[tri_owner]`)

### Metadata
- **`csys_names`**, **`lx_names`**, **`__var_names__`**: stored as UTF-8 string arrays
- Group **`/meta`** with attributes such as `csys_used`, `L_scale`, `U_scale`, `p_scale`, sizes, etc.

---

## Units and scaling

The NetCDF file is assumed to store values in **lattice units (LU)**. The script converts:

- `vertex_coords`:
  - apply selected coordinate system transform `T`
  - then `meters = LU * L_scale + L_off` using `LatticeLength` scale/offset

- Variables (via `convert_field()`):
  - `static_pressure`: `field_phys = field_lu * p_scale`
  - `x_velocity`, `y_velocity`, `z_velocity`: `field_phys = field_lu * U_scale`
  - all other variables: **no scaling** (passed through)

Where:
- `U_scale` comes from `LatticeVelocity`
- `rho_scale` comes from `LatticeDensity`
- `p_scale = rho_scale * U_scale^2`

> If your file already stores values in physical units, update `convert_field()` accordingly.

---

## Command line usage

### Basic export (default: set 0, `static_pressure`)
```bash
python export_snc_exactgrid_intime.py case.snc --mat case.mat
```

### Export multiple variables
```bash
python export_snc_exactgrid_intime.py case.snc \
  --mat case.mat \
  --vars static_pressure x_velocity y_velocity z_velocity
```

### Export all sets
```bash
python export_snc_exactgrid_intime.py case.snc --all-sets --vars all
```

### Export a set range (inclusive)
```bash
python export_snc_exactgrid_intime.py case.snc --first 10 --last 50 --vars static_pressure
```

---

## CLI options

- `input`: input `.snc` file
- `--mat`: output `.mat` (HDF5/v7.3 style). Default: `export_multi_v73.mat`
- `--csys`: coordinate system selection:
  - `base_frame` (default)
  - `lattice_csys`
  - `default_csys`
  - `duct_csys`
- `--invert-csys`: invert the selected 4×4 transform matrix

Variables:
- `--vars`: list of variable short names, OR `all`
  - also supports comma form: `--vars static_pressure,x_velocity`

Set selection:
- `--set`: single set index (default `0`) used when not exporting a range/all
- `--all-sets`: export all sets
- `--first`, `--last`: inclusive set range (if provided, overrides `--set`)

Output tuning:
- `--dtype`: `float32` (default) or `float64`
- `--export-C`: also write `/C/<var>` mapped-to-triangle datasets (can be **very large**)
- `--chunk-frames`: HDF5 chunk size along time dimension (default `8`)
- `--point-chunk`: HDF5 chunk size along point dimension (default `200000`)

---

## Output file layout (HDF5)

This `.mat` is an HDF5 file (MATLAB v7.3 style). Common keys:

- `V_m`, `F`, `tri_owner`, `centroids_m`, `set_indices`
- `__var_names__`, `csys_names`, `lx_names`
- group `fields/` containing datasets for each variable
- group `C/` (only if `--export-C`)
- group `meta/` with attributes (scales, sizes, csys used)

---

## MATLAB example

```matlab
S = matfile("case.mat");
V = S.V_m;
F = S.F;

p = S.fields.static_pressure;    % note: accessing groups may differ depending on workflow
```

(If you prefer `load()`, note that v7.3 `.mat` files are often accessed with `matfile` for partial loading.)

---

## Known quirks / maintenance notes

- `F` is **1-based**, `tri_owner` is **0-based**.
- The script stores `centroids_m` for convenience; it is derived from connectivity.
- The file contains a duplicated script body appended after `main()` (starts with a second shebang). Functionality still works, but you may want to delete the duplicate tail to keep the source clean.
