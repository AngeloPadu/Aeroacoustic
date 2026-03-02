# `fnc_to_mat.py`

Convert a PowerFLOW NetCDF (`.fnc` / `.nc`) dataset from **lattice units → physical units**, snap points onto an **exact structured grid**, and export **gridded time series** to MATLAB `.mat`.

This script targets datasets where `vertex_coords` already lie on a structured grid (possibly with tiny floating errors), and time is stored as discrete NetCDF “sets”.

---

## What it exports

- **`coords_m`**: `(nPoints, 3)` coordinates in **meters** (after coordinate transform + scaling)
- **`X_m`**, **`Y_m`**, optional **`Z_m`**: unique sorted grid axes in meters
- **`set_indices`**: exported set indices `(Nt,)`
- **`__var_names__`**: list of variable short names
- **`__scales__`**: dictionary with base scaling factors used (length, density, velocity, derived pressure scale)
- One gridded array per variable name:
  - If 2D: `(Ny, Nx, Nt)`
  - If 3D: `(Ny, Nx, Nz, Nt)` with spatial order `(y, x, z)` (MATLAB-friendly)

---

## How the gridding works

1. Convert coordinates to meters (see below).
2. Build axes using snapped unique values:
   - `X_m = unique(round(x, decimals))`
   - `Y_m = unique(round(y, decimals))`
   - `Z_m = unique(round(z, decimals))`
3. If `Z_m` has 0/1 unique value, the script treats the dataset as **2D** and omits `Z_m`.
4. Each point is mapped to a grid cell using `np.searchsorted()` on the axes.
   - If a point falls outside the axes, the script errors and suggests increasing `--round-decimals` (or the data are not structured).
   - If multiple points map to the same cell, a warning is printed and **the last point overwrites earlier ones**.

---

## Units and scaling

### Coordinates
- The script selects the coordinate system named **`lattice_csys`** from `csys_names` and applies its `4×4` transform.
- Then it converts LU → meters using `LatticeLength` (`coords_m = coords_lu * L_scale + L_off`).

> If your file does not contain `lattice_csys`, or you need a different frame, edit the selection in `main()`.

### Variables
For each set, the script performs:
- direct conversion: `meas_phys = meas_lu * scales + offsets` (per-variable, derived from `variable_lattice_unit_names` and the `lx_*` maps)
- LBM overrides when the direct scale is effectively `1`:
  - velocities (`x_velocity/y_velocity/z_velocity`, plus a few aliases) → multiply by `U_scale`
  - `static_pressure` → multiply by `p_scale = rho_scale * U_scale^2`

---

## Command line usage

```bash
python fnc_to_mat.py data.fnc --first 0 --last 100 --mat export_timeseries.mat
```

---

## CLI options

- `input`: path to `.fnc/.nc`
- `--first`: first set/snapshot (inclusive). Default `0`
- `--last`: last set/snapshot (inclusive). Default = `first`
- `--mat`: output `.mat` file name. Default `export_timeseries.mat`
- `--outdir`: output directory for plots (only generated for the first exported set). Default `out_plots`
- `--round-decimals`: coordinate snapping precision. Default `12`
- `--dtype`: `float32` (default) or `float64` for exported gridded arrays
- `--no-plots`: disable plots
- `--no-v73`: force MATLAB v5 writer (SciPy). By default it tries v7.3 via `hdf5storage` and falls back to v5.

---

## MATLAB notes

- v7.3 `.mat` is HDF5-based; use `matfile()` for partial loading if arrays are large.
- Gridded arrays are stored in `(y, x, z, t)` order (or `(y, x, t)` for 2D).

---

## Known quirks / maintenance notes

- **Hard-coded coordinate system**: the script currently picks `lattice_csys` with:
  ```python
  idx = names.index("lattice_csys")
  T = csys[idx]
  ```
  Update if you need a different coordinate system.

- **Debug `sys.argv` override**: at the bottom, `sys.argv` is overwritten before `main()` is called. Remove that block for normal CLI use.
