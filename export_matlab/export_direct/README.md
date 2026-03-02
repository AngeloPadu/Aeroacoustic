# PowerFLOW NetCDF → MATLAB export utilities

Small utilities for exporting PowerFLOW-style NetCDF outputs to MATLAB `.mat`.

## Scripts

- [`export_snc_exactgrid_intime.py`](docs/export_snc_exactgrid_intime.md) — surface `.snc` → v7.3-style `.mat` with streaming export
- [`fnc_to_mat.py`](docs/fnc_to_mat.md) — `.fnc/.nc` → physical units + structured gridding + `.mat`

## Quick start

```bash
python export_snc_exactgrid_intime.py case.snc --mat case.mat --vars all --all-sets
python fnc_to_mat.py data.fnc --first 0 --last 10 --mat export_timeseries.mat
```

## License

Add a `LICENSE` file (MIT/Apache-2.0/GPL-3.0, etc.).
