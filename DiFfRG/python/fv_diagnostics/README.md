# DiFfRG FV diagnostics

Standalone plotting and inspection tools for DiFfRG finite-volume diagnostic HDF5 output.

For a full workflow, including how to enable the diagnostic HDF5 output in a 1D FV/KT run, see
`DiFfRG/documentation/tutorials/fv_diagnostics.md`.

## FV dashboard

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  /path/to/run_fv_reconstruction_diagnostics.h5 \
  --xlim 0.015:0.035
```

When the matching residual-contribution HDF5 file is next to the reconstruction file, it is loaded automatically.
Interactive mode always includes sigma and time sliders.
Cluster files can be opened through SSH by using scp-style paths. The file is copied into
`${XDG_CACHE_HOME:-~/.cache}/diffrg-fv-diagnostics/` and then opened locally:

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  itp:path/to/run_fv_reconstruction_diagnostics.h5 \
  --xlim 0.015:0.035
```

SSH config aliases are resolved by `rsync`/`scp`, so aliases such as `itp` work when they
are defined in `~/.ssh/config`. Pass `--refresh` to re-copy a changed cluster file.

## Oscillation trace

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  /path/to/run_fv_reconstruction_diagnostics.h5 \
  --oscillation-trace --xlim 0.015:0.035
```

This prints the first dangerous convexity-margin slices and adds history panels for
the critical `k^2 + m^2` margin and the cell oscillation indicator. If the matching residual
diagnostic file is next to the reconstruction file, the table also reports the
dominant local residual contribution.

For quark-meson diagnostics, the advection/pion margin uses
`k^2 + (u + c_sigma) / sigma`. New HDF5 files provide `c_sigma` through
metadata; for older files pass it explicitly:

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  /path/to/run_fv_reconstruction_diagnostics.h5 \
  --oscillation-trace --advection-offset 0.001695
```
