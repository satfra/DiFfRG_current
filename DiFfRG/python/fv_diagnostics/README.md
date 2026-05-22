# DiFfRG FV diagnostics

Standalone plotting and inspection tools for DiFfRG finite-volume diagnostic HDF5 output.

## FV dashboard

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  /path/to/run_fv_reconstruction_diagnostics.h5 \
  --xlim 0.015:0.035
```

When the matching residual-contribution HDF5 file is next to the reconstruction file, it is loaded automatically.
Interactive mode always includes sigma and time sliders.

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
