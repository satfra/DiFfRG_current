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
`min(k^2 + m_sigma^2)` and the cell oscillation indicator. If the matching residual
diagnostic file is next to the reconstruction file, the table also reports the
dominant local residual contribution.
