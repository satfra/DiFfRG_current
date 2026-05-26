# Tutorial 4: FV diagnostics dashboard {#tut_fv_diagnostics}

This tutorial shows how to inspect one-dimensional finite-volume Kurganov-Tadmor output with the small
`diffrg-fv-diagnostics` Python package. The package reads DiFfRG diagnostic HDF5 files and opens a Matplotlib dashboard
for the cell state, reconstructed face values, face slopes, fluxes, and residual contributions.

The workflow has two parts:

1. run a 1D FV/KT simulation with diagnostic HDF5 output enabled;
2. open the generated HDF5 files locally, or copy them from a cluster through SSH.

The diagnostic output is meant for debugging reconstruction, convexity, and residual problems. It is not a replacement
for ordinary production output.

## Enable diagnostic HDF5 output

The dashboard expects HDF5 maps written by the FV/KT assembler. Enable ordinary HDF5 output and the two diagnostic
streams in your parameter file:

```json
{
  "output": {
    "hdf5": true,
    "vtk": false,
    "fv_reconstruction_diagnostics": true,
    "fv_residual_contribution_diagnostics": true
  },
  "timestepping": {
    "output_dt": 1e-5
  }
}
```

`output_dt` controls how densely the diagnostic history is written. For a failing run, choose it small enough that at
least a few snapshots appear before the failure. For a long production run, keep it larger; these files can grow quickly.

The current diagnostic writer is implemented for one-dimensional FV/KT models. It writes on the serial output path, so it
does not require changes to residual or Jacobian callbacks.

## Run the simulation

Run your application as usual, for example:

```bash
./your_kt_application -p parameter.json
```

With an output name such as `run`, the diagnostic files are written next to the ordinary output as:

```text
run_fv_reconstruction_diagnostics.h5
run_fv_residual_contribution_diagnostics.h5
```

The reconstruction file contains cell values, slopes, reconstructed face states, and KT face fluxes. The residual
contribution file contains the advection, diffusion, source, mass, and total residual contributions. The dashboard loads
the matching sibling file automatically when both files are present.

## Open a local dashboard

From the repository root, run:

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  /path/to/run_fv_reconstruction_diagnostics.h5 \
  --xlim 0.015:0.035
```

If `--output` is omitted, the dashboard opens interactively. The sigma slider changes the visible field-space window, and
the time slider moves through available HDF5 series. Cell boundaries are drawn as vertical dotted lines in the sigma
panels.

To save a static figure instead of opening a window, pass an output path:

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  /path/to/run_fv_reconstruction_diagnostics.h5 \
  --series 0,25,50,100 \
  --output fv_diagnostics.png
```

Use `--series` for exact HDF5 series numbers or `--times` for nearest available RG times. Use `--component` when the
model has multiple components and you want a component other than zero.

## Open files from a cluster

Cluster files can be opened with scp-style SSH paths. The dashboard copies the HDF5 file into a local cache first and
then reads it locally, which is much more reliable than streaming random-access HDF5 data over SSH.

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  itp:path/to/run_fv_reconstruction_diagnostics.h5 \
  --xlim 0.015:0.035
```

The host part can be an SSH config alias. For example, `itp:path/to/file.h5` uses the `Host itp` entry in
`~/.ssh/config`. Absolute remote paths also work:

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  itp:/scratch/project/run_fv_reconstruction_diagnostics.h5
```

Remote files are cached below:

```text
${XDG_CACHE_HOME:-~/.cache}/diffrg-fv-diagnostics/
```

If the cluster file changed and you want to copy it again, add `--refresh`.

## What the panels show

The main panels are:

- **Cell state and reconstruction**: compares the cellwise state with the linear reconstruction used at faces.
- **Face slopes**: shows left and right reconstructed slopes at faces. The y-axis is displayed relative to the convexity
  bound when this is useful.
- **Advection vs diffusion masses**: shows the pion-like and sigma-like convexity margins used by the dashboard.
- **Face fluxes**: splits KT face fluxes into advection, diffusion, and total flux.
- **Residual flux contributions** and **local residual contributions**: show which term dominates the assembled residual.
- **Residual check**: compares the stored total contribution with the residual passed through the output path when
  available.

Red warning lines mark the running `-k^2` or `-k^2 sigma` bounds derived from `Lambda`. The dashboard reads `Lambda` and
other metadata from the HDF5 `configuration_json` attribute when available.

For quark-meson diagnostics with explicit symmetry breaking, new HDF5 files store `physical.cSigma` in the metadata. For
older files, pass it explicitly:

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  /path/to/run_fv_reconstruction_diagnostics.h5 \
  --advection-offset 0.001695
```

## Trace an oscillation or convexity failure

For failing runs, enable the oscillation trace:

```bash
uv run --project DiFfRG/python/fv_diagnostics diffrg-fv-dashboard \
  /path/to/run_fv_reconstruction_diagnostics.h5 \
  --oscillation-trace \
  --xlim 0.015:0.035
```

This prints a CSV-like table to the terminal and adds history panels for:

- the smallest `k^2 + m^2` convexity margin;
- the largest local cell oscillation indicators;
- the dominant residual contribution when the residual diagnostic file is available.

Useful options are:

```bash
--convexity-margin 0.1
--trace-count 20
--advection-offset 0.001695
```

Use the printed `sigma` and series number to jump to the failing region in the dashboard. In practice, this is often the
fastest way to answer whether a failure starts in the reconstructed state, the face slopes, a flux term, or a local source
term.

## Common problems

- If no supported maps are found, check that `output.hdf5` and the FV diagnostic flags are enabled.
- If the sibling residual file is missing, the dashboard still opens the reconstruction file but residual-attribution
  panels are unavailable.
- If cluster data looks stale, rerun with `--refresh`.
- If a 2D or 3D FV run throws a diagnostic error, disable these flags; this diagnostic dashboard currently targets 1D
  KT output.
