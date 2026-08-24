# Changelog

## Version 2.0

### Changed

- `/output/verbosity` now also governs the run log's console echo, which previously ignored it and
  printed everything at `/output/log_level`: at 0 the console shows only warnings and errors, at 1
  the ordinary messages, from 2 upwards also `debug` records. The `.log` file is unaffected and
  still receives everything. As a side effect the per-frame output timing line, which is emitted at
  `debug` level from `/output/verbosity` 3, actually reaches the console now instead of being
  dropped by the default `info` logger level.
- **Breaking:** Kokkos' host execution space is now `Kokkos::Serial`. DiFfRG's CPU parallelism is
  TBB -- deal.II's `MeshWorker` drives a `tbb::parallel_pipeline`, and every production flow
  instantiates its CPU integrators with `TBB_exec` -- so the Kokkos `std::threads` backend was a
  second pool of spinning workers contending for the same cores, which deal.II also dispatched its
  own vector kernels into. Build with `-DKOKKOS_THREADS=ON` to restore it.
- **Breaking:** `Threads_exec` / `Threads_memory` are renamed to `KokkosHost_exec` /
  `KokkosHost_memory`, which is what they are now that the backend behind them is Serial.
  **Migration:** rename the alias; nothing else changes. Generated flows are unaffected -- codegen
  only ever emits `TBB_exec` and `GPU_exec`.
- **Breaking:** `/discretization/kokkos_threads` is removed. There is one CPU thread pool and
  `/discretization/threads` sizes it. DiFfRG warns if the key is still present.
- The CPU thread count is resolved from a documented precedence order, and DiFfRG warns loudly on
  stderr when several settings disagree: `DiFfRG_NUM_THREADS` > the launcher's allocation (the CPU
  affinity mask, else `SLURM_CPUS_PER_TASK` / `SLURM_JOB_CPUS_PER_NODE` / `SLURM_CPUS_ON_NODE`) >
  `/discretization/threads` > `OMP_NUM_THREADS` / `DEAL_II_NUM_THREADS` / `KOKKOS_NUM_THREADS` >
  automatic. `DEAL_II_NUM_THREADS` previously capped every one of these silently, from inside
  deal.II; it is now removed from the environment where it would undercut a higher-priority setting.
- The resolved budget is published as `DiFfRG::n_threads()` (`DiFfRG/common/threads.hh`) and is what
  the assembly schedule and the map scheduler's host/device split size themselves against. Prefer it
  to `dealii::MultithreadInfo::n_threads()`, which is a mutable static that any `set_thread_limit()`
  call rewrites.
- **Breaking:** the interpolators (`LinearInterpolator1D/2D/3D`, `LinearInterpolatorND`,
  `SplineInterpolator1D`, `SplineInterpolator1DStack`) no longer take a memory-space template
  parameter. One object now holds its data on both host and device, `update()` leaves both current,
  and the call operator evaluates on whichever side it is invoked from. **Migration**: delete the third
  template argument, e.g. `SplineInterpolator1D<double, Coordinates1D, GPU_memory>` becomes
  `SplineInterpolator1D<double, Coordinates1D>`. A single generated flow set can now be driven
  through `GPU_exec` and `TBB_exec` without changing the interpolator declarations.
- **Breaking:** `CPU()`, `GPU()` and `get_on<MemorySpace>()` are removed, along with the lazily
  allocated twin object they returned. Call the interpolator directly instead; a host-side
  `interpolator(x)` now reads the host mirror rather than requiring `interpolator.CPU()(x)`.
  The `is_interpolator` concept no longer requires `get_on<>`.
- **Breaking:** Interpolator's `data()` is now `const` and returns `const NT *`. Writing through it would have left
  the device buffers (and, for the splines, the coefficients) stale, and there is no longer a public
  way to push such an edit. Route mutations through `update()`.

### Added

- Every HDF5 file a run writes now carries its configuration as a browsable `/config` group -- one
  subgroup per configuration section, one attribute per leaf -- so a run's parameters can be read
  with the same tools that read its data (`h5ls -r out.h5/config`,
  `f["/config/physical"].attrs["Lambda"]`) instead of parsing JSON. This covers the run-level
  `<name>.h5` and any secondary file obtained from `OutputSession::hdf5(name)`. HDF5 has no boolean
  type, so booleans appear as 0/1 ints, arrays as their serialized JSON, and nulls are omitted.
- `/output/json` (default `false`) forces the `<name>.log.json` copy of the configuration to be
  written.
- `/timestepping/implicit/jacobian_diagnostics` (default `false`) switches the Jacobian diagnostics
  tables on. They used to be written unconditionally, which is not free: every Jacobian build paid a
  full sweep over the assembled matrix and, with a factorizing linear solver, a condition estimate
  costing several extra triangular solves. With the switch off none of that work happens and no
  `<run>_jacobian_diagnostics.csv` is created.
- HDF5 files record how their run ended, in two root attributes written when the output session
  closes. `finished` is 0 from the moment the file is created and becomes 1 once the session has
  flushed and closed everything, so a run that never got that far -- SIGKILL, OOM, node eviction --
  is still marked 0. `crashed` says whether the session closed because an exception was unwinding
  through it: a timestepper that gave up leaves `finished = 1, crashed = 1`, because its output
  *was* written out properly, it just stopped early. The three readable states are therefore
  `(0, 0)` killed or still running, `(1, 0)` ran to completion, `(1, 1)` stopped early but closed
  cleanly. `/scalars/time` tells you how far it got.
- `hdf5lib`'s `Group` gains `has_attribute`, `delete_attribute` and `overwrite_attribute`. The
  last is deliberately separate from `write_attribute`, which still fails on a name that is
  already taken -- for most attributes a second write is a bug, and that throw is what catches it.

### Removed

- **Breaking:** the `configuration_json` root attribute is no longer written to HDF5 files. The
  configuration is now in the `/config` group instead. **Migration:** replace
  `json.loads(f.attrs["configuration_json"])["physical"]["Lambda"]` with
  `f["/config/physical"].attrs["Lambda"]`.
- **Breaking:** `<name>.log.json` is no longer written when HDF5 output is on, since `/config`
  already records the configuration. **Migration:** read it from the `.h5` file, or set
  `/output/json` to `true` to get the file back. `SimulationData1D.params`
  (`DiFfRG/python/DiFfRG/file_io/vtk.py`) is `None` when the file is absent rather than raising.


## Version 1.1.0

### Changed

- Build system has been migrated fully to CMake-based, being now much faster, with much better stability and robustness. Preflight checks make sure that the requirements are met before the superbuild starts.

## Version 1.0.1

### Fixed

- There was a bug with TaskFlow internally in deal.ii. Fixed for now by simply disabling taskflow in deal.ii.
- deal.ii changed its interface for dealii::SolutionTransfer. Adapted the corresponding methods.

### Changed

- The FlowingVariables classes are now in separate namespaces. For finite elements, use DiFfRG::FE::FlowingVariables, for pure variable systems use DiFfRG::FlowingVariables.
