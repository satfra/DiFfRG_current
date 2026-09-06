# Changelog

## Version 2.0

### Changed

- **Breaking:** the application-facing discretization, assembler, timestepper and output-session
  types are keyed on one another instead of repeating the linear algebra:
  `CG::Discretization<Model, RectangularMesh<dim>>`, `CG::Assembler<Discretization>`,
  `TimeStepperSUNDIALS_IDA<Assembler>`, `OutputSession<Assembler>`. The spelled-out parameter
  lists live on the `_impl` classes, whose instantiations stay closed in the library.
  **Migration:** collapse the aliases, e.g.
  `TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>` becomes
  `TimeStepperSUNDIALS_IDA<Assembler>`; `OutputSession<dim, VectorType>` becomes
  `OutputSession<Assembler>` (anything exposing `dim` and `VectorType` works, e.g. the
  discretization).
- The `QuadratureProvider`'s quadrature inventory is written into the run log, `<output name>.log`, rather than into a
  separate `<output name>_quadrature.log`.
- **Breaking:** Kokkos' host execution space is now `Kokkos::Serial`. DiFfRG's CPU parallelism is
  TBB -- deal.II's `MeshWorker` drives a `tbb::parallel_pipeline`, and every production flow
  instantiates its CPU integrators with `TBB_exec` -- so the Kokkos `std::threads` backend was a
  second pool of spinning workers contending for the same cores, which deal.II also dispatched its
  own vector kernels into. Build with `-DKOKKOS_THREADS=ON` to restore it.
- **Breaking:** `Threads_exec` / `Threads_memory` are renamed to `KokkosHost_exec` /
  `KokkosHost_memory`, which is what they are now that the backend behind them is Serial.
  **Migration:** rename the alias; nothing else changes. Generated flows follow automatically --
  codegen maps a `"Device" -> "Threads"` kernel to `KokkosHost_exec` (via `DeviceExecSpace[]` in
  the `TemplateParameterGeneration` package); `TBB` and `GPU` kernels are unchanged.
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
- `update()` is now a single overload taking a raw pointer; the `Kokkos::View` overload had no
  callers outside the removed twin refresh. It performs one host-side fill and one device copy
  behind a single fence, instead of the previous two fences per call.
- `operator[]` on the rank-2 and rank-3 interpolators is now explicitly row-major, matching the
  order `update()` takes its input in, independently of the mirror's Kokkos layout.
- `other_memory_space_t` is removed from `common/kokkos.hh`; the interpolators were its only users.

### Added

- **Breaking:** the old `install.sh` is removed (its former curl one-liner URL now 404s).
  `install_diffrg.sh` replaces it entirely: the wizard for easy installs, its
  `--mode source` path (or driving the superbuild directly with CMake) for everything
  `install.sh` did, including the `THREADS`/`FOLDER`/`MPI` knobs as flags.
- An interactive installer, `install_diffrg.sh`: a curl-able wizard covering the typical
  choices (pre-built dependency bundle vs. full self-build, prefix, build folder, MPI/GPU/
  MUMPS/documentation features, optional copy of examples and tutorials) with command-line
  flags for scripted use.
- `std::format` replaced by `fmt::format` (spdlog's bundled fmt) in the two places it was
  used, restoring GCC 12 compatibility: the compiler floor is GCC >= 12 everywhere, except
  nvcc host compilation where GCC 13 is excluded by an nvcc/libstdc++-13 bug (12 and >= 14
  both validated against the CUDA bundle).
- A CUDA dependency bundle variant (`linux-x86_64-v3-cuda12`, sm_80/Ampere floor with PTX
  forward-compatibility for newer GPUs) plus its `release-deps-linux-cuda` workflow;
  the wizard offers it when an NVIDIA GPU is detected. deal.II's nvcc-wrapper shim now
  installs into `bundled/bin` and self-relocates instead of recording a build-tree path.
- On-demand release workflows (`release-deps-linux`, `release-deps-macos`): build the
  pre-built dependency bundles on GitHub runners with the same scripts as a local release
  build, validate them, and upload workflow artifacts (optionally attached to a draft
  `deps-v*` release for manual publishing). The macOS/Apple-Silicon path is experimental.
- Pre-built binary dependency bundles: `install-diffrg-deps.sh` downloads a relocatable
  tarball of the full dependency superbuild (deal.II, Kokkos, Boost, TBB, SUNDIALS, HDF5, ...)
  from GitHub Releases (`deps-v*` tags), so only the DiFfRG library itself is compiled locally.
  Linux x86_64, CPU-only, compiled for `x86-64-v3` (AVX2+FMA), glibc ≥ 2.34. The release
  pipeline lives in `containers/release/`.
- A `MARCH` string option supersedes the binary `NATIVE` switch: `-DMARCH=x86-64-v3` (or any
  `-march=` value, or `none`) is threaded through every bundled dependency — including Boost,
  which previously received no architecture flag at all, so `NATIVE=ON` source builds now
  compile bundled Boost with `-march=native` too. `NATIVE=ON/OFF` keeps working unchanged.
- MPI support: FEM and FV flows can be distributed over ranks. In an MPI build a plain
  `RectangularMesh<dim>` is a partitioned triangulation and the linear algebra follows it (PETSc
  vectors and matrices); `RectangularMeshSerial<dim>` pins a mesh serial and
  `RectangularMeshParallel<dim>` names the partitioned one explicitly. Mesh and vector type must
  agree -- a mismatch would be silently wrong at runtime, so it is rejected at compile time with a
  `static_assert` naming the cause. LDG is structurally serial and requires
  `RectangularMeshSerial<dim>`. Output stays on rank 0: the timesteppers refresh ghosted
  `SolutionView` replicas outside `write_frame`, so no rank enters a collective the others never
  reach. Momentum integration distributes over ranks (and GPUs) through the SPMD `MapScheduler`,
  which splits the external grid plan-driven and bitwise-reproducibly.
- Structured runtime progress reporting: timesteppers, solvers and assemblers submit
  `ProgressEvent`s through their `ReportPort` instead of formatting console text.
  `/output/verbosity` selects what is shown -- 0 nothing, 1 residual/timestep work, 2 adds
  Jacobian, linear-solver and solver diagnostics, 3 adds factorization and output work. Levels
  1--4 are aggregated to at most one line per topic per second in both the console and the run
  log; 5 prints every event. Assemblers return a structured `SummaryEvent` from `summary()`,
  reported automatically when a run drains its output.
- Additional field series can be written next to the primary fields: `frame.fields("name")`
  returns a sink whose data goes to `<run>_name.pvd`/VTUs and, with HDF5 on, to
  `/name/<six-digit frame>` groups inside the single `<run>.h5`. Series names are one safe path
  component; `FE`, `potential`, `eom_potential`, `scalars`, `maps` and `coordinates` are reserved.
- FV runs can recover a continuous mass Hessian at nonanalytic interfaces:
  `/discretization/raw_potential_recover_mass_hessian` (default off) reconstructs it from the
  DG0 gradient models, with `/discretization/raw_potential_mass_hessian_jump_threshold` enabling
  one-sided recovery across Hessian jumps and `/discretization/raw_potential_order` selecting the
  reconstruction order. The order is also honored on the cached reconstruction path and
  invalidates the cached potential system when it changes (previously the cached path silently
  used order 2).
- The Mathematica tests run on AUMP (vendored under `Mathematica/AUMP`), which launches every test
  leaf in a fresh `wolframscript` kernel; the old `TestRunner.m` is gone. Run them via the
  `mathematica-test` build target (registered with ctest as `mathematica_tests`) or directly
  through `AUMP/Runner.wls` with `--init Mathematica/DiFfRG/Tests/init.m`.
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
