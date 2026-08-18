# Changelog

## Unreleased

### Changed

- **Breaking:** the interpolators (`LinearInterpolator1D/2D/3D`, `LinearInterpolatorND`,
  `SplineInterpolator1D`, `SplineInterpolator1DStack`) no longer take a memory-space template
  parameter. One object now holds its data on both host and device, `update()` leaves both current,
  and the call operator evaluates on whichever side it is invoked from. Migration: delete the third
  template argument, e.g. `SplineInterpolator1D<double, Coordinates1D, GPU_memory>` becomes
  `SplineInterpolator1D<double, Coordinates1D>`. A single generated flow set can now be driven
  through `GPU_exec` and `TBB_exec` without changing the interpolator declarations.
- **Breaking:** `CPU()`, `GPU()` and `get_on<MemorySpace>()` are removed, along with the lazily
  allocated twin object they returned. Call the interpolator directly instead; a host-side
  `interpolator(x)` now reads the host mirror rather than requiring `interpolator.CPU()(x)`.
  The `is_interpolator` concept no longer requires `get_on<>`.
- **Breaking:** `data()` is now `const` and returns `const NT *`. Writing through it would have left
  the device buffers (and, for the splines, the coefficients) stale, and there is no longer a public
  way to push such an edit. Route mutations through `update()`.
- `update()` is now a single overload taking a raw pointer; the `Kokkos::View` overload had no
  callers outside the removed twin refresh. It performs one host-side fill and one device copy
  behind a single fence, instead of the previous two fences per call.
- `operator[]` on the rank-2 and rank-3 interpolators is now explicitly row-major, matching the
  order `update()` takes its input in, independently of the mirror's Kokkos layout.
- `other_memory_space_t` is removed from `common/kokkos.hh`; the interpolators were its only users.

## Version 1.0.1

### Fixed

- There was a bug with TaskFlow internally in deal.ii. Fixed for now by simply disabling taskflow in deal.ii.
- deal.ii changed its interface for dealii::SolutionTransfer. Adapted the corresponding methods.

### Changed

- The FlowingVariables classes are now in separate namespaces. For finite elements, use DiFfRG::FE::FlowingVariables, for pure variable systems use DiFfRG::FlowingVariables.
