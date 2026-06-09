# Project Structure

## Core Pipeline

DiFfRG follows a **Model → Discretization → Assembler → TimeStepper → Output** pipeline:

1. **Model** — The user defines the physics by inheriting from `AbstractModel<Model, Components>` (CRTP). A model specifies the mass function, flux, source terms, and optionally numerical fluxes and boundary conditions.

2. **Components** — A descriptor type that declares how many FE functions (and their polynomial orders), extractors, and additional variables the model uses.

3. **Discretization / Assembler** — An assembler takes the model and a mesh, then computes residuals and Jacobians using a chosen spatial discretization:
   - **CG** (Continuous Galerkin)
   - **DG** (Discontinuous Galerkin)
   - **DDG / LDG** (Direct/Local Discontinuous Galerkin, for higher-order derivatives)
   - **FV** (Finite Volume, Kurganov-Tadmor scheme)

4. **TimeStepper** — Evolves the discretized system in RG time. Options include:
   - *Explicit*: Forward Euler, Boost Runge-Kutta, Adams-Bashforth-Moulton
   - *Implicit*: SUNDIALS IDA, implicit Euler, TRBDF2
   - Linear/nonlinear solvers are configured underneath (UMFPack, GMRES, Newton, KINSOL).

5. **Output** — Results are written via `DataOutput` to CSV, HDF5, or VTK formats at configurable intervals.

## Directory Layout

```
DiFfRG/
├── include/DiFfRG/
│   ├── common/          Math utilities, JSON config, quadrature, Kokkos wrappers
│   ├── discretization/  Assemblers (CG/DG/FV), mesh, coordinates, data output
│   ├── model/           AbstractModel (CRTP), component descriptors, numerical fluxes
│   ├── physics/         Integrators, interpolators, regulators, threshold functions
│   └── timestepping/    Explicit/implicit steppers, linear/nonlinear solvers
├── src/DiFfRG/          Corresponding .cc source files
├── tests/               Catch2 test sources
├── python/              Post-processing (VTK/HDF5 I/O, plotting)
├── Mathematica/         Symbolic computation and C++ code generation
└── documentation/       Doxygen config and guides
```

## Configuration

Simulations are configured via a `parameter.json` file, which is read by `ConfigurationHelper`. Key sections:

- `/physical/` — Physics parameters (temperature, couplings, etc.)
- `/integration/` — Quadrature orders and tolerances for momentum integrals
- `/discretization/` — FE order, grid specification, adaptivity settings
- `/timestepping/` — Final time, output interval, explicit/implicit solver tolerances
- `/output/` — Output folder, name, verbosity

CLI flags (`-sd`, `-si`, `-sb`, `-ss`) can override any JSON parameter at runtime.

## Key Design Patterns

- **CRTP** (Curiously Recurring Template Pattern) for zero-overhead polymorphism in models and assemblers.
- **Kokkos execution spaces** (`GPU_exec`, `TBB_exec`, `Threads_exec`) abstract the CPU/GPU backend.
- **C++20 concepts** constrain template parameters with clear diagnostics.
- **Automatic differentiation** computes Jacobians without hand-coded derivatives.

## Parallelization (Kokkos)

DiFfRG uses [Kokkos](https://kokkos.org/) as a performance-portability layer, so the same kernel code runs on different backends selected through *execution spaces*:

- `TBB_exec` — CPU parallelization via [oneTBB](https://github.com/uxlfoundation/oneTBB).
- `Threads_exec` — CPU parallelization via Kokkos' native C++ threads backend.
- `GPU_exec` — GPU execution via Kokkos' CUDA (NVIDIA) or HIP (AMD) backend.

The GPU backend is enabled at configure time with `-DGPU=ON` (the default; see the [installation](installation.md) page). Whether a particular computation runs on the CPU or the GPU is chosen per execution space: the **FEM assemblers always run on the CPU (`TBB_exec`)**, because they interact with deal.II's data structures, while the **momentum-space integrators can target the GPU** for a large speedup on fully momentum-dependent flows. The integrator's execution space is fixed when its kernel is generated (the `"Device"` option of `MakeKernel`, see [Tutorial 3](../tutorials/tut3.md)); `TBB` is the default and is required when an integrator is called from within an assembler.

Because the backend is abstracted, models and generated kernels are written once (using the `KOKKOS_FORCEINLINE_FUNCTION` markers in the generated code) and compile for whichever backend is enabled.
