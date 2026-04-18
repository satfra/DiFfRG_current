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
