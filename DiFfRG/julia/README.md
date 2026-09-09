# DiFfRG.jl

A Julia interface to [DiFfRG](https://github.com/satfra/DiFfRG).

This package drives DiFfRG applications: it locates an installation, builds an
app against it, runs it over a parameter grid, and reads the results back.

```julia
using DiFfRG

app = App("Examples/ONfiniteT"; target = "CG")

res  = run(app; params = (T = 0.1,))
data = only(read_h5(res))
data.fe["FE"].fields["u"]        # one array per output frame

sweep = scan(app; T = 0.01:0.01:0.2, muq = [0.0, 0.3])
```

## Why a sweep is cheap

Every key of a parameter file can be set on the command line by JSON pointer, so
`scan` builds the app **once** and runs one process per point. Nothing is
recompiled between points.

Two ways to set a parameter, differing in where the value ends up:

- `params` becomes command-line overrides. Use it for what a sweep varies.
- `config` is merged into the parameter file the run is given. Use it for what a
  sweep holds fixed.

A bare key means `/physical/<key>`; a key starting with `/` is a JSON pointer to
anywhere in the tree.

```julia
run(app; params = (T = 0.1,),
         config = ("/timestepping/final_time" => 5.0,
                   "/discretization/grid/x_grid" => "0:1e-3:1"))
```

## Requirements

A DiFfRG installation, plus CMake and a C++20 compiler — apps are compiled
locally because DiFfRG is a static library.

If you have no DiFfRG yet, the package will install one for itself:

```julia
using DiFfRG
install_diffrg!()          # ~3.5 min: downloads the bundle, builds the library
```

It downloads the prebuilt dependency bundle for this platform and builds only
the library against it — minutes, not the hours a full dependency build takes.
The result lands in a Julia scratch space (`~/.julia/scratchspaces/<uuid>/diffrg`,
about 650 MB), so it survives package updates, is found automatically, and is
removed with the package. `uninstall_diffrg!()` removes it now.

This drives `install_diffrg.sh` rather than reimplementing it, so it produces the
same bytes a manual install would. Pass `gpu = true`, `mpi = true`, or
`mode = "source"` — prebuilt bundles cover Linux x86_64 and macOS arm64 only, and
are built without MPI. CMake and a C++20 compiler are required either way, since
apps are compiled locally regardless.

**If you already have DiFfRG installed, you do not need this.** The search order
puts your own install ahead of the one this package would make for itself:

1. `$DIFFRG_DIR` — explicit, wins over everything
2. a bound artifact, if any
3. `~/.local/share/DiFfRG` — where `install_diffrg.sh` puts it by default
4. the package's own install, as a fallback

`which_install()` prints that list and marks the one in use. `install_diffrg!()`
warns if something else already takes precedence, rather than leaving you to
wonder which DiFfRG a build picked up.

To use an install somewhere else:

```bash
bash <(curl -sL https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh)
```

Prebuilt bundles are published per platform, but Julia's platform triplet cannot
express the `x86-64-v3` (AVX2) and glibc 2.34 floors they are built against, so
`check_bundle_runnable()` performs those checks itself — `DIFFRG_SKIP_CPU_CHECK=1`
is the escape hatch, and `DIFFRG_VARIANT` selects cpu vs cuda12. Note that the
published bundles carry *dependencies only*: an artifact cannot supply a complete
install until the release pipeline also packages `libDiFfRG.a` and the headers.

## Build times, measured

| | |
|---|---|
| one app, compile + link | ~67 s |
| link alone | 0.23 s |
| CMake configure | 0.67 s |
| rebuild with nothing changed | 25 ms (cache hit) |
| a parameter scan point | no rebuild at all |

Compilation is essentially the whole cost, and 71% of it is `<DiFfRG/DiFfRG.hh>`.
That is template instantiation in deal.II, Kokkos and the AD types rather than
parsing, so the usual levers do not help: a precompiled header doubles the cold
build to buy 7% on rebuilds, and `-O0` saves 14%. Both were measured and
rejected.

What does help is already here — the build cache below, and `-sd` overrides that
let a scan skip the compiler entirely.

## Build caching

Build trees are package-owned, under `~/.cache/DiFfRG.jl/builds`, and keyed on
the source contents together with the build options. An unchanged app is a
no-op; an edited one is rebuilt in a fresh tree, so switching options does not
repeatedly invalidate one directory and your own `build/` is never touched.

## Reading output

`read_h5` returns a `SimulationData` with the file's structure preserved:

| field | contents |
|---|---|
| `config` | the parameters the run used — output is self-describing |
| `scalars` | name => series |
| `maps` | name => `(times, coordinates, data)` |
| `coordinates` | standalone coordinate sets |
| `fe` | group => `(times, nodes, fields)`, one entry per frame |

`read_csv` reads the CSV readouts. Both accept either a path or a `RunResult`.

## One invocation is not always one flow

The executable is treated as opaque. Some drivers run a single flow; others
sweep internally and write a directory per probe. `RunResult` therefore reports
every file a run produced, and `h5_files`/`read_h5` return collections.

## Schemes

Which C++ types a discretization and timestepper are spelled with lives in
`frontend/schemes.toml` in the **DiFfRG install**, not in this package, so that
adding a discretization to the library updates this frontend in the same commit.

```julia
s = resolve_scheme(model = "Tut1", dim = 1, discretization = "CG")
s.assembler   # "CG::Assembler<CG::Discretization<Tut1, RectangularMesh<1>>>"
s.solution    # ["fe_functions", "fe_derivatives", "fe_hessians", ...]
s.headers     # in include order
```

`resolve_scheme` rejects combinations the library cannot express, rather than
letting them become template errors:

| constraint | why |
|---|---|
| `LDG` needs a serial triangulation | `RectangularMesh<dim>` is shared-parallel in an MPI build |
| `KT` needs `NoAdaptivity` | the FV assembler cannot be refined |
| `SUNDIALS_IDA_BoostRK` needs an explicit solver and `prec` | its alias has no defaults |
| `Variables` has no mesh or discretization | `dim == 0`, no FE space |

It also records what each assembler puts in the `Solution` bundle — DG has no
`fe_derivatives`, `Variables` has only `variables` — which is what lets a
generator reject a model body before the compiler sees it.

The table is checked by compiling it: `test/gen_probe.jl` emits a project
instantiating every legal (discretization, stepper) pair, and the integration
suite builds it. Corrupting an entry does make that fail.

## Generating a model

Declare the physics in Julia and get an ordinary DiFfRG project out — `model.hh`,
a driver, `CMakeLists.txt` — which you can read, build by hand, or take over.

```julia
model = Model("Burgers";
    dim = 1,
    fe  = [Scalar("u")],
    params = (Lambda = 1.0, a = 0.0, b = 1.0),

    initial_condition = cxx"""
        values[idxf("u")] = prm.a + prm.b * pos[0];
        """,
    flux = cxx"""
        const auto u = get<"fe_functions">(sol)[idxf("u")];
        flux[idxf("u")][0] = 0.5 * powr<2>(u);
        """)

app = generate(model, "build/burgers"; discretization = "CG", stepper = "SUNDIALS_IDA")
run(build!(app))
```

What that saves is the boilerplate whose every line is implied by those choices:
the `Parameters` struct (a key is named once here instead of three times, and
`Nc = 3` reaches C++ through `get_int`, not `get_double`), the descriptor
typedefs, the CRTP base list, the exact template signatures, the driver, and the
build file. Only members you supply are emitted — `AbstractModel` has a working
default for all but `initial_condition`.

Member bodies can be C++ via `cxx"..."`, or Julia expressions via `@julia`.

### Bodies as Julia expressions

```julia
flux = @julia((F, x, sol) -> begin
    u  = sol.fe_functions[:u]
    du = sol.fe_derivatives[:u][1]
    F[:u][1] = 0.5 * u^2 + x[1] * du
end)
```

The head mirrors the C++ signature rather than hiding it, and that is what earns
the checking: the generator knows which bundle entries a body reads and which the
chosen assembler provides, so **reading `fe_hessians` under a DG assembler is a
Julia error naming the line**, not a template error a hundred lines deep. So is
branching on a value, pinning a number type, or naming a component the model does
not have.

Indices are 1-based as everywhere else in Julia — `[1]` is the first spatial
direction and comes out as `[0]`. `u^2` becomes `powr<2>(u)`, and every numeric
literal is emitted as a double so `1/2` cannot become C++ integer division.

The subset is arithmetic over a whitelist and nothing else, because the same body
is instantiated with `double`, with forward-AD types, and on device. Anything
outside it should be `cxx"..."`.

*A macro in a keyword argument must be parenthesised, or it swallows the rest of
the call.*

### `dt_variables` as a graph

`dt_variables` is statement-shaped, so it has its own form:

```julia
dt_variables = @variables_flow(begin
    @guard has_run_away(m2A, prm.m2A) => FlowAbort(t)
    @update ZA, Zc
    args = @tie(k, ZA, Zc, dtZA, dtZc)

    @fixedpoint (dtZA, dtZc) size=p_grid_size tol=prm.eta_tol maxiter=prm.eta_iter_max begin
        r[:ZA] = flows.ZA.map(coordinates1D, args)
        r[:Zc] = flows.Zc.map(coordinates1D, args)
        @update dtZA from r[:ZA]
        @update dtZc from r[:Zc]
    end
end)
```

Writing it as a graph lets the generator own three invariants that the C++ keeps
by hand and by comment:

- **`DeferredMaps` batches are derived from the data flow.** Choosing them by
  hand is an optimization whose failure mode, per `map_completion.hh`, "is not a
  compile error but wrong numbers". A launch that reads an earlier destination
  closes the batch automatically.
- **Nesting is unrepresentable.** `DeferredMaps` is a bare RAII flag whose
  destructor flushes, so a nested scope flushes early and silently. Exactly one
  scope is emitted per derived batch, and a lone launch gets none.
- **The abort guard precedes every launch.** The assembler fences only after
  `dt_variables` returns, so throwing past an in-flight kernel is what its
  placement defends against; here that is structural.

`@fixedpoint` also generates the bookkeeping — the `old_*` copies, the maximum
relative change, the iteration cap.

Checked against `Examples/YangMills/Full`: the derived batches are the same two
the author chose by hand.

Generated apps ship no parameter file: the schema comes from the built binary via
`--generate-parameter-file`, so it cannot drift from what the binary reads. Give
the model's own defaults through `config`.

Attach a Mathematica-generated flow library with `flows = FlowLibrary("flows/")`;
the class and CMake target names are read out of the generated files, and the
subdirectory, member and `set_k` wiring follow.

**Two conventions are silent when wrong:** RG time runs as `t = ln(Λ/k)`, and
`dt_variables` fills the *residual* with `A_dot = -residual` — so a decaying
variable needs a positive residual.

### Is it right?

The generated apps are compared against the hand-written originals, not merely
compiled: regenerating `Tutorials/tut1` and `Tutorials/tut3` (the latter with its
flow library, h-adaptivity and `set_time` wiring) reproduces their fields
**bitwise** on identical settings.

## Examples

Runnable scripts in `examples/`:

| script | what it shows |
|---|---|
| `01_run_and_read.jl` | build, run, read HDF5. Uses `Tutorials/tut1` (Burgers' equation), whose exact solution `u = x/(1+t)` the script checks against. ~1 min. |
| `02_scan_temperature.jl` | a temperature scan of the O(N) model, printing the chiral condensate `sigma(T)` melting to zero. One build, nine processes. |
| `03_compare_discretizations.jl` | the same physics through CG, dDG and LDG from one source directory; they agree to ~1e-7. Several minutes on the first run. |
| `04_generate_a_model.jl` | declare a model in Julia, emit the C++, build and run it, then re-emit it through another discretization. |
| `05_generated_flows.jl` | a model whose flux is a Mathematica-generated loop integral; regenerates `Tutorials/tut3` and checks it bitwise against the original. Several minutes on the first run. |
| `06_julia_expressions.jl` | the same model with no C++ at all, plus what the DSL refuses and why. |

```bash
julia --project=. examples/02_scan_temperature.jl
```

## Tests

```bash
julia --project=. test/runtests.jl                      # unit tests
DIFFRG_JL_INTEGRATION=1 julia --project=. test/runtests.jl   # also builds and runs tut1
```
