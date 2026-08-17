# Running flow equations across several GPUs

This is the hand-over for the follow-up session that runs DiFfRG on a cluster with 1–4 GPUs and
produces a scaling curve. It describes what was built, how to build and launch it, what to measure,
and — importantly — what a *disappointing* curve would and would not mean.

---

## 1. What was built

`mpirun -n N ./YourApp` now spreads the flow-equation work over `N` ranks (and, with one GPU per
rank, over `N` GPUs). **No model, no generated flow file and no application source had to change.**
There is deliberately no user-facing API for this — no balancer to construct, no distribution to
register, nothing to tune per flow.

| Component | File | Role |
|---|---|---|
| `MapScheduler` | `include/DiFfRG/physics/integration/map_scheduler.hh`, `src/.../map_scheduler.cc` | Decides which rank computes which part of each `map()`, and performs the batched exchange |
| `MapCompletion` | `include/DiFfRG/physics/integration/map_completion.hh` | Lands deferred device results, then drives the exchange |
| Rank → GPU affinity | `src/DiFfRG/common/init.cc` | Gives each rank its own device and its share of the CPU threads |
| MPI utilities | `include/DiFfRG/common/mpi.hh`, `src/.../mpi.cc` | `allgatherv`, `agree`, `bcast`, `any_of`, `split_shared`, `abort` |

The previous sketch (`NodeDistribution`, `IntegrationLoadBalancer`,
`AbstractIntegrator::set_node_distribution`) has been deleted.

### 1.1 What gets split, and why that choice

**Only the external coordinate grid of a `map()` — never a quadrature reduction.**

Every individual integral is therefore still summed entirely on one rank, in the original order, so
the distributed result is **bitwise identical to a serial run at any rank count**. This is not
fastidiousness. The flow kernels feed a stiff DAE; `common/tbb.hh:75-85` spells out the consequence:
a last-bit change in one residual changes the accepted step sequence, and from there the entire
trajectory. A scheme that summed partial quadratures across ranks would give a *different physics
answer* for every rank count, and no scaling curve drawn from it would be comparable.

It also makes rebalancing free: moving a partition boundary changes *which* rank computes a grid
point, never *what* it computes.

Splitting applies only to one-dimensional coordinate systems. `SubCoordinates` derives its per-axis
window from `from_linear_index()`, which is exact for `dim == 1` and not for higher dimensions, so a
multi-dimensional external grid is assigned whole to one rank. Every `map()` in `QCD_Nf2` is over a
1D momentum grid, so this costs nothing there.

### 1.2 How the split width is chosen

Per `map()` call, with `G` external grid points and quadrature volume `Q = prod(grid_size)`:

```
S = G * Q                                    kernel evaluations in this map
r = clamp(S / quantum, 1, min(n_ranks, G))   how many ranks to spread it over
owners = the r least-loaded ranks so far in this batch
```

`quantum` defaults to `5e4` and is a **GPU fill threshold**, not a launch-overhead threshold. Below
roughly 5·10⁴ evaluations these register-heavy NumTracer kernels do not occupy a modern device
anyway, so splitting further buys nothing and costs a launch plus a share of a gather. (Deriving it
from launch cost — order 10³ evaluations — would oversplit by two orders of magnitude.)

The consequence that matters for load balance: **a cheap flow is assigned whole** to the
least-loaded rank rather than chopped into slivers, so a batch of many small flows spreads *across*
ranks instead of every rank doing a sliver of every flow.

For `QCD_Nf2/no_mesons_AAqbq18` at `x_order=32, cos1=cos2=phi=6`:

| class | count | Q | S = 64·Q | r at quantum = 5e4 |
|---|---|---|---|---|
| `4D_3ang` (18 `ZAAqbq` + `ZA4`) | 19 | 6912 | 442 368 | 8 |
| `4D_2ang` (`ZA3`, `ZAcbc`, `ZAqbq1/4/7`) | 5 | 1152 | 73 728 | 1 |
| `p2_1ang` (`ZA`, `Zc`, `Zq`) | 3 | 192 | 12 288 | 1 |

### 1.3 How results come back

`MapCompletion::flush()` fences, lands the pinned staging buffers, and then performs **one**
`MPI_Allgatherv` covering every map in the open batch (~12.8 kB for the 25-map block). One collective
per `DeferredMaps` scope, not one per flow.

The rule that keeps this from deadlocking, and the single most important invariant in the design:

> The collective is driven by the **shared plan**, never by rank-local state. Every rank registers a
> plan entry for every `map()`, *including* the ranks that own nothing (`count = 0`). `flush()`
> decides "no-op or collective" purely from whether a plan is open.

Rank-local predicates (`MapCompletion::has_pending()`, "is my pending list empty") differ between
owners and non-owners, so any decision based on them would have some ranks enter a collective the
others skip — a silent hang. Before the exchange, the ranks agree on a hash of the plan
(`MPI::agree`, one tiny collective); a mismatch calls `MPI_Abort` with a diagnosis rather than
hanging in a malformed `Allgatherv`.

`MPI_Allgatherv` rather than `MPI_Allreduce(SUM)` over zero-filled buffers: the latter moves ~25×
more bytes and is **not bit-safe**, since `(-0.0) + 0.0 == +0.0` destroys a signed zero.

There is deliberately **no** `MPI_Iallgatherv`. Every gather result is consumed on the very next host
statement (`dtZA.update()`, `zqF[0]`, the AAqbq basis rotation), and an explicit ABM stepper cannot
start the next RHS, so there is nothing to overlap with.

### 1.4 Rank → GPU affinity

`Init` now, after `MPI_Init`:

1. builds a node-local communicator (`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`),
2. divides `/discretization/threads` by the number of node-local ranks, so that setting still means
   *threads per rank* (both deal.II's limit and DiFfRG's own `tbb::global_control`),
3. picks `device = local_rank % n_visible_devices`,
4. initialises Kokkos itself with that device — replicating everything deal.II's
   `ensure_kokkos_initialized()` would have done, **including** setting
   `dealii::internal::dealii_initialized_kokkos = true` and registering `std::atexit(Kokkos::finalize)`.
   Both of those live inside its `if (!Kokkos::is_initialized())` branch; skipping them reintroduces
   the exit segfault documented at `init.cc:59-92`. This path is taken only when there is more than
   one node-local rank; a single-rank run still goes through deal.II unchanged.

**Sharing is detected by PCI bus id, not by counting.** Under Slurm `--gpus-per-task=1` every rank
sees exactly one device and `local_rank % n_devices == 0` is the *correct* answer; a count-based
check would report sharing on every well-configured cluster and send you chasing a phantom. The
node-local ranks gather their `cudaDeviceGetPCIBusId` strings and look for duplicates.

On a genuine duplicate you get a **loud warning on stderr from every affected rank, and the run
continues** — a shared GPU is slower, not wrong, and a scaling study has to be able to measure that
case:

```
================================ WARNING ================================
 GPU 0000:65:00.0 is shared by 2 node-local ranks (0, 1).
 The run is still correct, but those ranks time-slice the device and each
 costs several hundred MB of extra context, so it will NOT scale.
 Launch with one GPU per rank (e.g. Slurm --gpus-per-task=1), or set
 CUDA_VISIBLE_DEVICES per rank.
=========================================================================
```

Timestepper progress is stdout-only, so this deliberately goes to stderr — check both streams, or
merge them, if your launcher separates them.

---

## 2. Building with MPI on the cluster

MPI is **a single switch for the whole superbuild**. deal.II declares the MPI types (real ones with
MPI, dummies in `namespace dealii` without), SUNDIALS and PETSc have their own switches, and a mixed
configuration miscompiles rather than failing cleanly. There is now a configure-time guard for the
deal.II/DiFfRG mismatch and a `#error` in `common/mpi.hh` as a backstop, but the fix is always the
same: turn MPI on for the superbuild.

```bash
MPI=ON bash install.sh              # installer route
# or, configuring the superbuild directly:
cmake <src> -DMPI=ON -DCMAKE_INSTALL_PREFIX=<prefix>
```

There is also a `release-mpi` CMake preset in `DiFfRG/CMakePresets.json`.

**Build-parallelism rules for this workflow** (learned the hard way on a 30 GB machine):

- Bound the parallelism explicitly: `-j6`. Never bare `-j` or `-j$(nproc)`.
- To rebuild only the library and tests, build the **inner** target
  `build/DiFfRG/src/DiFfRG-build`, not the superbuild target — the latter rebuilds deal.II.
- **A header-layout change requires a clean rebuild of the application.** This increment changed the
  layout of `AbstractIntegrator` and deleted `distribution.hh`, so any app object file from before
  it must be rebuilt. A stale one gives a silent `SIGSEGV` that looks exactly like a logic bug.
  Also delete any stale `nsys` `.sqlite` next to the binary.

---

## 3. Launching

```bash
mpirun -n 4 ./QCD parameter.toml
# Slurm, one GPU per rank:
srun --ntasks=4 --gpus-per-task=1 ./QCD parameter.toml
```

Confirm each rank actually got a distinct device — by bus id, not by count:

```bash
mpirun -n 4 bash -c 'echo "rank $OMPI_COMM_WORLD_RANK sees: $CUDA_VISIBLE_DEVICES"; nvidia-smi -L'
```

and check that no sharing warning appeared on stderr.

Two knobs, both optional:

| knob | where | meaning |
|---|---|---|
| `DIFFRG_MAP_QUANTUM` | environment | overrides the split threshold (wins over the parameter file) |
| `DIFFRG_MAP_VERBOSE=1` | environment | prints the plan once: per-flow split width, owning ranks, per-rank load |
| `/integration/map_quantum` | parameter file | same as above, lower precedence |
| `/integration/map_verbose` | parameter file | same as above |

Run once with `DIFFRG_MAP_VERBOSE=1` before any timing and keep the output: the plan is what any
imbalance in the curve has to be explained against.

---

## 4. The gate: bitwise equality, before any timing

**A scaling number from a run that fails this is meaningless.** Run the same short configuration at
`-n 1`, `-n 2`, `-n 3` (deliberately not a divisor of 64) and `-n 4`, and compare the outputs
*exactly*, on raw `double` bits — not to a tolerance:

```bash
for n in 1 2 3 4; do
  mpirun -n $n ./QCD short.toml            # writes output_n$n.h5
done
# then compare the HDF5 datasets bit-for-bit, e.g. with h5diff -d 0, or numpy .tobytes()
```

If `-n 1` and `-n 4` differ in the last bits, stop and investigate before measuring anything: the
most likely cause is that the Kokkos `AUTO` team size in the phase-2 reduction
(`quadrature_integrator.hh:305`) resolved differently at a different `league_size`. That is a
correctness question, not a performance one.

The library-level part of this contract is already covered by tests:

- `tests/physics/integration/map_scheduler.cc` — simulates 1…16 ranks in a single process and checks
  that the slices tile the grid exactly, that the split width follows the cost score, that cheap
  flows spread rather than fragment, and that a batch exchange reassembles the full result bitwise
  on every rank. Runs on any build, MPI or not.
- `tests/physics/integration/distributed_quadrature_integrator.cc` — compares `map()` against
  `map_dist()` bitwise, and is registered under `mpirun` at 2, 3 and 4 ranks when the build has MPI
  (`setup_mpi_test`, `MPI_TEST_RANKS`).

---

## 5. The measurement protocol

1. **Stop every background run first.** Any build, sweep or leftover job contaminates the numbers —
   this has produced misleading benchmarks before, and it looks like a machine problem rather than a
   self-inflicted one.
2. Use a short, fixed configuration: `/tuning/m2A/tune = false`, `/tuning/STI/tune = false`,
   `/timestepping/final_time` small enough for a few dozen ABM steps at `dt = 0.05`. Keep it
   identical across all rank counts.
3. Time **the timestepping loop**, not the process: process wall time includes MPI startup, Kokkos
   init and HDF5 setup, which do not scale and are not what is being measured. The assembler's own
   `average_time_residual_assembly()` × `num_residuals()` is the cleanest quantity; failing that,
   time from the first to the last output frame.
4. Repeat each rank count at least twice and report the *minimum*, not the mean — GPU clock ramp
   makes first runs slow and there is no ramp-down noise to average away.
5. Record, per rank count: wall time of the timed region, number of RHS evaluations (they must be
   identical across rank counts — if they are not, the bitwise gate failed), and the plan dump.

---

## 6. What to expect, and what would be surprising

### The ceiling is GPU-side, not communication-side

Two things do **not** scale, and they set the ceiling:

- **The η loop** (`model.hh:279-310`). Up to 50 iterations of 3 one-angle maps, each only
  64 × 192 = 12 288 kernel evaluations. Those do not fill a single GPU today, so the η loop **cannot
  be accelerated by more GPUs under any split**. It is a hard Amdahl floor of roughly
  `n_iter × (3 launches + 2 global fences + 2 spline uploads)`.
- **The phase-2 team reduction** (`quadrature_integrator.hh:304-306`).
  `TeamPolicy(space, integral_view.size(), AUTO, 32)` has `league_size == G_local`, so it shrinks
  64 → 16 → 8 → 4 as ranks grow. At 64 teams it is *already* below an A100's 108 SMs and
  latency-bound, so its wall time is roughly **constant in N** while phase 1 shrinks linearly. Its
  share `f₂` is documented as 0.3–4 %, which caps per-flow speedup at `1/(f₂ + (1−f₂)/N)`: at
  `f₂ = 4 %` that is 3.6× at N=4 but only 10× at N=16.

**Both of these were measured before any MPI code ran, and both turned out to be much smaller than
feared: the η loop is 0.24 % of wall time and the phase-2 reduction 0.16 % of kernel time.** See
**section 7** for the numbers and the resulting speedup table. Read them before interpreting the
curve; without them the curve cannot be interpreted at all.

### Things that are *not* the problem

- **Collective latency.** ~1 collective per main block plus 2 per η iteration, i.e. ~21–61 per RHS at
  realistic `n_iter`. At 5–40 µs each plus straggler skew that is ~5–15 ms against a multi-second
  RHS. If someone proposes non-blocking collectives to fix the curve, the arithmetic says no.
- **Redundant host work.** Every rank runs the whole host side, dominated by 29 `Interp::update()`
  at ~25 µs each, i.e. ~1.8 ms/RHS. Under 1 % even at N=16.
- **The position cache.** `SubCoordinates` *does* have a `to_string()`, so the MPI path keeps the
  cached `forward()` positions. (A comment in `quadrature_integrator.hh` used to claim otherwise;
  it has been corrected.) A schedule that moved a slice boundary would re-key and re-run a ~5 µs
  fill — the schedule is stable across batches, so this does not happen.

### If the curve is worse than section 7 predicts, suspect, in order

1. **GPU sharing.** Check stderr for the warning and confirm distinct bus ids.
2. **Grid quantization.** 64 points over N: exactly 0 % waste at N = 2, 4, 8, 16; ≤ 12.5 % elsewhere,
   with N = 7, 9 and 12 the awkward ones.
3. **Unequal flows within the `4D_3ang` class**, but *only* at N > 8 — see the end of section 7.
   This is the most likely explanation for a disappointing 16-GPU point and cannot explain a
   disappointing 2-, 3- or 4-GPU one.
4. **`quantum` mis-sized for this GPU.** Sweep `DIFFRG_MAP_QUANTUM` over, say, 1e4 … 2e5 and watch
   the plan dump — too small fragments the big flows past the fill threshold, too large leaves ranks
   idle.
5. **`n_iter` in the η loop.** It is data-dependent (≈ 14.7 per RHS in the Step-0 run); if it grew
   substantially, the serial fraction grew with it — but from a base of 0.24 % it would have to grow
   enormously to matter.

### The lever for scaling past ~8 GPUs

`p_grid_size` (`model.hh:80`) is `static constexpr uint p_grid_size = 64`. It is the external grid
being split, so it directly bounds the usable rank count — and raising it is physically desirable
anyway. It is compile-time, so raising it means rebuilding the app (see the clean-rebuild note in
section 2).

---

## 7. Step-0 measurements

Measured on the development workstation before any MPI code ran: one **RTX 4070 Laptop GPU (36 SMs,
sm_89)**, `nsys profile -t cuda` over a serial run of `no_mesons_AAqbq18` with tuning off,
`final_time = 0.5`, `dt = 0.05`. 34 RHS evaluations, 4735 kernel launches, 142.0 s of GPU-activity
span.

### How the per-kernel numbers were obtained

Straight from the nsys CUDA activity trace: each row of `CUPTI_ACTIVITY_KIND_KERNEL` carries a
device-side `start`/`end`, and every flow is a distinct template instantiation, so the demangled
kernel name identifies the flow exactly (`DiFfRG::<Flow>_kernel<...>`). Group by name, sum
`end - start`. Phase 1 and phase 2 are separated by whether the name contains
`KokkosNDLambdaWrapper`.

Summing durations is only legitimate if kernels do not overlap, so that was checked rather than
assumed:

| | |
|---|---|
| distinct CUDA streams | 9 |
| GPU-busy time with exactly **one** kernel resident | **97.5 %** |
| sum of kernel durations vs. union of intervals | 144.51 s vs 140.90 s, i.e. **2.56 %** inflation |

So despite nine streams the flows effectively execute one at a time — each phase-1 launch is 3456
blocks × 128 threads against 36 SMs, i.e. ~96 blocks per SM, which leaves nothing for a neighbour to
overlap with. The per-flow numbers are therefore additive to within ~2.5 % and are already
effectively isolated measurements. Per-launch spread within one flow is ~1.3× min-to-max over 34
launches (clock state), so quote the median, not a single launch.

`S = G · Q` is exactly the launched thread count: 64 × 6912 = 442 368 = 3456 × 128. That is the
sense in which the scheduler's score is exact — it is exact about *work issued*, and says nothing
about work *per thread*, which is where the 6–8× spread below comes from.

| quantity | value | share |
|---|---|---|
| GPU busy (union of kernel intervals) | 140.90 s | **99.25 %** of the span |
| main flow block, wall | 140.96 s | 99.29 % |
| **η loop, wall** | **0.341 s** | **0.24 %** |
| host-only gaps between GPU segments | 0.661 s | 0.47 % |
| **phase-2 team reduction, kernel time** | **0.226 s** | **0.16 %** of 144.5 s kernel time |
| η iterations per RHS | 501 / 34 | ≈ 14.7 |

**Both feared ceilings are far smaller than projected, and neither is worth acting on.**

- The η loop was expected to be a hard Amdahl floor, possibly over 10 % — it is **0.24 %**. There is
  no reason to loosen `eta_tol`, raise `eta_relax` or Anderson-accelerate the fixed point for
  scaling reasons.
- The phase-2 reduction was cited at 0.3–4 %; measured **0.16 %**, at or below the bottom of that
  range. Restructuring the league (`G_local × n_chunks`) is not warranted — and would have changed
  the reduction tree and therefore the bitwise baseline.

Non-scaling total: `0.341 + 0.661 + 0.226 ≈ 1.23 s` against 141.96 s, i.e. **f ≈ 0.87 %**. With
`T(N) = 140.7/N + 1.23`:

| N | predicted speedup | predicted efficiency |
|---|---|---|
| 2 | 1.98× | 99 % |
| 4 | 3.90× | 97 % |
| 8 | 7.54× | 94 % |
| 16 | 14.2× | 89 % |

Collective overhead is not in this model because it does not matter: ~21–61 collectives per RHS at
5–40 µs is 7–83 ms over the *entire* 142 s run.

**These fractions are hardware-specific, and they get worse on a faster GPU.** The η loop costs
0.68 ms per iteration, of which only ~0.41 ms is kernel time; the remaining ~0.27 ms is launch and
fence latency, which does not shrink when the device gets faster. The 0.661 s of host-only gaps is
likewise invariant. Scaling the *kernel* time down by 6× (roughly a 4070 Laptop → A100 step) while
holding the invariant terms fixed gives `f ≈ 3.6 %` rather than 0.87 %, and an N = 16 ceiling near
**10×** rather than 14×. The ranking of the terms does not change — collectives still do not matter
— but on a fast cluster GPU the dominant non-scaling term stops being the η loop and becomes the
per-RHS host work behind those 0.661 s. If the cluster curve flattens earlier than the table above,
re-measure the three invariant terms on the cluster's own GPU before looking for a bug.

**The one real imbalance risk, and it only appears at N > 8.** The 19 `4D_3ang` flows all score
`S = 442 368`, so the scheduler treats them as equally expensive — but they are not. Measured
per-flow kernel time:

- `ZAAqbq1`–`ZAAqbq15`: 8.1–9.4 s each (5.6–6.5 % of GPU time each)
- `ZAAqbq16`, `ZAAqbq17`: ~2.0 s each
- `ZAAqbq18`: 1.6 s
- `ZA4`: 1.2 s

i.e. a ~6–8× spread within one cost class. At N ≤ 8 this is harmless: `r = min(8, N)` means *every*
rank owns an equal slice of *every* big flow, so the per-rank load is identical whatever the flows
actually cost. At N = 16, `r` is still 8, so 19 × 8 = 152 slices are packed onto 16 ranks by a score
that cannot tell a 9.4 s flow from a 1.6 s one — and imbalance of tens of percent becomes possible.
**If a 16-GPU run underperforms the table above, this is the first thing to check** (dump the plan
with `DIFFRG_MAP_VERBOSE=1` and weigh the flows by the kernel times above). The fix is the
calibration listed as open in section 8, or simply raising `p_grid_size` so that `r` can exceed 8.

---

## 8. Known-open items

These were consciously left out of this increment:

- **FE work is not distributed.** `QuadratureIntegrator::get()` (the scalar, blocking path the FE
  assemblers use) is untouched, and the mesh is not partitioned. The SPMD choice was made partly so
  that adding this later is an orthogonal addition rather than a rewrite — deal.II is SPMD.
- **Cost calibration is not implemented.** `w_kernel` is fixed at 1, i.e. the score is exactly
  `G · Q`. The Step-0 profile shows this is *exactly* right for N ≤ 8 and *wrong by up to 6–8×
  per flow* at N > 8 — see the end of section 7 for why the first statement holds despite the
  second. If a 16-GPU run needs it, the cheapest version is: run one calibration batch with
  `r = 1, owners = all ranks` so every rank times the *whole* flow rather than its own slice, then
  one `Allreduce` yields both `w_kernel` and the per-rank speed ratios (the latter is what a
  heterogeneous cluster would additionally need). Keep calibration RHS evaluations out of any timed
  region.

  An earlier draft of this document warned that isolated per-flow timings overstate a block that in
  production overlaps on several streams. **That is measurably false for this workload** — see the
  concurrency table in section 7: 97.5 % of GPU-busy time has exactly one kernel resident, because a
  single phase-1 launch already saturates the device. Isolated timings and in-situ timings agree to
  ~2.5 % here, so the measured per-flow times may be used directly as weights. Re-check the
  concurrency table if the truncation, the quadrature orders, or the GPU change enough that a launch
  no longer fills the device.
- **Multi-dimensional external grids are not split** (section 1.1). Fixing this means fixing
  `SubCoordinates`' per-axis window derivation, which is currently only correct for `dim == 1`.
- **The application must not branch on rank-local data.** `QCD.cc` reads control-flow scalars back
  out of the HDF5 file that only rank 0 writes, at four sites: `:46` and `:84`
  (`time_of_divergence`), `:52` (`m2A`, `Zc`), `:57` (`sti_deviation_or_nan`). Those values drive the
  `tune_m2A` / `tune_STI` bisection, and each probe is a whole run — so if ranks disagree by one
  float about convergence they execute a *different number of runs*, and the first collective of the
  extra run hangs. **A barrier does not fix this.** The fix is rank-0-reads-and-broadcasts for every
  scalar that influences control flow, including `run()`'s boolean return
  (`MPI::bcast` / `MPI::all_of` are provided for exactly this). Until that is done, **run the scaling
  study with tuning switched off**, which is what section 5 prescribes anyway.
- **An exception thrown inside a `DeferredMaps` scope is not recoverable under MPI.** It would unwind
  one rank out of a batch the others are still filling. The scheduler detects this (`poison()`) and
  turns the next flush into an `MPI_Abort` with a diagnosis rather than a hang. `QCD`'s own
  `FlowAbort` is thrown *before* any map and outside any scope, deliberately, so it is unaffected.
- **Consider pinning the phase-2 team size** instead of `Kokkos::AUTO`. `AUTO` resolves from
  registers and shared memory rather than from `league_size`, so it is very likely invariant under
  the split — but the whole design is sold on bitwise identity, and the bitwise gate in section 4 is
  what actually verifies it.
