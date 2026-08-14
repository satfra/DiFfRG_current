# Runtime output

Runtime output uses one move-only `OutputPath` and one `OutputSession` at the application boundary. The path owner must
outlive the session, which owns the scientific sinks, bounded asynchronous VTK queue, effective-configuration snapshot,
and run logger. Both are explicitly constructed; there is no global output or logger registry and no hidden path
fallback in a timestepper.

Applications may adapt their JSON configuration at the boundary:

```cpp
OutputPath path(json);
OutputSession<dim, VectorType> output(path, Config::OutputSettings(json));
```

Unit tests request an automatically cleaned system-temporary directory without constructing output JSON:

```cpp
OutputPath path = OutputPath::temporary();
OutputSession<dim, VectorType> output(path, Config::OutputSettings{});
```

Use `OutputPath::temporary(TemporaryRetention::keep)` while debugging a test. `OutputPath` has no default constructor,
is not copyable, and transfers its cleanup responsibility when moved. Persistent typed and JSON-created paths are never
deleted automatically.

Each scheduled output event is collected in one scoped `OutputFrame`:

```cpp
output.write_frame(time, [&](auto &frame) {
  assembler.attach_data_output(frame, solution, variables);
});
```

All assembler families retain the familiar `attach_data_output` seam. Model `readouts` receive the frame through it.
`readouts_multiple` assigns every readout a stable ID so duplicate contributions fail immediately. A frame error makes
the session fail-stop, preventing staged data from contaminating a later timestep. Existing CSV, HDF5, PVD, and VTU
names and schemas remain unchanged.

Code which runs independently of scheduled frames receives a `DiagnosticPort` by value. For example, a timestepper can
export Jacobian eigenvalues without knowing a filename or owning a writer:

```cpp
class MyStepper {
public:
  explicit MyStepper(DiagnosticPort diagnostics) : diagnostics(std::move(diagnostics)) {}

  void inspect(double time, double lambda_min, double lambda_max) {
    diagnostics.record("jacobian_eigenvalues.csv", time,
                       {{"lambda_min", lambda_min}, {"lambda_max", lambda_max}});
  }

private:
  DiagnosticPort diagnostics;
};
```

The same injection works for a model pre-assembly hook: store a `DiagnosticPort` in the model and record the diffusion
coefficient when the hook computes it. The port is thread-safe and serializes a complete record atomically. It should
not be used for high-volume field output; add that data to an `OutputFrame` instead.

All implicit time steppers write one record per Jacobian callback to `<run>_jacobian_diagnostics.csv`. Runs with a
separate dense variable Jacobian also write `<run>_variable_jacobian_diagnostics.csv`; both records use the same
`jacobian_build_id`. The numeric `stepper_kind` values are `0=IDA`, `1=IDA+Boost RK`, `2=IDA+Boost ABM`, `3=implicit
Euler`, and `4=TRBDF2`. The `stage` values are `0=main`, `1=TR`, and `2=BDF2`.

For implicit Euler and TRBDF2, `step` counts accepted steps and `retry_index` counts rejected attempts at the same step.
For IDA, `step` and `ida_step` both contain the native IDA step counter and `retry_index` is `NaN`; `ida_step` is `NaN`
for other steppers. `current_h` is the attempted step size and `last_h` is the previous accepted step size. `alpha` and
`beta` are the mass-Jacobian weights from the
assembler interface, while `residual_weight` is the residual-Jacobian weight. For implicit Euler these are
`alpha=beta=1` and `residual_weight=h`. For TRBDF2, the TR stage uses `alpha=beta=1` and
`residual_weight=gamma*h/2`; the BDF2 stage uses `alpha=beta=2-gamma` and
`residual_weight=(1-gamma)*h`, where `gamma=2-sqrt(2)`.

Each record also contains matrix dimensions, actual nonzero count, entry and matrix norms, diagonal and row/column
scale measures, and factorization diagnostics. `min_diagonal_dominance` is the minimum of
`abs(diagonal) / sum(abs(off-diagonal))` over all rows. A diagonal-only nonzero row has infinite dominance, while an
empty or zero row has zero dominance.

`factorization_ms`, `factorization_success`, and `scaled_rcond_estimate` are populated for UMFPACK-based solvers. The
condition estimate applies the same maximum-entry row/column equilibration used by `ScaledLinearSolver` and performs
at most five estimator iterations after numeric factorization. Dense variable Jacobians report inversion time and
success but leave the condition estimate as `NaN`. Iterative GMRES paths leave all three factorization fields as `NaN`;
no eigenvalue computation is performed.

JSON is only an adapter for the typed `Config::OutputSettings`. The corresponding JSON settings are:

```json
{
  "output": {
    "max_pending_frames": 2,
    "log_queue_size": 8192,
    "log_flush_interval": 10.0,
    "quadrature_log": true
  }
}
```

The run log is written by an asynchronous logger whose file sink is otherwise only flushed on shutdown, so a long run
would show a log file lagging by a full stdio buffer and a killed job would lose its tail. `log_flush_interval` is the
period in seconds at which the log file is flushed from a dedicated thread; set it to `0` to disable periodic flushing.

A `QuadratureProvider` constructed from the configuration reports every quadrature it builds to its own side-channel
log, `<output name>_quadrature.log`, written next to the run log and sharing its queue, level and flush settings. It
owns that logger instead of borrowing the session's: integrators request their quadratures from within their
constructors, usually before any `OutputSession` exists, and the provider is a process-level cache that outlives a
single run. The file has no console sink, so the quadrature inventory does not interleave with the timestepper
progress report. Set `quadrature_log` to `false` to suppress the file, or pass an explicit `LogPort` as the second
constructor argument to route the messages into a log of your own.

Debugging and process-memory policy stay in C++ rather than simulation configuration. Set
`Config::OutputSettings::asynchronous` to `false` for synchronous field writes. The default pending-byte limit is 2
GiB and can be changed through `Config::OutputSettings::max_pending_bytes` when constructing the session.

## HDF5 writing

`scalar()`, `map()` and the finite-element field data do not touch the file when they are called. They validate their
arguments, copy their payload, and stage the write. `flush()` then commits the whole frame in a single
open → write → `H5Fflush` → close cycle. Two consequences are worth knowing:

- **A frame is on disk only after its flush.** Under an `OutputSession` this is automatic, once per `write_frame`.
  Code that drives an `HDF5Output` directly and then reopens the file to read it back must flush first.
- **Every frame is still closed on disk.** The file stays readable while the run continues, and an abort leaves a
  complete prefix of frames behind rather than a corrupt file.

One writer thread per session performs those cycles, so the integrator does not wait for the disk. It is one thread
and not one per sink because the HDF5 library is built here without thread safety, which means every HDF5 call in the
process has to be serialised. For the same reason, code that opens an HDF5 file itself in the middle of a run — an
`HDF5Input`, say — must call `output.drain()` first.

`Config::OutputSettings::hdf5_queue_depth` (default 1) bounds how many frames may be in flight. It is separate from
`max_pending_frames`, which bounds VTK/`DataOut` memory. The default of 1 is a durability choice: a hard abort
(`SIGKILL`, node eviction) can lose the queued frames on top of the one being written, so depth 1 caps that at two
frames while still hiding the whole write behind the next output interval. Setting
`Config::OutputSettings::asynchronous` to `false` removes the thread entirely and restores commit-on-return, at the
cost of putting the write back on the critical path.

Each run reports where its output time went, at `info` level, when the session finishes: the contributor callback and
the potential reconstructions inside it, `build_patches`, the data filter, the HDF5 write and open/close, queue
waiting, and the writer thread's own total, which is listed separately because it is off the critical path. Set
`/output/verbosity` to 3 or more to additionally get one line and one `output_timings.csv` row per frame.

Only MPI rank zero creates sinks. A full queue blocks the producer rather than dropping scientific data. At the end of
each timestepper `run()`, the session drains pending writers and reports worker errors without closing, so the same
session can continue across multiple time intervals. Normal applications do not call `finish()` explicitly: the
`OutputSession` destructor performs terminal cleanup. Advanced callers may use `finish()` to close a session before it
leaves scope; subsequent frame writes are then rejected.
