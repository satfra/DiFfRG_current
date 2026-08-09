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
    "log_flush_interval": 10.0
  }
}
```

The run log is written by an asynchronous logger whose file sink is otherwise only flushed on shutdown, so a long run
would show a log file lagging by a full stdio buffer and a killed job would lose its tail. `log_flush_interval` is the
period in seconds at which the log file is flushed from a dedicated thread; set it to `0` to disable periodic flushing.

Debugging and process-memory policy stay in C++ rather than simulation configuration. Set
`Config::OutputSettings::asynchronous` to `false` for synchronous field writes. The default pending-byte limit is 2
GiB and can be changed through `Config::OutputSettings::max_pending_bytes` when constructing the session.

Only MPI rank zero creates sinks. A full queue blocks the producer rather than dropping scientific data. At the end of
each timestepper `run()`, the session drains pending writers and reports worker errors without closing, so the same
session can continue across multiple time intervals. Normal applications do not call `finish()` explicitly: the
`OutputSession` destructor performs terminal cleanup. Advanced callers may use `finish()` to close a session before it
leaves scope; subsequent frame writes are then rejected.
