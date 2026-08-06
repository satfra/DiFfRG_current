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

JSON is only an adapter for the typed `Config::OutputSettings`. The corresponding JSON settings are:

```json
{
  "output": {
    "max_pending_frames": 2,
    "log_queue_size": 8192
  }
}
```

Debugging and process-memory policy stay in C++ rather than simulation configuration. Set
`Config::OutputSettings::asynchronous` to `false` for synchronous field writes. The default pending-byte limit is 2
GiB and can be changed through `Config::OutputSettings::max_pending_bytes` when constructing the session.

Only MPI rank zero creates sinks. A full queue blocks the producer rather than dropping scientific data. At the end of
each timestepper `run()`, the session drains pending writers and reports worker errors without closing, so the same
session can continue across multiple time intervals. Normal applications do not call `finish()` explicitly: the
`OutputSession` destructor performs terminal cleanup. Advanced callers may use `finish()` to close a session before it
leaves scope; subsequent frame writes are then rejected.
