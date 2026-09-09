"""
Running applications and sweeping parameters.

A run is a process, a parameter file and an output directory. What that process
produces is discovered from the directory afterwards rather than assumed, so a
driver that runs one flow and a driver that runs a tuning loop over many probes
are both handled.
"""

"""
    RunResult

The outcome of one invocation.

# Fields
- `app`: the application that was run.
- `outdir`: the directory it wrote into.
- `params`: the overrides it was given.
- `success`: whether the process exited zero.
- `stdout_log`: captured standard output.
- `files`: every output file found under `outdir`, relative to it.
"""
struct RunResult
    app::App
    outdir::String
    params::Dict{String,Any}
    success::Bool
    stdout_log::String
    files::Vector{String}
end

function Base.show(io::IO, r::RunResult)
    print(io, "RunResult(", r.success ? "ok" : "FAILED", ", ", r.outdir,
          ", ", length(r.files), " files)")
end

"""
    h5_files(result) -> Vector{String}

Absolute paths of the HDF5 files a run produced, in sorted order.

A single-flow driver yields one; a driver that sweeps internally yields one per
probe, which is why this returns a collection rather than a path.
"""
h5_files(r::RunResult) =
    [joinpath(r.outdir, f) for f in r.files if endswith(lowercase(f), ".h5")]

"""
    csv_files(result) -> Vector{String}

Absolute paths of the CSV files a run produced.
"""
csv_files(r::RunResult) =
    [joinpath(r.outdir, f) for f in r.files if endswith(lowercase(f), ".csv")]

"""
    run_app(app; params=(), config=(), outdir=nothing, name="output",
            args=String[], verbose=true, check=true) -> RunResult

Build the application if needed, then run it once.

`params` are applied as command-line overrides, so no rebuild happens between
points; `config` is merged into the parameter file the run is handed. A bare key
means `/physical/<key>`, and a key starting with `/` is a JSON pointer into any
part of the tree.

`outdir` defaults to a fresh temporary directory, which is *not* cleaned up --
the results are the point of the run. Pass one to keep results somewhere known.

Standard output is streamed as it arrives when `verbose`, because timestepper
progress is written to stdout and never reaches the log file; waiting for the
process to exit would show nothing until it was over.
"""
function run_app(app::App;
                 params = (),
                 config = (),
                 outdir = nothing,
                 name::AbstractString = "output",
                 args::AbstractVector{<:AbstractString} = String[],
                 verbose::Bool = true,
                 check::Bool = true)
    build!(app; verbose = verbose)
    dir = outdir === nothing ? mktempdir(; prefix = "diffrg_run_", cleanup = false) :
          abspath(expanduser(String(outdir)))
    mkpath(dir)

    # The run gets its own copy of the parameter file. That keeps the run
    # directory self-describing and independent of the build tree, and it is
    # what lets the process run with its cwd in the output directory.
    parfile = prepare_parameter_file(app, dir, config)

    flags = ["-p", parfile]
    append!(flags, override_flags(params))
    append!(flags, ["-ss", "/output/folder=$(dir)/", "-ss", "/output/name=$(name)"])
    append!(flags, args)

    exe = executable_path(app)
    cmd = Cmd(`$exe $flags`; dir = dir)
    log, ok = _run_streaming(cmd, verbose)

    if check && !ok
        error("""
              DiFfRG run failed (exit != 0): $exe
              output directory: $dir
              last output:
              $(_tail(log, 40))
              """)
    end
    return RunResult(app, dir, nest_pairs(params), ok, log, _output_files(dir))
end

"""
    run(app::App; kwargs...) -> RunResult

Alias for [`run_app`](@ref).
"""
Base.run(app::App; kwargs...) = run_app(app; kwargs...)

nest_pairs(params) = Dict{String,Any}(pointer(k) => v for (k, v) in kvpairs(params))

"""
    prepare_parameter_file(app, dir, config) -> String

Write the parameter file for one run into `dir`, starting from the app's own and
merging `config` into it.

Use `config` for values that belong in the file -- grid specifications, tolerances,
anything a scan holds fixed. Use `params` for the values being varied: those go
on the command line, which is what keeps a sweep free of rebuilds.

A generated app ships no parameter file of its own, so its schema comes from the
binary via `--generate-parameter-file` and cannot drift from what the binary
actually reads.
"""
function prepare_parameter_file(app::App, dir::AbstractString, config)
    base = base_parameter_file(app)
    dest = joinpath(dir, base === nothing ? "parameter.json" : basename(base))
    if base === nothing
        generated_parameter_file(app, dest)
    else
        cp(base, dest; force = true, follow_symlinks = true)
        chmod(dest, 0o644)
    end
    # Precedence: the app's own defaults, then whatever the generator recorded
    # for this app, then this run's config.
    tree = read_config(dest)
    isempty(app.defaults) || (tree = merge_tree(tree, app.defaults))
    isempty(config) || (tree = merge_tree(tree, nest(config)))
    write_config(dest, tree)
    return dest
end

# Line-buffered so progress appears while the flow runs, and captured at the
# same time so a failure can be reported with its own output attached.
function _run_streaming(cmd::Cmd, verbose::Bool)
    buf = IOBuffer()
    out = Pipe()
    proc = Base.run(pipeline(cmd; stdout = out, stderr = out); wait = false)
    close(out.in)
    for line in eachline(out)
        verbose && println(line)
        println(buf, line)
    end
    wait(proc)
    return String(take!(buf)), success(proc)
end

function _tail(s::AbstractString, n::Integer)
    lines = split(s, '\n')
    return join(lines[max(1, end - n + 1):end], '\n')
end

function _output_files(dir::AbstractString)
    out = String[]
    for (root, _, files) in walkdir(dir)
        for f in files
            startswith(f, '.') || push!(out, relpath(joinpath(root, f), dir))
        end
    end
    return sort!(out)
end

"""
    scan(app; config=(), outdir=nothing, verbose=false, check=false, kwargs...) -> Vector{RunResult}

Run the application once for every point of a parameter grid.

The grid is the cartesian product of the keyword arguments, so

```julia
scan(app; T = 0.01:0.01:0.2, muq = [0.0, 0.3])
```

is forty runs, ordered with the first keyword varying fastest. The application
is built once and each point is a separate process differing only in its
override flags, so nothing is recompiled.

Use `config` for what the sweep holds fixed; it is merged into the parameter
file every point is given.

`check` defaults to `false`: a sweep usually crosses into regions where the flow
legitimately fails, and one such point should not abort the sweep. Inspect
`success` on each result.
"""
function scan(app::App;
              config = (),
              outdir = nothing,
              verbose::Bool = false,
              check::Bool = false,
              args::AbstractVector{<:AbstractString} = String[],
              kwargs...)
    isempty(kwargs) && throw(ArgumentError("scan needs at least one parameter to vary"))
    build!(app; verbose = true)

    keys_ = collect(Base.keys(kwargs))
    axes_ = [collect(v) for v in Base.values(kwargs)]
    root = outdir === nothing ? mktempdir(; prefix = "diffrg_scan_", cleanup = false) :
           abspath(expanduser(String(outdir)))
    mkpath(root)

    results = RunResult[]
    for point in Iterators.product(axes_...)
        params = NamedTuple{Tuple(keys_)}(point)
        tag = scan_name(params)
        push!(results, run_app(app; params = params, config = config,
                               outdir = joinpath(root, tag), name = tag,
                               args = args, verbose = verbose, check = check))
    end
    return results
end

"""
    scan_name(params) -> String

The directory and output name for one point of a sweep.

Uses the `T:0.1_muq:0.3` spelling the Python package already writes and parses,
so a sweep produced from Julia can be read by the existing tooling.
"""
function scan_name(params)
    parts = String[]
    for (k, v) in kvpairs(params)
        push!(parts, string(last(split(pointer(k), '/')), ":", v))
    end
    return join(parts, "_")
end
