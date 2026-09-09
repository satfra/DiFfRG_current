"""
Configuring and building a DiFfRG application.

The executable is treated as opaque: this layer knows how to produce it and how
to start it, but makes no assumption about what one invocation does. Some apps
run a single flow; others (a tuning driver, say) run many and write a directory
per probe.
"""

const _BUILD_ROOT = joinpath(get(ENV, "XDG_CACHE_HOME", joinpath(homedir(), ".cache")),
                             "DiFfRG.jl", "builds")

"""
    App

A DiFfRG application: a source directory containing a `CMakeLists.txt`, built
against a located installation.

Construct with [`App`](@ref); build with [`build!`](@ref); run with
[`run_app`](@ref) or [`scan`](@ref).
"""
mutable struct App
    source::String
    target::Union{String,Nothing}
    install::DiFfRGInstall
    builddir::String
    build_type::String
    jobs::Int
    cmake_args::Vector{String}
    parameter_file::Union{String,Nothing}
    defaults::Dict{String,Any}
    built::Bool
end

"""
    App(source; target=nothing, install=nothing, build_type="Release",
        jobs=default_jobs(), builddir=nothing, cmake_args=String[])

Prepare the application whose sources live in `source`.

`target` names the executable to build when the project defines several (as
`Examples/ONfiniteT` does, with one per discretization); it defaults to the
project's only executable, resolved at build time.

The build tree is package-owned and keyed on the source contents and options, so
switching options does not repeatedly invalidate one directory and the user's
own `build/` is never touched. Pass `builddir` to override.
"""
function App(source::AbstractString;
             target = nothing,
             install = nothing,
             build_type::AbstractString = "Release",
             jobs::Integer = default_jobs(),
             builddir = nothing,
             parameter_file = nothing,
             defaults = Dict{String,Any}(),
             cmake_args::AbstractVector{<:AbstractString} = String[])
    src = abspath(expanduser(String(source)))
    isdir(src) || throw(ArgumentError("no such directory: $src"))
    isfile(joinpath(src, "CMakeLists.txt")) ||
        throw(ArgumentError("$src has no CMakeLists.txt; it is not a DiFfRG app"))
    inst = install === nothing ? find_install() : install
    bdir = builddir === nothing ?
           joinpath(_BUILD_ROOT, build_key(src, build_type, cmake_args, inst.prefix)) :
           abspath(expanduser(String(builddir)))
    app = App(src, target === nothing ? nothing : String(target), inst, bdir,
              String(build_type), Int(jobs), String.(cmake_args),
              parameter_file === nothing ? nothing : String(parameter_file),
              Dict{String,Any}(defaults), false)
    # Resolve an explicitly named parameter file now: a typo should surface here
    # rather than after a build, when it would look like a runtime failure.
    parameter_file === nothing || _resolve_parameter_file(app)
    return app
end

function Base.show(io::IO, app::App)
    print(io, "App(", app.source)
    app.target === nothing || print(io, ", target=", app.target)
    print(io, app.built ? ", built" : ", not built", ")")
end

"""
    default_jobs() -> Int

How many compile jobs to run.

Bounded by cores, by memory at roughly one job per 2 GB (the documented budget;
deal.II linking is what makes it necessary), and by a hard ceiling of 8 -- an
app TU is a heavy template instantiation and this is a workstation, not a
build farm.

`Sys.free_memory()` is memory that is free right now, not counting reclaimable
page cache, so this errs low on a busy machine. That is the right direction to
err: too few jobs is slow, too many is an OOM in the middle of a link.
"""
function default_jobs()
    by_memory = floor(Int, Sys.free_memory() / (2 * 1024^3))
    return clamp(min(Sys.CPU_THREADS, by_memory), 1, 8)
end

# Files that can change what the build produces. Anything else in the tree
# (outputs, notebooks, figures) must not invalidate it, or the cache never hits.
const _SOURCE_EXTS = (".cc", ".cpp", ".cxx", ".hh", ".hpp", ".h", ".txt", ".cmake", ".in")

function _source_files(dir::AbstractString)
    out = String[]
    for (root, dirs, files) in walkdir(dir)
        filter!(d -> !startswith(d, '.') && d != "build" && d != "CMakeFiles", dirs)
        for f in files
            (lowercase(splitext(f)[2]) in _SOURCE_EXTS) && push!(out, joinpath(root, f))
        end
    end
    return sort!(out)
end

"""
    build_key(source, build_type, cmake_args, install) -> String

A short digest of everything that determines the build tree, used to name it so
distinct configurations do not evict each other.

The install prefix is part of it: two installs can differ in headers, in feature
flags or in ABI, and a tree built against one is not valid against the other.

Deliberately not keyed on the target: a project's targets share one CMake tree,
so building `CG` and then `LDG` of `Examples/ONfiniteT` reuses the generated
flow library instead of compiling it once per target.
"""
function build_key(source, build_type, cmake_args, install)
    ctx = SHA.SHA256_CTX()
    SHA.update!(ctx, codeunits(string(build_type, '\0', join(cmake_args, '\0'),
                                      '\0', install)))
    for f in _source_files(source)
        SHA.update!(ctx, codeunits(relpath(f, source)))
        SHA.update!(ctx, read(f))
    end
    return string(basename(source), "-", bytes2hex(SHA.digest!(ctx))[1:12])
end

"""
    build!(app; force=false, verbose=true) -> App

Configure and build the application, reusing an up-to-date build tree.

The build tree is keyed on the source contents, so an unchanged app is a no-op
and an edited one is rebuilt in a fresh directory.
"""
function build!(app::App; force::Bool = false, verbose::Bool = true)
    # One stamp per target, since targets share the tree.
    stamp = joinpath(app.builddir,
                     ".diffrg_jl_built" * (app.target === nothing ? "" : "_" * app.target))
    if !force && isfile(stamp) && isfile(executable_path(app))
        app.built = true
        return app
    end
    mkpath(app.builddir)
    # Drop the stamp first: a forced rebuild that fails must not leave the tree
    # looking up to date.
    rm(stamp; force = true)
    # CMake resolves <pkg>_DIR to the directory holding <pkg>Config.cmake, not to
    # the install prefix; CMAKE_PREFIX_PATH covers CMakeLists that hint at the
    # prefix instead.
    cfg = `cmake -S $(app.source) -B $(app.builddir)
           -DCMAKE_BUILD_TYPE=$(app.build_type)
           -DDiFfRG_DIR=$(dirname(config_file(app.install.prefix)))
           -DCMAKE_PREFIX_PATH=$(app.install.prefix)
           $(app.cmake_args)`
    _run_logged(cfg, "configure", verbose)
    bld = app.target === nothing ?
          `cmake --build $(app.builddir) -j $(app.jobs)` :
          `cmake --build $(app.builddir) --target $(app.target) -j $(app.jobs)`
    _run_logged(bld, "build", verbose)
    # The build directory is named by a content hash, so record what it was built
    # from -- otherwise the cache is unreadable to anyone inspecting it by hand.
    write(stamp, """
                 source: $(app.source)
                 target: $(something(app.target, "(default)"))
                 type:   $(app.build_type)
                 built:  $(Dates.now())
                 """)
    app.built = true
    return app
end

function _run_logged(cmd::Cmd, what::AbstractString, verbose::Bool)
    if verbose
        run(cmd)
    else
        out = IOBuffer()
        try
            run(pipeline(cmd; stdout = out, stderr = out))
        catch
            error("DiFfRG app $what failed:\n" * String(take!(out)))
        end
    end
    return nothing
end

"""
    base_parameter_file(app) -> Union{String,Nothing}

The app's own parameter file, which a run starts from.

A driver may carry its own default file name rather than share the project's --
`Examples/ONfiniteT/KT.cc` calls `Init(argc, argv, "parameter_KT.toml")`, and
running it against `parameter.toml` would silently use the wrong grid. So a
`parameter_<target>.*` beside the sources wins for that target, and the shared
`parameter.*` is the fallback. Within each, `.json` is preferred over `.toml`,
matching the order `ConfigurationHelper` resolves them in.

Pass `parameter_file` to `App` to name the file outright.
"""
function base_parameter_file(app::App)
    app.parameter_file === nothing || return _resolve_parameter_file(app)
    stems = app.target === nothing ? ("parameter",) : ("parameter_" * app.target, "parameter")
    for stem in stems, ext in (".json", ".toml", ".tml")
        path = joinpath(app.source, stem * ext)
        isfile(path) && return path
    end
    return nothing
end

function _resolve_parameter_file(app::App)
    path = isabspath(app.parameter_file) ? app.parameter_file :
           joinpath(app.source, app.parameter_file)
    isfile(path) || throw(ArgumentError("no such parameter file: $path"))
    return path
end

"""
    generated_parameter_file(app, dest) -> String

Ask the built executable for a default parameter file.

Used when the app ships none. This is also how the app's schema is discovered
rather than hardcoded here, so it cannot drift from the binary.
"""
function generated_parameter_file(app::App, dest::AbstractString)
    mkpath(dirname(abspath(dest)))
    exe = executable_path(app)
    cd(dirname(abspath(dest))) do
        run(pipeline(`$exe --generate-parameter-file -p $(basename(dest))`;
                     stdout = devnull, stderr = devnull))
    end
    return dest
end

"""
    executable_path(app) -> String

Where the built executable is, resolving `target=nothing` to the project's only
executable if there is exactly one.
"""
function executable_path(app::App)
    if app.target !== nothing
        direct = joinpath(app.builddir, app.target)
        isfile(direct) && return direct
        # A project may send its output elsewhere in the tree via
        # RUNTIME_OUTPUT_DIRECTORY, so fall back to searching for the target by
        # name rather than reporting it as missing.
        for (root, _, files) in walkdir(app.builddir)
            app.target in files && return joinpath(root, app.target)
        end
        return direct
    end
    cands = _executables(app.builddir)
    isempty(cands) && return joinpath(app.builddir, basename(app.source))
    length(cands) == 1 && return only(cands)
    throw(ArgumentError(
        "$(app.source) builds several executables ($(join(sort!(basename.(cands)), ", "))); " *
        "pass target= to choose one"))
end

# Executables sit at the top of the build tree, alongside the parameter-file
# symlinks CMake puts there. The permission bit cannot tell them apart -- a
# checkout on a filesystem that reports everything as 0777 marks the parameter
# files executable too -- so the magic number is what decides.
function _executables(builddir::AbstractString)
    isdir(builddir) || return String[]
    out = String[]
    for f in readdir(builddir; join = true)
        isfile(f) || continue
        startswith(basename(f), '.') && continue
        _is_native_binary(f) && push!(out, f)
    end
    return out
end

const _ELF_MAGIC = UInt8[0x7f, 0x45, 0x4c, 0x46]
# Mach-O 64-bit, both endiannesses, plus the universal binary wrapper.
const _MACHO_MAGICS = (UInt8[0xcf, 0xfa, 0xed, 0xfe], UInt8[0xfe, 0xed, 0xfa, 0xcf],
                       UInt8[0xca, 0xfe, 0xba, 0xbe], UInt8[0xbe, 0xba, 0xfe, 0xca])

function _is_native_binary(path::AbstractString)
    head = try
        open(path, "r") do io
            read(io, 4)
        end
    catch
        return false
    end
    length(head) == 4 || return false
    return head == _ELF_MAGIC || any(==(head), _MACHO_MAGICS)
end
