"""
Locating the DiFfRG C++ installation that generated apps are built against.

An app is always compiled on this machine -- DiFfRG is a static library -- so
what is needed is a prefix containing `cmake/DiFfRGConfig.cmake`, plus a working
C++ compiler and CMake. `find_package(DiFfRG REQUIRED HINTS <prefix>)` is what
the generated `CMakeLists.txt` ultimately does with it.
"""

"""
    DiFfRGInstall

A located DiFfRG installation.

# Fields
- `prefix`: the install prefix, e.g. `~/.local/share/DiFfRG`.
- `bundled`: the dependency tree (`<prefix>/bundled`), if present.
- `use_cuda`: whether the install was configured with CUDA.
- `use_mpi`: whether the install was configured with MPI.
- `cxx_flags`: the flags the install propagates to every consumer target.
"""
struct DiFfRGInstall
    prefix::String
    bundled::Union{String,Nothing}
    use_cuda::Bool
    use_mpi::Bool
    cxx_flags::String
end

function Base.show(io::IO, inst::DiFfRGInstall)
    feats = String[]
    inst.use_cuda && push!(feats, "CUDA")
    inst.use_mpi && push!(feats, "MPI")
    tag = isempty(feats) ? "CPU only" : join(feats, "+")
    print(io, "DiFfRGInstall(", inst.prefix, ", ", tag, ")")
end

"""
    config_file(prefix) -> String

Path to the exported CMake package config inside `prefix`.

The library installs it to `<prefix>/cmake/`, but a plain GNUInstallDirs layout
puts it under `<prefix>/lib*/cmake/DiFfRG/`, so both are accepted.
"""
function config_file(prefix::AbstractString)
    for rel in ("cmake/DiFfRGConfig.cmake",
                "lib/cmake/DiFfRG/DiFfRGConfig.cmake",
                "lib64/cmake/DiFfRG/DiFfRGConfig.cmake")
        path = joinpath(prefix, rel)
        isfile(path) && return path
    end
    return ""
end

is_diffrg_prefix(prefix::AbstractString) = !isempty(config_file(prefix))

"""
    candidate_prefixes() -> Vector{String}

Places to look for an installation, most specific first.

`DIFFRG_DIR` wins so a session can be pinned to one build; the default install
prefix used by `install_diffrg.sh` and hinted by every example's CMakeLists
comes next.
"""
function candidate_prefixes()
    out = String[]
    for var in ("DIFFRG_DIR", "DiFfRG_DIR")
        haskey(ENV, var) && push!(out, ENV[var])
    end
    # An artifact, where one is bound and usable on this machine.
    art = artifact_prefix()
    art === nothing || push!(out, art)
    # A hand-maintained install at the conventional location comes before the
    # one this package may have made for itself: someone who installed DiFfRG
    # themselves should not have it silently shadowed by our fallback.
    push!(out, joinpath(homedir(), ".local", "share", "DiFfRG"))
    push!(out, install_prefix())
    # DIFFRG_DIR often names the conventional location, which would otherwise
    # appear twice.
    return unique(abspath.(expanduser.(out)))
end

"""
    which_install(io = stdout)

Print every place an install is looked for, in order, and which one is used.

Useful when a build picks up a DiFfRG other than the expected one.
"""
function which_install(io::IO = stdout)
    chosen = ""
    for p in candidate_prefixes()
        usable = is_diffrg_prefix(p)
        if usable && isempty(chosen)
            chosen = p
            println(io, "  -> ", p)
        else
            println(io, usable ? "     " : "  x  ", p)
        end
    end
    isempty(chosen) && println(io, "\n  none usable; see install_diffrg!()")
    return nothing
end

# `set(VAR value)` / `set(VAR "value")` out of the generated config. The config is
# machine-written by configure_package_config_file, so a line-oriented read is
# enough and avoids depending on cmake just to answer a question about flags.
function _read_cmake_set(text::AbstractString, var::AbstractString)
    m = match(Regex("set\\(\\s*" * var * "\\s+(.*?)\\s*\\)", "s"), text)
    m === nothing && return nothing
    val = strip(m.captures[1])
    startswith(val, '"') && endswith(val, '"') && length(val) >= 2 && (val = val[2:end-1])
    return String(val)
end

_cmake_truthy(v) = v !== nothing &&
                   uppercase(strip(v)) in ("ON", "TRUE", "YES", "1", "Y")

"""
    find_install(prefix = nothing) -> DiFfRGInstall

Locate a DiFfRG installation, or throw with instructions for installing one.

Pass `prefix` to use a specific install; otherwise [`candidate_prefixes`](@ref)
is searched in order: `DIFFRG_DIR`, a bound artifact, `~/.local/share/DiFfRG`,
and finally the install this package made for itself. [`which_install`](@ref)
shows the resolution.
"""
function find_install(prefix = nothing)
    prefixes = prefix === nothing ? candidate_prefixes() : [String(prefix)]
    for p in prefixes
        cfg = config_file(p)
        isempty(cfg) && continue
        text = read(cfg, String)
        bundled = joinpath(p, "bundled")
        return DiFfRGInstall(abspath(p),
                             isdir(bundled) ? bundled : nothing,
                             _cmake_truthy(_read_cmake_set(text, "DiFfRG_USE_CUDA")),
                             _cmake_truthy(_read_cmake_set(text, "DiFfRG_MPI")),
                             something(_read_cmake_set(text, "DiFfRG_CXX_FLAGS"), ""))
    end
    searched = join(("  " * p for p in prefixes), "\n")
    error("""
          No DiFfRG installation found. Searched:
          $searched

          Install one with:
            bash <(curl -sL https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh)

          or point DIFFRG_DIR at an existing install prefix (the directory
          containing cmake/DiFfRGConfig.cmake).

          $(artifact_status())
          """)
end
