"""
Installing DiFfRG into a directory this package owns.

The library is not something Julia can ship as an artifact on its own: the
published bundles carry dependencies only, and DiFfRG is a static library that
has to be compiled here anyway. What can be automated is the supported install
path -- download the prebuilt dependency bundle for this platform, then build
just the library against it, which is minutes rather than the hours a full
dependency build takes.

That is exactly what `install_diffrg.sh` does, so this drives the wizard rather
than reimplementing it: one place to fix, and the same bytes a manual install
would produce.
"""

"""
    install_prefix() -> String

The directory this package installs DiFfRG into.

A scratch space belonging to the package, so it survives package updates, is
found automatically by [`find_install`](@ref), and is removed with the package
rather than being left behind in the home directory.
"""
install_prefix() = Scratch.get_scratch!(@__MODULE__, "diffrg")

"""
    wizard_script() -> String

The installer to drive, preferring the one in a local checkout.

Downloads it when the package is used outside a DiFfRG source tree, so a
standalone install still works.
"""
function wizard_script()
    # DiFfRG/julia/src/bootstrap.jl -> the repository root.
    local_copy = normpath(joinpath(@__DIR__, "..", "..", "..", "install_diffrg.sh"))
    isfile(local_copy) && return local_copy

    dest = joinpath(install_prefix(), "install_diffrg.sh")
    isfile(dest) && return dest
    url = "https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh"
    @info "Fetching the DiFfRG installer" url
    mkpath(dirname(dest))
    Downloads.download(url, dest)
    return dest
end

"""
    install_diffrg!(; variant=bundle_variant(), version=nothing, threads=..., gpu=false,
                    mpi=false, mode="prebuilt", force=false, verbose=true) -> String

Install DiFfRG into this package's own directory and return the prefix.

Downloads the prebuilt dependency bundle for this platform and builds the
library against it. `find_install` picks the result up afterwards, so nothing
else has to be configured.

The bundle variant is chosen for the running platform -- `macos-arm64-cpu` on
Apple Silicon, `linux-x86_64-v3-cpu` on Linux x86_64 -- and passed to the
installer explicitly rather than left to its own detection, so `DIFFRG_VARIANT`
selects a different one (`linux-x86_64-v3-cuda12`) predictably.

Prebuilt bundles cover Linux x86_64 and macOS arm64 only; pass `mode="source"`
for anything else, or for MPI, which they are not built with. Requires CMake and
a C++20 compiler either way, since apps are compiled locally regardless.
"""
function install_diffrg!(; variant = bundle_variant(),
                         version = nothing,
                         threads::Integer = default_jobs(),
                         gpu::Bool = false,
                         mpi::Bool = false,
                         mode::AbstractString = "prebuilt",
                         force::Bool = false,
                         verbose::Bool = true)
    prefix = install_prefix()

    # Installing does not make this the install that gets used: anything earlier
    # in the search order still wins, and silently building against a different
    # DiFfRG than the one just installed is exactly the confusion to avoid.
    candidates = candidate_prefixes()
    existing = findfirst(is_diffrg_prefix, candidates)
    if existing !== nothing && candidates[existing] != prefix
        @warn """
              Another DiFfRG install takes precedence and will keep being used. \
              Set DIFFRG_DIR to override it, or leave this one uninstalled.
              """ in_use = candidates[existing] this_one = prefix
    end

    if !force && is_diffrg_prefix(prefix)
        verbose && @info "DiFfRG is already installed here; pass force=true to rebuild" prefix
        return prefix
    end

    if mode == "prebuilt"
        isempty(variant) && error("""
            No prebuilt dependency bundle is published for this platform.
            $(artifact_status())
            Build from source instead:  install_diffrg!(mode = "source")
            """)
        # Refuse before downloading fifty megabytes that cannot run here.
        check_bundle_runnable()
    end

    # A build directory of our own, not the installer's shared /tmp/diffrg-build.
    # That one is reused across runs, and `deal.II_DIR` is a cached CMake PATH
    # variable which is not re-derived once set -- so installing to a second
    # prefix on a machine that already has one silently compiles the library
    # against the *first* install's deal.II and bakes its flags into
    # DiFfRGTargets.cmake.
    args = String[wizard_script(), "--mode", String(mode),
                  "--prefix", prefix,
                  "--build-dir", joinpath(prefix, "build"),
                  "--threads", string(threads),
                  "--no-mathematica", "--no-docs", "--yes"]
    push!(args, gpu ? "--gpu" : "--no-gpu")
    push!(args, mpi ? "--mpi" : "--no-mpi")
    force && push!(args, "--force")
    if mode == "prebuilt"
        append!(args, ["--deps-variant", variant])
        version === nothing || append!(args, ["--deps-version", string(version)])
    end

    verbose && @info "Installing DiFfRG" prefix variant mode threads
    cmd = `bash $args`
    if verbose
        run(cmd)
    else
        out = IOBuffer()
        try
            run(pipeline(cmd; stdout = out, stderr = out))
        catch
            error("DiFfRG installation failed:\n" * String(take!(out)))
        end
    end

    is_diffrg_prefix(prefix) || error("""
        The installer finished but $prefix does not contain a DiFfRG package
        config. Run it by hand to see what happened:
          bash $(wizard_script()) --mode $mode --prefix $prefix --yes
        """)
    return prefix
end

"""
    uninstall_diffrg!()

Remove the DiFfRG install this package owns. Leaves any other install alone.
"""
function uninstall_diffrg!()
    prefix = install_prefix()
    isdir(prefix) && rm(prefix; recursive = true)
    return nothing
end
