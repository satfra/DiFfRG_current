"""
Which prebuilt bundle this machine can actually run.

Julia's package manager selects artifacts by platform triplet, and a triplet
says nothing about instruction-set level or C library version. The published
bundles are built for `x86-64-v3` on a glibc 2.34 floor, so `x86_64-linux-gnu`
would happily resolve on a pre-2013 CPU and then die with an illegal
instruction. These checks are what the triplet cannot express, and mirror the
ones `install-diffrg-deps.sh` performs.
"""

"""
    supports_x86_64_v3() -> Union{Bool,Nothing}

Whether this CPU has the `x86-64-v3` baseline (AVX2 and FMA).

`nothing` where the question does not apply or cannot be answered: a non-x86
machine, or a system exposing neither `ld.so --help` nor `/proc/cpuinfo`.
"""
function supports_x86_64_v3()
    Sys.ARCH === :x86_64 || return nothing

    # The dynamic loader is authoritative: it reports the levels it will
    # actually dispatch on.
    try
        out = read(`ld.so --help`, String)
        if occursin("x86-64-v3", out)
            for line in split(out, '\n')
                occursin("x86-64-v3", line) && return occursin("supported", line)
            end
        end
    catch
        # No ld.so on this system, or it does not take --help.
    end

    if isfile("/proc/cpuinfo")
        flags = read("/proc/cpuinfo", String)
        return occursin("avx2", flags) && occursin("fma", flags)
    end
    return nothing
end

"""
    glibc_version() -> Union{VersionNumber,Nothing}

The system glibc version, or `nothing` where there is none to speak of.
"""
function glibc_version()
    Sys.islinux() || return nothing
    try
        line = first(split(read(`ldd --version`, String), '\n'))
        m = match(r"(\d+)\.(\d+)\s*$", strip(line))
        m === nothing && return nothing
        return VersionNumber(parse(Int, m.captures[1]), parse(Int, m.captures[2]))
    catch
        return nothing
    end
end

"""
    macos_version() -> Union{VersionNumber,Nothing}

The macOS product version, or `nothing` elsewhere.
"""
function macos_version()
    Sys.isapple() || return nothing
    try
        parts = split(strip(read(`sw_vers -productVersion`, String)), '.')
        nums = [parse(Int, p) for p in parts[1:min(3, end)]]
        while length(nums) < 3
            push!(nums, 0)
        end
        return VersionNumber(nums...)
    catch
        return nothing
    end
end

"""
    bundle_variant() -> String

The dependency bundle this machine should use.

Defaults to the CPU build; set `DIFFRG_VARIANT` to pick another, since Pkg has
no way to express "the CUDA build of a non-Julia toolchain".
"""
function bundle_variant()
    haskey(ENV, "DIFFRG_VARIANT") && return ENV["DIFFRG_VARIANT"]
    if Sys.isapple() && Sys.ARCH === :aarch64
        return "macos-arm64-cpu"
    elseif Sys.islinux() && Sys.ARCH === :x86_64
        return "linux-x86_64-v3-cpu"
    end
    return ""
end

"""
    check_bundle_runnable(; glibc_floor = v"2.34", strict = true) -> Bool

Whether a published bundle will run here, throwing with the reason when not.

Pass `strict = false`, or set `DIFFRG_SKIP_CPU_CHECK=1`, to report rather than
throw -- the same escape hatch `install-diffrg-deps.sh --skip-cpu-check` offers.
The binaries will still fault on an unsupported CPU; the flag only moves where
that is noticed.
"""
function check_bundle_runnable(; glibc_floor::VersionNumber = v"2.34",
                               macos_floor::VersionNumber = v"14.0",
                               strict::Bool = true)
    skip = get(ENV, "DIFFRG_SKIP_CPU_CHECK", "0") == "1"
    problems = String[]

    # Each platform has one floor the triplet cannot express: an instruction-set
    # baseline plus glibc on Linux, the deployment target on macOS.
    v3 = supports_x86_64_v3()
    v3 === false && push!(problems,
        "this CPU does not support x86-64-v3 (AVX2 + FMA), which the prebuilt " *
        "bundles require")

    have = glibc_version()
    if have !== nothing && have < glibc_floor
        push!(problems, "this system's glibc ($have) is older than the bundle " *
                        "requires ($glibc_floor)")
    end

    mac = macos_version()
    if mac !== nothing && mac < macos_floor
        push!(problems, "this macOS ($mac) is older than the bundle's deployment " *
                        "target ($macos_floor)")
    end

    variant = bundle_variant()
    isempty(variant) && push!(problems,
        "no prebuilt bundle is published for $(Sys.ARCH) on " *
        (Sys.isapple() ? "macOS" : Sys.islinux() ? "Linux" : string(Sys.KERNEL)) *
        " (they cover Linux x86_64 and macOS arm64 only)")

    isempty(problems) && return true
    message = """
              A prebuilt DiFfRG dependency bundle cannot be used here:
              $(join(("  - " * p for p in problems), "\n"))

              Build from source instead:
                bash <(curl -sL https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install_diffrg.sh) --mode source
              """
    (skip || !strict) && (@warn message; return false)
    error(message)
end

"""
    artifact_prefix() -> Union{String,Nothing}

The DiFfRG install carried by a bound artifact, if there is one.

Returns `nothing` when the package ships no artifact for this platform, which is
the case until the release bundle carries the library itself -- see
`artifact_status()`.
"""
function artifact_prefix()
    toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    isfile(toml) || return nothing
    variant = bundle_variant()
    isempty(variant) && return nothing
    try
        @eval Main using Pkg.Artifacts
        hash = Base.invokelatest(Main.Artifacts.artifact_hash, variant, toml)
        hash === nothing && return nothing
        Base.invokelatest(Main.Artifacts.artifact_exists, hash) || return nothing
        path = Base.invokelatest(Main.Artifacts.artifact_path, hash)
        return is_diffrg_prefix(path) ? path : nothing
    catch
        return nothing
    end
end

"""
    artifact_status() -> String

Why there is, or is not, a usable artifact -- for diagnosing an installation.
"""
function artifact_status()
    io = IOBuffer()
    println(io, "platform : ", Sys.ARCH, " / ", Sys.KERNEL)
    println(io, "variant  : ", isempty(bundle_variant()) ? "(none published)" : bundle_variant())
    println(io, "x86-64-v3: ", something(supports_x86_64_v3(), "n/a"))
    println(io, "glibc    : ", something(glibc_version(), "n/a"))
    println(io, "macOS    : ", something(macos_version(), "n/a"))
    toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    if !isfile(toml)
        println(io, "artifact : none bound.")
        println(io, """
                    The published bundles carry dependencies only -- deal.II,
                    Kokkos, Boost, TBB, SUNDIALS, HDF5 -- and not libDiFfRG.a or
                    the headers, so no artifact can supply a complete install on
                    its own. `install_diffrg!()` does the rest: it downloads the
                    bundle for this platform and builds the library against it,
                    a few minutes rather than the hours a full dependency build
                    takes.""")
    else
        p = artifact_prefix()
        println(io, "artifact : ", p === nothing ? "bound but not usable here" : p)
    end
    return String(take!(io))
end
