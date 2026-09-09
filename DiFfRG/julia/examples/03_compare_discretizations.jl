# The same physics through three discretizations.
#
# Examples/ONfiniteT ships five drivers -- CG.cc, dDG.cc, LDG.cc, KT.cc,
# KT_sigma.cc -- differing in about three `using` lines each. Comparing them by
# hand means several builds and several invocations. Here it is one loop, and
# the targets share a build tree so the generated flow library is compiled once
# rather than once per scheme.
#
# KT is deliberately left out. It is not a discretization variant of the same
# problem: it uses a different model header (model_KT.hh) with a different
# parameterization -- Lambda/m2/lambda against the FEM family's
# Lambda/lambda2/lambda4 -- and its own parameter_KT.toml. Putting it in the
# same table would compare two different physical setups. It also writes no
# scalar readouts at all, so it has no CSV to read.
#
#   julia --project=. examples/03_compare_discretizations.jl
#
# The first run compiles three executables and takes several minutes.

using DiFfRG
using Printf

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SOURCE = joinpath(REPO, "Examples", "ONfiniteT")
const T = 0.1

# The FEM family: same model.hh, same parameter.toml, different discretization.
schemes = ["CG", "dDG", "LDG"]
results = Dict{String,Any}()

for scheme in schemes
    app = App(SOURCE; target = scheme)
    @printf("%-5s tree %s   parameters %s\n", scheme, basename(app.builddir),
            basename(DiFfRG.base_parameter_file(app)))
    build!(app; verbose = false)
    results[scheme] = run(app; params = (T = T,), verbose = false, check = false)
end

"""
    pion_mass_squared(result) -> Float64

m^2_pi at the end of the flow, read from the field rather than the readouts.

In the symmetric phase the minimum of the potential sits at rho = 0, so the mass
function evaluated at the first grid point is m^2_pi. Taking it from the field
rather than the readouts keeps the comparison independent of what each model
chose to write out.

The field has to be named rather than guessed: CG and dDG share model.hh and
call it `m2`, while LDG uses model_LDG.hh and calls it `u`, and LDG additionally
carries auxiliary levels (`LDG1`, ...) in the same group.
"""
function pion_mass_squared(result)
    result.success || return NaN
    frames = only(read_h5(result)).fe["FE"]
    name = findfirst(in(keys(frames.fields)), ["m2", "u"])
    name === nothing && return NaN
    rho = vec(frames.nodes[end])
    return vec(frames.fields[["m2", "u"][name]][end])[argmin(rho)]
end

masses = Dict(s => pion_mass_squared(results[s]) for s in schemes)
reference = masses[first(schemes)]

println("\nm^2_pi at T = $T, from the shipped parameters:\n")
@printf("%-8s %-14s %-16s %s\n", "scheme", "m^2_pi", "readout (CSV)", "vs $(first(schemes))")
for scheme in schemes
    files = csv_files(results[scheme])
    csv = isempty(files) ? NaN : last(read_csv(first(files))["m2_piGeV2"])
    @printf("%-8s %-14.6f %-16.6f %+.2e\n",
            scheme, masses[scheme], csv, masses[scheme] / reference - 1)
end

println("""

The three schemes solve the same flow equation and differ only in how field
space is discretized, so their agreement is the check that the discretization is
not what is setting the answer.""")
