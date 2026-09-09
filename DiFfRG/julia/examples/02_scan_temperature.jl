# A temperature scan of the O(N) model, and the chiral condensate it gives.
#
# This is what the driver layer is for: the app is compiled once and every
# temperature is a separate process differing only in `-sd /physical/T=...`, so
# the scan itself costs no compilation at all.
#
#   julia --project=. examples/02_scan_temperature.jl

using DiFfRG
using Printf

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))

app = App(joinpath(REPO, "Examples", "ONfiniteT"); target = "CG")
println("building ", basename(app.source), " [", app.target, "] ...")
build!(app; verbose = false)

# The shipped parameters (lambda2 = -0.1) sit in the symmetric phase, where
# sigma is zero at every temperature and there is nothing to see. lambda2 = -0.5
# puts the model in the broken phase at low temperature, so raising T melts the
# condensate and the transition is visible.
const LAMBDA2 = -0.5

temperatures = [0.001, 0.05, 0.10, 0.15, 0.20, 0.22, 0.24, 0.26, 0.30]
println("scanning ", length(temperatures), " temperatures on that one build\n")

sweep = scan(app; T = temperatures, config = ("/physical/lambda2" => LAMBDA2,))

"""
    condensate(result) -> Float64

The chiral condensate at the end of the flow.

The model writes `sigmaGeV` at the equation-of-motion point once per output
step, so the last row is the infrared value -- the order parameter.
"""
function condensate(result)
    files = csv_files(result)
    isempty(files) && return NaN
    series = get(read_csv(first(files)), "sigmaGeV", Float64[])
    return isempty(series) ? NaN : last(series)
end

@printf("%-10s %-6s %-12s %s\n", "T [GeV]", "ok", "sigma [GeV]", "")
for (T, res) in zip(temperatures, sweep)
    sigma = condensate(res)
    bar = "#"^round(Int, 400 * sigma)
    @printf("%-10.3f %-6s %-12.5f %s\n", T, res.success ? "yes" : "NO", sigma, bar)
end

println("""

sigma melts between T = 0.20 and T = 0.25 and is exactly zero above: that is the
chiral transition. Nothing was recompiled between these points.

results under: $(dirname(first(sweep).outdir))""")
