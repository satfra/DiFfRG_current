# Build an app, run it, read the result.
#
# Uses Tutorials/tut1, which is Burgers' equation -- no generated flows, so it
# builds in about a minute and the answer is one you can check by hand.
#
#   julia --project=. examples/01_run_and_read.jl

using DiFfRG

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))

app = App(joinpath(REPO, "Tutorials", "tut1"))
println("app        : ", app)
println("install    : ", app.install)
println("build tree : ", app.builddir)

# tut1's initial condition is u(0, x) = a + b*x + c*x^2 + d*x^3 on x in [0, 1].
# `params` become command-line overrides; `config` is merged into the parameter
# file. Nothing here triggers a rebuild.
res = run(app;
          params = (a = 0.0, b = 1.0, c = 0.0, d = 0.0),
          config = ("/timestepping/final_time" => 0.5,))

println("\nrun        : ", res)
println("output dir : ", res.outdir)

data = only(read_h5(res))
println("\n", data)

# The output carries the parameters that produced it, so a result read months
# later still says what it was.
println("physical   : ", data.config["physical"])

frames = data.fe["FE"]
println("\nframe times: ", round.(frames.times; digits = 3))
println("fields     : ", sort(collect(keys(frames.fields))))

u0 = frames.fields["u"][1]
x = vec(frames.nodes[1])
println("\nu at t=0   : range ", extrema(u0), " over x in ", extrema(x))
println("            (should be exactly 0..1, since u(0,x) = x)")

# Burgers' equation with u(0,x) = x has the exact solution u(t,x) = x/(1+t),
# so the maximum at t = 0.5 must be 1/1.5 = 0.6667.
uT = frames.fields["u"][end]
tT = frames.times[end]
@assert isapprox(maximum(uT), 1 / (1 + tT); rtol = 1e-3)
println("u at t=", tT, " : max ", round(maximum(uT); digits = 5),
        "  (exact: ", round(1 / (1 + tT); digits = 5), ")")
