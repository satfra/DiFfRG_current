# The same model with no C++ written at all.
#
# Example 04 supplied the member bodies as cxx"...". Here they are Julia
# expressions, translated at generation time -- when the discretization, and so
# the solution bundle, is known.
#
# The restriction is that the emitted code must stay generic in its number type:
# the same body is instantiated with double, with forward-AD types, and on
# device. So the subset is arithmetic over a whitelist and nothing else. What
# falls outside it is a Julia error naming the line, not a template error a
# hundred lines deep.
#
#   julia --project=. examples/06_julia_expressions.jl

using DiFfRG
using Printf

#const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))

# Note: a macro in a keyword argument has to be parenthesised, or it swallows
# the rest of the call.
model = Model("Burgers";
    dim=1,
    fe=[Scalar("u")],
    params=(Lambda=1.0, a=0.0, b=1.0, c=0.0, d=0.0),
    initial_condition=@julia((values, pos) -> begin
        x = pos[1]
        values[:u] = prm.a + prm.b * x + prm.c * x^2 + prm.d * x^3
    end),
    flux=@julia((F, x, sol) -> begin
        u = sol.fe_functions[:u]
        F[:u][1] = 0.5 * u^2
    end)
)

dir = mktempdir(; prefix="diffrg_dsl_", cleanup=false)
app = generate(model, dir,
    # we use the KT discretization, as it can directly handle shocks
    discretization="KT"
)

println("the two bodies became:\n")
hh = read(joinpath(dir, "model.hh"), String)
println(hh[findfirst("  template <typename NT", hh)[1]:end])

# Indices are 1-based in the DSL, as everywhere else in Julia: `pos[1]` is the
# first spatial direction and comes out as `pos[0]`. `u^2` becomes powr<2>(u),
# and every numeric literal is emitted as a double so that `1/2` cannot become
# C++ integer division.

build!(app; verbose=true)
res = run(app; config=(
        "/physical/a" => 0.5,
        "/physical/b" => 4.5,
        "/physical/c" => -9.5,
        "/physical/d" => 4.5,
        "/discretization/grid/x_grid" => "0:0.002:1.5, 1.5:0.01:2",
        "/timestepping/final_time" => 1.0
    ), verbose=false)
u = only(read_h5(res)).fe["FE"].fields["u"]
x = only(read_h5(res)).fe["FE"].nodes

# Plot the solution, if you have Plots.jl installed.
try
    using Plots
    plot(x[1][1, :], u[1], title="Burgers solution", xlabel="x", ylabel="u", legend=false)
    for i in 2:size(u, 1)-1
        plot!(x[i][1, :], u[i])
    end
    plot!(x[end][1, :], u[end])
catch err
    @warn "Plots.jl not installed, skipping plot" exception = (err, catch_backtrace())
end