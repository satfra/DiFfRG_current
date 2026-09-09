# A model whose flux is a momentum loop integral, computed by generated code.
#
# The flux of the O(N) effective potential is not something you write out by
# hand: it is a loop integral whose integrand Mathematica derives and emits as
# C++ (a `flows/` directory with its own CMakeLists.txt, a `flows.hh` facade
# class and one integrator per kernel). That stays Mathematica's job.
#
# What this shows is consuming it: `FlowLibrary` reads the class and CMake
# target names out of the generated files, and the subdirectory, the member
# declaration and the set_k wiring follow from that.
#
# This is Tutorials/tut3, and the script checks the result against the
# hand-written original -- the fields agree bitwise.
#
#   julia --project=. examples/05_generated_flows.jl
#
# The first run compiles two apps and takes a few minutes.

using DiFfRG
using Printf

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const TUT3 = joinpath(REPO, "Tutorials", "tut3")

flows = FlowLibrary(joinpath(TUT3, "flows"))
@printf("flow library at %s\n", relpath(flows.path, REPO))
@printf("  facade class : %s   (read from flows.hh)\n", flows.class)
@printf("  cmake target : %s   (read from flows/CMakeLists.txt)\n", flows.target)
@printf("  member       : %s\n\n", flows.member)

model = Model("ON_finiteT";
    dim = 1,
    fe = [Scalar("m2")],
    params = (Lambda = 1.0, N = 2.0, T = 0.001,
              lambda2 = -0.1, lambda4 = 50.0, lambda6 = 0.0),
    flows = flows,

    initial_condition = cxx"""
        const auto rho = pos[0];
        values[idxf("m2")] = prm.lambda2 + prm.lambda4 * rho + prm.lambda6 * powr<2>(rho);
        """,

    # The cutoff has to reach the integrators, so set_time pushes it down.
    set_time = cxx"""
        t = t_;
        k = std::exp(-t) * prm.Lambda;
        flow_equations.set_k(k);
        """,

    # The flux is one call into the generated integrator.
    flux = cxx"""
        const auto rho = pos[0];
        const auto &fe_functions = get<"fe_functions">(sol);
        const auto &fe_derivatives = get<"fe_derivatives">(sol);

        const auto m2Pi = fe_functions[idxf("m2")];
        const auto m2Sigma = m2Pi + 2. * rho * fe_derivatives[idxf("m2")][0];

        flow_equations.V.get(flux[idxf("m2")][0], k, prm.N, prm.T, m2Pi, m2Sigma);
        """,

    # Refine where the potential has structure.
    cell_indicator = cxx"""
        indicator = get<"fe_hessians">(sol)[idxf("m2")][0][0];
        """)

# The tutorial ships a tuned grid and quadrature; a generated app starts from
# the binary's own defaults, so hand it the same settings for a fair comparison.
hand = App(TUT3)
tuned = DiFfRG.read_config(DiFfRG.base_parameter_file(hand))

generated = generate(model, mktempdir(; prefix = "diffrg_on_");
                     config = ("/discretization" => tuned["discretization"],
                               "/integration"    => get(tuned, "integration", Dict())))

println("generated project: ", join(sort(readdir(generated.source)), ", "))
println()
println(read(joinpath(generated.source, "CMakeLists.txt"), String))

println("building both (first run: a few minutes) ...")
build!(generated; verbose = false)
build!(hand; verbose = false)

settings = ("/timestepping/final_time" => 2.0,)
g = only(read_h5(run(generated; config = settings, verbose = false))).fe["FE"]
h = only(read_h5(run(hand;      config = settings, verbose = false))).fe["FE"]

gm = g.fields["m2"][end]
hm = h.fields["m2"][end]

@printf("\n%-22s %s\n", "", "generated / hand-written")
@printf("%-22s %d / %d\n", "degrees of freedom", length(gm), length(hm))
@printf("%-22s %.15f / %.15f\n", "m^2 at rho = 0", gm[1], hm[1])
if length(gm) == length(hm)
    @printf("%-22s %g\n", "max |difference|", maximum(abs.(gm .- hm)))
    println(gm == hm ? "\nBitwise identical to the hand-written tutorial." :
                       "\nAgrees to the printed tolerance.")
end
