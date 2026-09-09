# Declare a model in Julia; get a DiFfRG app out.
#
# The physics here is two lines. Everything around it -- the Parameters struct
# reading /physical, the descriptor typedefs, the CRTP base list, the exact
# template signatures, the driver, the build file -- is determined by those two
# lines plus the choice of scheme, and is written for you.
#
# The result is an ordinary DiFfRG project: three files you can read, build by
# hand, or take over if the generator runs out of road.
#
#   julia --project=. examples/04_generate_a_model.jl

using DiFfRG
using Printf

# Burgers' equation, which is Tutorials/tut1. u(0,x) = a + b x + c x^2 + d x^3,
# flux = u^2/2, whose exact solution for u(0,x) = x is u = x/(1+t).
model = Model("Burgers";
    dim = 1,
    fe = [Scalar("u")],
    params = (Lambda = 1.0, a = 0.0, b = 1.0, c = 0.0, d = 0.0),

    initial_condition = cxx"""
        const auto x = pos[0];
        values[idxf("u")] = prm.a + prm.b * x + prm.c * powr<2>(x) + prm.d * powr<3>(x);
        """,

    flux = cxx"""
        const auto u = get<"fe_functions">(sol)[idxf("u")];
        flux[idxf("u")][0] = 0.5 * powr<2>(u);
        """)

println(model, "\n")

dir = mktempdir(; prefix = "diffrg_burgers_", cleanup = false)
app = generate(model, dir;
               discretization = "CG",
               stepper = "SUNDIALS_IDA",
               config = ("/discretization/grid/x_grid" => "0:0.01:1",
                         "/discretization/fe_order" => 3,
                         "/timestepping/final_time" => 0.5))

println("wrote ", join(sort(readdir(dir)), ", "), "\n  in ", dir, "\n")
println(repeat("-", 72))
println(read(joinpath(dir, "model.hh"), String))
println(repeat("-", 72))

build!(app; verbose = false)
res = run(app; verbose = false)

data = only(read_h5(res))
frames = data.fe["FE"]
u = frames.fields["u"][end]
t = frames.times[end]

@printf("\nmax u at t = %.2f : %.6f\n", t, maximum(u))
@printf("exact x/(1+t)      : %.6f\n", 1 / (1 + t))

# Changing the scheme is a keyword, not five files. dDG needs a numerical flux,
# which the default mixin set supplies for the schemes that work on faces.
println("\nthe same model through another discretization:")
ddg = generate(model, mktempdir(; prefix = "diffrg_burgers_ddg_");
               discretization = "dDG",
               config = ("/discretization/grid/x_grid" => "0:0.01:1",
                         "/timestepping/final_time" => 0.5))
println("  ", read(joinpath(ddg.source, "Burgers.cc"), String) |>
              s -> join(filter(l -> startswith(l, "using "), split(s, '\n')), "\n  "))
