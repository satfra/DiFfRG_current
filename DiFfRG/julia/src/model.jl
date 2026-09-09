"""
Emitting a DiFfRG model and its driver.

What a model has to say is short -- which degrees of freedom exist, which
parameters it reads, and the bodies of a few member functions. What surrounds
that is boilerplate whose every line is determined by those choices: a
`Parameters` struct mirroring the parameter file, descriptor typedefs, a CRTP
base list, exact template signatures, and a `main()` that differs between apps
only in four `using` lines.

This module writes the surroundings. Member bodies are supplied as C++ by
[`cxx`](@ref); a Julia-expression layer on top of this is a later step.
"""

# --------------------------------------------------------------------- components

"""
    Component

One named degree of freedom: `Scalar("u")` or `FunctionND("ZA", 96)`.
"""
struct Component
    name::String
    dims::Vector{Int}
end

"""
    Scalar(name) -> Component

A single value, emitted as `Scalar<"name">`.
"""
Scalar(name) = Component(String(name), Int[])

"""
    FunctionND(name, dims...) -> Component

A grid-valued quantity, emitted as `FunctionND<"name", dims...>`.
"""
FunctionND(name, dims::Integer...) = Component(String(name), collect(Int, dims))

cxx_type(c::Component) =
    isempty(c.dims) ? "Scalar<\"$(c.name)\">" :
    "FunctionND<\"$(c.name)\", $(join(c.dims, ", "))>"

Base.show(io::IO, c::Component) = print(io, cxx_type(c))

# --------------------------------------------------------------------- raw C++

"""
    CxxBody

A C++ member body, supplied verbatim.

Built with the [`cxx`](@ref) string macro. The generator emits the signature
around it and never inspects the text, so anything the compiler accepts is
allowed here.
"""
struct CxxBody
    code::String
end

"""
    cxx"..." -> CxxBody

A raw C++ member body.

```julia
flux = cxx\"\"\"
    const auto &fe_functions = get<"fe_functions">(sol);
    flux[idxf("u")][0] = 0.5 * powr<2>(fe_functions[idxf("u")]);
\"\"\"
```
"""
macro cxx_str(s)
    return CxxBody(s)
end

body_code(b::CxxBody) = b.code
body_code(s::AbstractString) = String(s)

# --------------------------------------------------------------------- flows

"""
    FlowLibrary

A generated flow library: the `flows/` directory a Mathematica `UpdateFlows`
run produces, with its own `CMakeLists.txt`, a `flows.hh` facade class and one
integrator per kernel.

The generated project adds it as a subdirectory, links the library, declares the
facade as a member, and pushes the cutoff into it from `set_time`.
"""
struct FlowLibrary
    path::String
    class::String
    target::String
    member::String
end

"""
    FlowLibrary(path; class=nothing, target=nothing, member="flow_equations")

Attach the generated flows at `path`.

The class and CMake target names are read out of the generated
`CMakeLists.txt` and `flows.hh` when not given, since `UpdateFlows` names both
after the library.
"""
function FlowLibrary(path::AbstractString; class = nothing, target = nothing,
                     member::AbstractString = "flow_equations")
    dir = abspath(expanduser(String(path)))
    isdir(dir) || throw(ArgumentError("no such flows directory: $dir"))
    cml = joinpath(dir, "CMakeLists.txt")
    isfile(cml) ||
        throw(ArgumentError("$dir has no CMakeLists.txt; it is not a generated flows directory"))

    detected = match(r"add_library\(\s*(\w+)\s+STATIC", read(cml, String))
    tgt = target === nothing ?
          (detected === nothing ?
           throw(ArgumentError("could not find add_library(<name> STATIC ...) in $cml; pass target=")) :
           detected.captures[1]) : String(target)

    cls = class
    if cls === nothing
        hh = joinpath(dir, "flows.hh")
        isfile(hh) || throw(ArgumentError("$dir has no flows.hh; pass class="))
        m = match(r"class\s+(\w+)", read(hh, String))
        m === nothing && throw(ArgumentError("could not find a class in $hh; pass class="))
        cls = m.captures[1]
    end
    return FlowLibrary(dir, String(cls), tgt, String(member))
end

# --------------------------------------------------------------------- the model

"""
    Model

Everything needed to emit a model header and its driver.

Build one with the [`Model`](@ref) keyword constructor and hand it to
[`generate`](@ref).
"""
struct Model
    name::String
    dim::Int
    fe::Vector{Component}
    variables::Vector{Component}
    extractors::Vector{Component}
    params::Vector{Pair{String,Any}}
    mixins::Vector{String}
    constraint_component::Union{String,Nothing}
    bodies::Dict{String,Any}
    flows::Union{FlowLibrary,Nothing}
    includes::Vector{String}
    preamble::String
end

"""
    Model(name; dim=1, fe=[], variables=[], extractors=[], params=(),
          mixins=nothing, flows=nothing, includes=[], preamble="", bodies...)

Declare a model.

`params` become both a `Parameters` struct reading `/physical/<key>` and the
defaults written into the parameter file, so a name is spelled once instead of
the three times a hand-written model needs.

`mixins` names the CRTP bases; when omitted, a default set is chosen for the
discretization at generation time. Member bodies are passed as further keyword
arguments -- `flux`, `source`, `initial_condition`, `dt_variables`, `extract`,
`readouts`, ... -- each a [`cxx`](@ref) body.

Two conventions are easy to get backwards and silent when wrong:

  * RG time runs as `t = ln(Lambda/k)`, so `k = Lambda * exp(-t)` and `t`
    increases towards the infrared.
  * `dt_variables` fills the *residual*, and the assembler takes
    `A_dot = -residual`. A variable that should decay is written with a positive
    residual.
"""
function Model(name::AbstractString;
               dim::Integer = 1,
               fe = Component[],
               variables = Component[],
               extractors = Component[],
               params = (),
               mixins = nothing,
               constraint_component = nothing,
               flows::Union{FlowLibrary,Nothing} = nothing,
               includes::AbstractVector{<:AbstractString} = String[],
               preamble::AbstractString = "",
               bodies...)
    known = keys(MODEL_MEMBERS)
    for (k, _) in bodies
        String(k) in known || throw(ArgumentError(
            "unknown model member $(repr(String(k))); known members are " *
            join(sort(collect(known)), ", ")))
    end
    return Model(String(name), Int(dim),
                 collect(Component, fe), collect(Component, variables),
                 collect(Component, extractors),
                 [String(k) => v for (k, v) in kvpairs(params)],
                 mixins === nothing ? String[] : String.(collect(mixins)),
                 constraint_component === nothing ? nothing : String(constraint_component),
                 Dict{String,Any}(String(k) => v for (k, v) in bodies),
                 flows, String.(collect(includes)), String(preamble))
end

function Base.show(io::IO, m::Model)
    print(io, "Model(", m.name, ", dim=", m.dim,
          ", fe=", length(m.fe), ", variables=", length(m.variables),
          ", members=", length(m.bodies), ")")
end

"""
The member functions a model may define, with the signature emitted around each
body. `{dim}` is substituted; the bodies themselves are never inspected.

Taken from `def::AbstractModel`, where every member but `initial_condition` has
a working default -- so only what the user supplied is emitted.
"""
const MODEL_MEMBERS = Dict(
    "initial_condition" => """
        template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const""",
    "initial_condition_variables" => """
        template <typename Vector> void initial_condition_variables(Vector &values) const""",
    "flux" => """
        template <typename NT, typename Solution>
          void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux,
                    const Point<dim> &pos, const Solution &sol) const""",
    "diffusion_flux" => """
        template <typename NT, typename Solution>
          void diffusion_flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux,
                              const Point<dim> &pos, const Solution &sol) const""",
    "source" => """
        template <typename NT, typename Solution>
          void source(std::array<NT, Components::count_fe_functions(0)> &source,
                      const Point<dim> &pos, const Solution &sol) const""",
    "mass" => """
        template <typename NT, typename Solution, typename Vector>
          void mass(std::array<NT, Components::count_fe_functions(0)> &mass, const Point<dim> &pos,
                    const Solution &sol, const Vector &dt_sol) const""",
    # NOTE the sign: the assembler takes A_dot = -residual, so a decaying
    # variable is written with a *positive* residual. Getting this backwards
    # runs cleanly and gives the wrong answer.
    "dt_variables" => """
        template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &sol) const""",
    "extract" => """
        template <typename NT, typename Solution>
          void extract(std::array<NT, Components::count_extractors()> &extractors, const Point<dim> &pos,
                       const Solution &sol) const""",
    "readouts" => """
        template <int _dim, typename DataOut, typename Solution>
          void readouts(DataOut &output, const Point<_dim> &pos, const Solution &sol) const""",
    "set_time" => "void set_time(double t_)",
    "cell_indicator" => """
        template <int _dim, typename NumberType, typename Solution>
          void cell_indicator(NumberType &indicator, const Point<_dim> &pos, const Solution &sol) const""",
    "face_indicator" => """
        template <int _dim, typename NumberType, typename Solutions_s, typename Solutions_n>
          void face_indicator(std::array<NumberType, 2> &indicator, const Tensor<1, _dim> &normal,
                              const Point<_dim> &pos, const Solutions_s &sol_s,
                              const Solutions_n &sol_n) const""",
    "wave_speed_blocks" => """
        template <size_t n_fe_functions> void wave_speed_blocks(std::array<int, n_fe_functions> &blocks) const""",
    "EoM" => """
        template <int _dim, typename Vector> std::array<double, _dim> EoM(const Point<_dim> &x, const Vector &u) const""",
)
