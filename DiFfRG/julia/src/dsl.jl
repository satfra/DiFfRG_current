"""
Translating a restricted subset of Julia into C++ member bodies.

The point of the restriction is that the generated code has to stay generic in
its number type: the same body is instantiated with `double`, with forward-AD
types, and on device. So the subset is arithmetic over a whitelist of functions
and nothing else -- no branching on values, no allocation, no Julia library
calls that have no C++ counterpart. Anything outside it is a Julia-level error
naming the offending line, rather than a template error from the compiler.

The head mirrors the C++ signature instead of hiding it, which is what lets the
bundle entries a body reads be checked against what the chosen assembler
actually provides.

**Indices are 1-based**, as everywhere else in Julia: `sol.fe_derivatives[:u][1]`
is the first spatial direction and becomes `[0]` in the emitted C++.
"""

# Functions that exist on both sides with the same meaning and are safe for
# every number type the body is instantiated with.
const _DSL_FUNCTIONS = Dict{Symbol,String}(
    :sqrt => "sqrt", :exp => "exp", :log => "log", :abs => "abs",
    :sin => "sin", :cos => "cos", :tan => "tan",
    :sinh => "sinh", :cosh => "cosh", :tanh => "tanh",
    :atan => "atan", :asin => "asin", :acos => "acos",
    :min => "min", :max => "max",
    :coth => "coth", :sech => "sech", :csch => "csch",
    :expm1 => "expm1", :log1p => "log1p",
)

# Threshold functions and regulators, which take an explicit level as a template
# parameter rather than an argument.
const _DSL_TEMPLATED = Dict{Symbol,String}(
    :lB => "tF::lB", :lF => "tF::lF", :lFb => "tF::lFb",
    :dlB => "tF::dlB", :dlF => "tF::dlF",
)

const _DSL_CONSTANTS = Dict{Symbol,String}(:pi => "M_PI", :π => "M_PI", :ℯ => "M_E")

# The bundle entries and the accessor each one is indexed with. A model names
# its components, and the descriptor turns the name into a compile-time offset.
const _BUNDLE_INDEX = Dict(
    "fe_functions" => "idxf", "fe_derivatives" => "idxf", "fe_hessians" => "idxf",
    "fe_third_derivatives" => "idxf", "variables" => "idxv", "extractors" => "idxe",
)

# How many indices past the component each entry takes: a value, a gradient, a
# hessian.
const _BUNDLE_RANK = Dict(
    "fe_functions" => 0, "fe_derivatives" => 1, "fe_hessians" => 2,
    "fe_third_derivatives" => 3, "variables" => 0, "extractors" => 0,
)

"""
    DSLContext

What the transpiler needs to know to check a body: the model's component names,
which bundle entries the assembler provides, and what the output argument is
called in C++.
"""
mutable struct DSLContext
    member::String
    output::String
    output_rank::Int
    components::Dict{String,Vector{String}}
    available::Vector{String}
    argnames::Dict{Symbol,String}
    used_bundle::Set{String}
    locals::Set{Symbol}
end

struct DSLError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DSLError) = print(io, "DiFfRG DSL: ", e.msg)

_dsl_error(msg) = throw(DSLError(msg))

# The output argument of each member, and how many indices follow the component.
const _MEMBER_OUTPUT = Dict(
    "flux" => ("flux", 1), "diffusion_flux" => ("flux", 1),
    "source" => ("source", 0), "mass" => ("mass", 0),
    "initial_condition" => ("values", 0),
    "initial_condition_variables" => ("values", 0),
    "extract" => ("extractors", 0), "dt_variables" => ("residual", 0),
)

"""
    transpile_body(member, head, body; components, available) -> (code, used)

Translate one member body, returning the C++ and the bundle entries it read.
"""
function transpile_body(member::AbstractString, head, body;
                        components::Dict{String,Vector{String}},
                        available::Vector{String})
    haskey(_MEMBER_OUTPUT, member) ||
        _dsl_error("no Julia-expression form for member '$member'; supply it as cxx\"...\"")
    out, rank = _MEMBER_OUTPUT[member]
    ctx = DSLContext(String(member), out, rank, components, available,
                     Dict{Symbol,String}(), Set{String}(), Set{Symbol}())

    # The head names the arguments: (F, x, sol) binds the output, the point and
    # the bundle to whatever the user wants to call them.
    for (i, name) in enumerate(head)
        ctx.argnames[name] = i == 1 ? out : (i == 2 ? "pos" : "sol")
    end
    length(head) >= 1 || _dsl_error("the head must name at least the output argument")

    lines = String[]
    _emit_block(body, ctx, lines)

    # The `get<>` bindings go in front, one per entry actually read.
    prelude = String[]
    for entry in sort(collect(ctx.used_bundle))
        push!(prelude, "const auto &$entry = get<\"$entry\">(sol);")
    end
    return join(vcat(prelude, lines), "\n"), ctx.used_bundle
end

function _emit_block(expr, ctx, lines)
    if expr isa Expr && expr.head === :block
        for st in expr.args
            st isa LineNumberNode && continue
            _emit_statement(st, ctx, lines)
        end
    else
        _emit_statement(expr, ctx, lines)
    end
end

function _emit_statement(st, ctx, lines)
    if st isa Expr && st.head === :(=)
        lhs, rhs = st.args
        code = _expr(rhs, ctx)
        if lhs isa Symbol
            # A new name is a const binding; `auto` keeps it generic in the
            # number type, which is the whole point of the restriction.
            push!(ctx.locals, lhs)
            push!(lines, "const auto $lhs = $code;")
        else
            push!(lines, "$(_expr(lhs, ctx)) = $code;")
        end
    elseif st isa Expr && st.head === :if
        _dsl_error("`if` branches on a value, which does not survive AD or a " *
                   "device instantiation; use ifelse(cond, a, b)")
    elseif st isa Expr && st.head in (:for, :while)
        _dsl_error("`$(st.head)` is not available in an expression body; " *
                   "supply the member as cxx\"...\" if it needs one")
    elseif st isa Expr && st.head === :return
        push!(lines, "return $(_expr(only(st.args), ctx));")
    else
        push!(lines, _expr(st, ctx) * ";")
    end
    return nothing
end

# ------------------------------------------------------------------ expressions

_expr(x::Bool, ctx) = x ? "true" : "false"
# Every numeric literal is emitted as a double. An integer would make `1 / 2`
# integer division in C++ -- zero, silently -- where Julia gives 0.5. The places
# where C++ genuinely needs an integer (a `powr` exponent, a template parameter,
# an array index) bypass this and take the literal directly.
_expr(x::Integer, ctx) = string(x, ".")
_expr(x::AbstractFloat, ctx) = isinteger(x) ? string(Int(x), ".") : repr(x)

function _expr(s::Symbol, ctx)
    haskey(_DSL_CONSTANTS, s) && return _DSL_CONSTANTS[s]
    s in ctx.locals && return String(s)
    haskey(ctx.argnames, s) && return ctx.argnames[s]
    # Anything else is a name from the surrounding C++: a parameter, a member
    # function, a constant. It is passed through and the compiler checks it.
    return String(s)
end

_expr(x::QuoteNode, ctx) = _expr(x.value, ctx)
_expr(x::String, ctx) = "\"$x\""

function _expr(e::Expr, ctx)
    e.head === :call && return _call(e, ctx)
    e.head === :ref && return _index(e, ctx)
    e.head === :. && return _field(e, ctx)
    e.head === :(::) && _dsl_error("type annotations fix the number type and " *
                                   "break AD; drop the `::$(e.args[2])`")
    e.head === :block && return _expr(only(filter(a -> !(a isa LineNumberNode), e.args)), ctx)
    e.head === :&& && return "($(_expr(e.args[1], ctx)) && $(_expr(e.args[2], ctx)))"
    e.head === :|| && return "($(_expr(e.args[1], ctx)) || $(_expr(e.args[2], ctx)))"
    e.head === :if && _dsl_error("`if` branches on a value; use ifelse(cond, a, b)")
    _dsl_error("unsupported Julia construct `$(e.head)` in $(ctx.member)")
end

const _BINARY = Dict(:+ => "+", :- => "-", :* => "*", :/ => "/",
                     :(==) => "==", :!= => "!=", :< => "<", :<= => "<=",
                     :> => ">", :>= => ">=")

function _call(e::Expr, ctx)
    f = e.args[1]
    args = e.args[2:end]

    if f === :^
        return _power(args, ctx)
    elseif f === :ifelse
        length(args) == 3 || _dsl_error("ifelse takes three arguments")
        return "($(_expr(args[1], ctx)) ? $(_expr(args[2], ctx)) : $(_expr(args[3], ctx)))"
    elseif f isa Symbol && haskey(_BINARY, f)
        length(args) == 1 && f === :- && return "(-$(_expr(args[1], ctx)))"
        length(args) == 1 && f === :+ && return _expr(args[1], ctx)
        return "(" * join((_expr(a, ctx) for a in args), " $(_BINARY[f]) ") * ")"
    elseif f isa Symbol && haskey(_DSL_FUNCTIONS, f)
        return _DSL_FUNCTIONS[f] * "(" * join((_expr(a, ctx) for a in args), ", ") * ")"
    elseif f isa Expr && f.head === :curly
        # lB{0}(m2, k, T, d) -- a level given as a template parameter.
        name = f.args[1]
        haskey(_DSL_TEMPLATED, name) ||
            _dsl_error("`$name` takes no template parameter")
        params = join((p isa Integer ? string(p) : _expr(p, ctx) for p in f.args[2:end]), ", ")
        return "$(_DSL_TEMPLATED[name])<$params>(" *
               join((_expr(a, ctx) for a in args), ", ") * ")"
    elseif f isa Symbol && haskey(_DSL_TEMPLATED, f)
        _dsl_error("`$f` needs its level as a template parameter, as in $f{0}(...)")
    elseif f isa Symbol
        # A helper the model defines itself, or something from the DiFfRG
        # headers. Passed through for the compiler to resolve.
        return String(f) * "(" * join((_expr(a, ctx) for a in args), ", ") * ")"
    end
    _dsl_error("cannot call `$f` from an expression body")
end

# An integer power becomes powr<n>, which DiFfRG defines for every number type
# it uses and which avoids a call to pow.
function _power(args, ctx)
    base, exponent = args
    if exponent isa Integer
        return "powr<$exponent>($(_expr(base, ctx)))"
    end
    return "pow($(_expr(base, ctx)), $(_expr(exponent, ctx)))"
end

function _field(e::Expr, ctx)
    obj, field = e.args
    name = field isa QuoteNode ? field.value : field
    if obj isa Symbol && get(ctx.argnames, obj, "") == "sol"
        entry = String(name)
        haskey(_BUNDLE_INDEX, entry) ||
            _dsl_error("`$entry` is not a solution bundle entry; available here: " *
                       join(ctx.available, ", "))
        entry in ctx.available ||
            _dsl_error("the chosen assembler does not provide `$entry` " *
                       "(it provides " * join(ctx.available, ", ") * ")")
        push!(ctx.used_bundle, entry)
        return entry
    end
    return "$(_expr(obj, ctx)).$name"
end

function _index(e::Expr, ctx)
    obj = e.args[1]
    idx = e.args[2:end]

    # sol.fe_derivatives[:u][1]
    if obj isa Expr && obj.head === :.
        entry = _field(obj, ctx)
        length(idx) == 1 || _dsl_error("index a bundle entry by component name, " *
                                       "as in sol.$entry[:u]")
        comp = _component_name(idx[1], entry, ctx)
        return "$entry[$(_BUNDLE_INDEX[entry])(\"$comp\")]"
    end

    # A further index on an already-indexed bundle entry, or on the output.
    inner = _expr(obj, ctx)
    parts = String[]
    for i in idx
        push!(parts, "[" * _zero_based(i, ctx) * "]")
    end
    return inner * join(parts)
end

# The point and every tensor index are 1-based in the DSL, as they are
# everywhere else in Julia, and shift by one on the way out.
function _zero_based(i, ctx)
    if i isa Integer
        i >= 1 || _dsl_error("index $i is out of range; the DSL is 1-based")
        return string(i - 1)
    elseif i isa QuoteNode || i isa Symbol
        # A component name used directly on the output array.
        return "" * _index_component(i, ctx)
    end
    return "(" * _expr(i, ctx) * " - 1)"
end

function _index_component(i, ctx)
    name = i isa QuoteNode ? String(i.value) : String(i)
    for (kind, names) in ctx.components
        if name in names
            return "$(_BUNDLE_INDEX[kind])(\"$name\")"
        end
    end
    _dsl_error("`$name` is not a component of this model; it has " *
               join(sort(vcat(values(ctx.components)...)), ", "))
end

function _component_name(i, entry, ctx)
    name = i isa QuoteNode ? String(i.value) : (i isa Symbol ? String(i) : nothing)
    name === nothing && _dsl_error("index `$entry` by a component name, as in sol.$entry[:u]")
    kind = _BUNDLE_INDEX[entry] == "idxf" ? "fe_functions" :
           (_BUNDLE_INDEX[entry] == "idxv" ? "variables" : "extractors")
    names = get(ctx.components, kind, String[])
    name in names || _dsl_error("`$name` is not one of this model's $kind " *
                                "(" * join(names, ", ") * ")")
    return name
end

# --------------------------------------------------------------------- front end

"""
    JuliaBody

A member body written as a Julia expression, translated at generation time --
when the discretization, and therefore the bundle, is known.
"""
struct JuliaBody
    head::Vector{Symbol}
    expr::Any
    source::String
end

"""
    @julia (F, x, sol) -> begin ... end

A member body as a Julia expression.

The head names the output, the point and the solution bundle; the body reads the
bundle by name and assigns into the output:

Parenthesise it in a keyword argument, or the macro swallows the rest of the
call:

```julia
flux = @julia((F, x, sol) -> begin
    u   = sol.fe_functions[:u]
    du  = sol.fe_derivatives[:u][1]
    F[:u][1] = 0.5u^2 + x[1] * du
end)
```

Indices are 1-based, so `[1]` is the first spatial direction and becomes `[0]`.
`u^2` becomes `powr<2>(u)`. Reading a bundle entry the chosen assembler does not
provide is an error here rather than in the compiler.
"""
macro julia(lambda)
    lambda isa Expr && lambda.head === :-> || error("""
        @julia takes a lambda. In a keyword argument it must be parenthesised,
        because a macro otherwise swallows the rest of the call:

            flux = @julia((F, x, sol) -> begin
                F[:u][1] = 0.5 * sol.fe_functions[:u]^2
            end),
        """)
    head = lambda.args[1]
    names = head isa Symbol ? [head] :
            Symbol[a for a in head.args if a isa Symbol]
    return JuliaBody(names, lambda.args[2], string(lambda))
end

"""
    render_body(body, member; components, available) -> String

The C++ for a member body, whichever form it was written in.
"""
render_body(b::CxxBody, member; kwargs...) = b.code
render_body(s::AbstractString, member; kwargs...) = String(s)
function render_body(b::JuliaBody, member; components, available)
    code, _ = transpile_body(member, b.head, b.expr;
                             components = components, available = available)
    return code
end
