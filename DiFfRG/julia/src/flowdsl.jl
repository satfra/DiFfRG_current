"""
The graph form of `dt_variables`.

`dt_variables` is statement-shaped rather than expression-shaped: it guards, it
updates interpolators, it launches map kernels, and it solves fixed points. What
orders those statements is data dependency, and `DeferredMaps` -- the mechanism
that lets independent kernels overlap -- is a contract about exactly that:

    Use it around a run of *independent* flows [...] the caller promises not to
    read any `dest` until the scope ends.        (map_completion.hh)

So statements inside a scope are reordering-neutral by construction, and the
only real constraint is read-after-write between blocks. Writing that as a graph
lets the generator own three invariants the C++ maintains by hand:

  * The batches are *derived* from the written data flow. Choosing them by hand
    is an optimization whose failure mode, per that same header, "is not a
    compile error but wrong numbers".
  * `DeferredMaps` is a bare RAII flag whose destructor flushes, so nesting one
    inside another flushes early and silently. Emitting exactly one scope per
    derived batch makes that unrepresentable.
  * The abort guard has to run before any kernel launch -- the assembler fences
    only after `dt_variables` returns, so throwing past an in-flight kernel is
    what the placement is defending against. Here it is emitted first by
    construction rather than by a comment.
"""

"""
    FlowBody

A `dt_variables` body written as a dependency graph.
"""
struct FlowBody
    statements::Vector{Any}
    source::String
end

"""
    @variables_flow(begin ... end)

A `dt_variables` body, as a graph of guards, updates and map launches.

```julia
dt_variables = @variables_flow(begin
    @guard has_run_away(m2A, prm.m2A) => FlowAbort(t)
    @update ZA3, ZAcbc, ZA, Zc
    args = @tie(k, ZA3, ZAcbc, dtZc, Zc, dtZA, ZA)

    @fixedpoint (dtZA, dtZc) size=p_grid_size tol=prm.eta_tol maxiter=prm.eta_iter_max begin
        r[:ZA] = flows.ZA.map(coordinates1D, args)
        r[:Zc] = flows.Zc.map(coordinates1D, args)
        @update dtZA from r[:ZA]
        @update dtZc from r[:Zc]
    end

    r[:ZA3]   = flows.ZA3.map(coordinates3D, args)
    r[:ZAcbc] = flows.ZAcbc.map(coordinates3D, args)
end)
```

Consecutive map launches that do not read each other's destination become one
`DeferredMaps` scope; anything that reads a destination closes the batch first.

Note the residual convention: the assembler takes `A_dot = -residual`, so a
decaying variable is written with a positive residual.
"""
macro variables_flow(block)
    block isa Expr && block.head === :block ||
        error("@variables_flow takes a begin ... end block, parenthesised in a " *
              "keyword argument: @variables_flow(begin ... end)")
    return FlowBody(Any[s for s in block.args if !(s isa LineNumberNode)], string(block))
end

# --------------------------------------------------------------------- analysis

"""
One statement of a flow body, classified so that batching can be derived.

`writes` is the destination a map lands in; `reads` is what the statement needs
to be correct, which is what forces a batch to close.
"""
struct FlowStatement
    kind::Symbol            # :guard, :update, :tie, :map, :fixedpoint, :plain
    code::String            # rendered C++, for everything but :fixedpoint
    writes::Vector{String}
    reads::Vector{String}
    inner::Vector{Any}      # :fixedpoint only
    meta::Dict{Symbol,Any}
end

FlowStatement(kind, code; writes = String[], reads = String[],
              inner = Any[], meta = Dict{Symbol,Any}()) =
    FlowStatement(kind, code, writes, reads, inner, meta)

struct FlowContext
    variables::Vector{String}
    flows_member::String
    residual::String
end

"""
    classify(statement, ctx) -> FlowStatement
"""
function classify(st, ctx::FlowContext)
    if st isa Expr && st.head === :macrocall
        return _classify_macro(st, ctx)
    elseif st isa Expr && st.head === :(=)
        lhs, rhs = st.args
        # `r[:X] = flows.Y.map(coords, args)` is a launch, not an assignment:
        # map() writes through a pointer to the destination slot.
        if _is_map_call(rhs)
            dest = _residual_slot(lhs, ctx)
            kernel, coords, args = _map_parts(rhs, ctx)
            code = "$(ctx.flows_member).$kernel.map(&$(ctx.residual)[idxv(\"$dest\")], " *
                   "$coords, $args);"
            return FlowStatement(:map, code; writes = [dest], reads = _names_in(rhs))
        end
        if rhs isa Expr && rhs.head === :macrocall &&
           _macro_name(rhs) === Symbol("@tie")
            name = lhs isa Symbol ? String(lhs) : _dsl_error("@tie must be bound to a name")
            args = join((_flow_expr(a, ctx) for a in _macro_args(rhs)), ", ")
            return FlowStatement(:tie, "const auto $name = device::tie($args);";
                                 writes = [name], reads = _names_in(rhs))
        end
        lhs isa Symbol && return FlowStatement(:plain,
            "const auto $lhs = $(_flow_expr(rhs, ctx));";
            writes = [String(lhs)], reads = _names_in(rhs))
        return FlowStatement(:plain, "$(_flow_expr(lhs, ctx)) = $(_flow_expr(rhs, ctx));";
                             reads = _names_in(rhs))
    end
    return FlowStatement(:plain, _flow_expr(st, ctx) * ";"; reads = _names_in(st))
end

_macro_name(e::Expr) = e.args[1] isa Symbol ? e.args[1] :
                       (e.args[1] isa GlobalRef ? e.args[1].name : Symbol(string(e.args[1])))
_macro_args(e::Expr) = [a for a in e.args[2:end] if !(a isa LineNumberNode)]

function _classify_macro(st::Expr, ctx::FlowContext)
    name = _macro_name(st)
    args = _macro_args(st)

    if name === Symbol("@guard")
        length(args) == 1 && args[1] isa Expr && args[1].head === :call &&
            args[1].args[1] === :(=>) ||
            _dsl_error("@guard reads `@guard <condition> => <exception>`")
        cond, exc = args[1].args[2], args[1].args[3]
        code = "if ($(_flow_expr(cond, ctx))) throw $(_flow_expr(exc, ctx));"
        return FlowStatement(:guard, code; reads = _names_in(args[1]))

    elseif name === Symbol("@update")
        # `@update a, b` refreshes interpolators from the incoming state;
        # `@update x from r[:X]` refreshes one from a destination just landed.
        # The second form reaches the macro as three arguments, `from` among
        # them, since it is not Julia infix syntax.
        if length(args) == 3 && args[2] === :from
            target = args[1]
            slot = _residual_slot(args[3], ctx)
            return FlowStatement(:update,
                "$(target).update(&$(ctx.residual)[idxv(\"$slot\")]);";
                writes = [String(target)], reads = [slot])
        end
        names = Symbol[]
        for a in args
            a isa Symbol && push!(names, a)
            a isa Expr && a.head === :tuple && append!(names, a.args)
        end
        isempty(names) && _dsl_error("@update needs at least one name")
        lines = ["$(n).update(&variables.data()[idxv(\"$(n)\")]);" for n in names]
        return FlowStatement(:update, join(lines, "\n");
                             writes = String.(names), reads = ["variables"])

    elseif name === Symbol("@reduce")
        _dsl_error("@reduce is only meaningful inside another expression")

    elseif name === Symbol("@fixedpoint")
        return _classify_fixedpoint(args, ctx)
    end
    _dsl_error("unknown macro `$name` in a flow body")
end

function _classify_fixedpoint(args, ctx::FlowContext)
    length(args) >= 2 || _dsl_error("@fixedpoint reads " *
        "`@fixedpoint (a, b) size=N tol=... maxiter=... begin ... end`")
    targets = args[1] isa Expr && args[1].head === :tuple ?
              Symbol[a for a in args[1].args] : Symbol[args[1]]
    body = args[end]
    body isa Expr && body.head === :block ||
        _dsl_error("@fixedpoint needs a begin ... end body")

    meta = Dict{Symbol,Any}(:size => nothing, :tol => nothing, :maxiter => nothing)
    for kw in args[2:end-1]
        kw isa Expr && kw.head === :(=) ||
            _dsl_error("@fixedpoint options are size=, tol= and maxiter=")
        key = kw.args[1]
        haskey(meta, key) || _dsl_error("unknown @fixedpoint option `$key`")
        meta[key] = kw.args[2]
    end
    for key in (:size, :tol, :maxiter)
        meta[key] === nothing && _dsl_error("@fixedpoint needs $key=")
    end
    meta[:targets] = targets
    inner = Any[s for s in body.args if !(s isa LineNumberNode)]
    return FlowStatement(:fixedpoint, ""; inner = inner, meta = meta,
                         writes = String.(targets))
end

_is_map_call(e) = e isa Expr && e.head === :call && e.args[1] isa Expr &&
                  e.args[1].head === :. && e.args[1].args[2] isa QuoteNode &&
                  e.args[1].args[2].value === :map

function _map_parts(e::Expr, ctx)
    target = e.args[1].args[1]          # flows.ZA
    target isa Expr && target.head === :. ||
        _dsl_error("a map launch reads `flows.<kernel>.map(coordinates, args)`")
    kernel = String(target.args[2].value)
    length(e.args) == 3 ||
        _dsl_error("`map` takes the coordinates and the argument pack")
    return kernel, _flow_expr(e.args[2], ctx), _flow_expr(e.args[3], ctx)
end

function _residual_slot(lhs, ctx)
    lhs isa Expr && lhs.head === :ref || _dsl_error("a map lands in r[:name]")
    idx = lhs.args[2]
    name = idx isa QuoteNode ? String(idx.value) : String(idx)
    name in ctx.variables ||
        _dsl_error("`$name` is not one of this model's variables " *
                   "(" * join(ctx.variables, ", ") * ")")
    return name
end

# Every symbol a statement mentions. Used only to decide when a batch has to
# close, so over-approximating is safe and under-approximating is not.
function _names_in(e)
    out = String[]
    _walk_names(e, out)
    return out
end
function _walk_names(e, out)
    if e isa Symbol
        push!(out, String(e))
    elseif e isa QuoteNode
        push!(out, String(e.value))
    elseif e isa Expr
        for a in e.args
            _walk_names(a, out)
        end
    end
    return nothing
end

# --------------------------------------------------------------------- emission

# Expressions inside a flow body are host code, not kernel code, so they are not
# put through the number-type-generic subset. Only the pieces the flow DSL owns
# -- residual slots, reductions -- are rewritten.
function _flow_expr(e, ctx::FlowContext)
    if e isa Expr && e.head === :macrocall && _macro_name(e) === Symbol("@reduce")
        args = _macro_args(e)
        length(args) == 2 || _dsl_error("@reduce reads `@reduce(op, name)`")
        op, name = args
        op in (:min, :max) || _dsl_error("@reduce supports min and max")
        init = op === :min ? "std::numeric_limits<double>::infinity()" :
                             "-std::numeric_limits<double>::infinity()"
        return "[&]{ double acc = $init; " *
               "for (uint i = 0; i < $(name)_size; ++i) acc = std::$op(acc, $name[i]); " *
               "return acc; }()"
    elseif e isa Expr && e.head === :ref && e.args[1] === :r
        return "$(ctx.residual)[idxv(\"$(_residual_slot(e, ctx))\")]"
    elseif e isa Expr
        return string(_rewrite(e, ctx))
    end
    return string(e)
end

# Rewrites r[:X] anywhere inside a larger expression, and leaves the rest as the
# host code it is.
function _rewrite(e, ctx)
    e isa Expr || return e
    if e.head === :ref && e.args[1] === :r
        return Symbol("$(ctx.residual)[idxv(\"$(_residual_slot(e, ctx))\")]")
    end
    return Expr(e.head, (_rewrite(a, ctx) for a in e.args)...)
end

"""
    derive_batches(statements) -> Vector{Vector{Int}}

Group consecutive map launches into `DeferredMaps` batches.

A batch extends while the next launch neither reads nor rewrites a destination
already written in it. Anything that is not a launch closes the batch, because
it may read one -- over-grouping is what silently produces wrong numbers, so the
grouping errs towards closing early.
"""
function derive_batches(statements::Vector{FlowStatement})
    batches = Vector{Vector{Int}}()
    current = Int[]
    written = Set{String}()
    for (i, st) in enumerate(statements)
        if st.kind === :map && isempty(intersect(st.reads, written)) &&
           isempty(intersect(st.writes, written))
            push!(current, i)
            union!(written, st.writes)
        else
            isempty(current) || push!(batches, current)
            current = st.kind === :map ? [i] : Int[]
            written = st.kind === :map ? Set(st.writes) : Set{String}()
        end
    end
    isempty(current) || push!(batches, current)
    return batches
end

"""
    emit_flow(statements, ctx; indent) -> String

Render a flow body, with the batching derived rather than written.
"""
function emit_flow(statements::Vector{FlowStatement}, ctx::FlowContext; indent = "")
    batches = derive_batches(statements)
    in_batch = Dict{Int,Vector{Int}}()
    for b in batches
        for i in b
            in_batch[i] = b
        end
    end

    lines = String[]
    emitted = Set{Int}()
    for (i, st) in enumerate(statements)
        i in emitted && continue
        if st.kind === :map
            batch = in_batch[i]
            if length(batch) == 1
                # Deferral buys nothing for a lone launch, and outside a scope
                # map() keeps its original contract: the result is in `dest`
                # when it returns.
                push!(lines, indent * st.code)
            else
                push!(lines, indent * "{ // independent launches: one batch")
                push!(lines, indent * "  DeferredMaps defer;")
                for j in batch
                    push!(lines, indent * "  " * statements[j].code)
                end
                push!(lines, indent * "} // all land here")
            end
            union!(emitted, batch)
        elseif st.kind === :fixedpoint
            append!(lines, _emit_fixedpoint(st, ctx, indent))
            push!(emitted, i)
        else
            for l in split(st.code, '\n')
                push!(lines, indent * l)
            end
            push!(emitted, i)
        end
    end
    return join(lines, "\n")
end

function _emit_fixedpoint(st::FlowStatement, ctx::FlowContext, indent)
    targets = st.meta[:targets]
    size = _flow_expr(st.meta[:size], ctx)
    tol = _flow_expr(st.meta[:tol], ctx)
    maxiter = _flow_expr(st.meta[:maxiter], ctx)
    inner = FlowStatement[classify(s, ctx) for s in st.inner]

    L = String[]
    push!(L, indent * "// Fixed point in " * join(string.(targets), ", ") *
             ": each appears inside its own right-hand side.")
    for t in targets
        push!(L, indent * "std::vector<double> old_$t($size);")
    end
    push!(L, indent * "bool converged = false;")
    push!(L, indent * "uint n_iter = 0;")
    push!(L, indent * "while (!converged) {")
    push!(L, indent * "  for (uint i = 0; i < $size; ++i) {")
    for t in targets
        push!(L, indent * "    old_$t[i] = $t[i];")
    end
    push!(L, indent * "  }")
    push!(L, emit_flow(inner, ctx; indent = indent * "  "))
    push!(L, indent * "  double dist = 0.;")
    push!(L, indent * "  for (uint i = 0; i < $size; ++i) {")
    for t in targets
        push!(L, indent * "    dist = std::max(dist, std::abs($t[i] - old_$t[i]) / " *
                          "std::abs($t[i]));")
    end
    push!(L, indent * "  }")
    push!(L, indent * "  ++n_iter;")
    push!(L, indent * "  if (dist < $tol || n_iter >= $maxiter) converged = true;")
    push!(L, indent * "}")
    return L
end

"""
    render_flow(body, variables, flows_member) -> String

The C++ for a `@variables_flow` body.
"""
function render_flow(body::FlowBody, variables::Vector{String}, flows_member::AbstractString)
    ctx = FlowContext(variables, String(flows_member), "residual")
    statements = FlowStatement[classify(s, ctx) for s in body.statements]

    # The guard reads the incoming state and must precede every launch: the
    # assembler fences only after dt_variables returns, so throwing past an
    # in-flight kernel is what its placement defends against. Here that is
    # structural rather than a comment.
    guards = [s for s in statements if s.kind === :guard]
    rest = [s for s in statements if s.kind !== :guard]

    first_guard = findfirst(s -> s.kind === :guard, statements)
    first_launch = findfirst(s -> s.kind in (:map, :fixedpoint), statements)
    if first_guard !== nothing && first_launch !== nothing && first_guard > first_launch
        @warn "a @guard was written after a kernel launch; it is emitted before " *
              "every launch, because the assembler fences only after dt_variables returns"
    end

    prelude = "const auto &variables = get<\"variables\">(sol);"
    return prelude * "\n" * emit_flow(vcat(guards, rest), ctx)
end
