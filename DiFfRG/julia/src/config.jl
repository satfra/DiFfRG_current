"""
Parameter files and command-line overrides.

DiFfRG apps are configured by a nested `parameter.json` (or `.toml`), and every
key in it can be overridden on the command line by JSON pointer. That is what
makes a parameter scan free: the binary is built once and each point is a
process with a few extra flags, so nothing is recompiled per point.
"""

"""
    kvpairs(params)

Key/value pairs of a parameter collection, accepting a `NamedTuple`, a `Dict` or
any iterable of pairs.

`pairs` alone will not do: on a tuple of pairs it yields index => pair, which
silently turns `("/physical/T" => 0.1,)` into a parameter named `1`.
"""
kvpairs(params::NamedTuple) = pairs(params)
kvpairs(params::AbstractDict) = pairs(params)
kvpairs(params) = params

"""
    pointer(key) -> String

The JSON pointer for a scan or override key.

A bare name is taken to live in `/physical`, which is where a model's own
parameters go; anything that already starts with `/` is passed through, so any
part of the tree stays reachable.

```julia
pointer(:T)                  # "/physical/T"
pointer("/discretization/fe_order")
```
"""
function pointer(key)
    s = String(key)
    return startswith(s, "/") ? s : "/physical/" * s
end

# The flag is chosen by type, so `T = 0.1` and `Nc = 3` reach the app as the
# double and the integer they are. Bool is checked first: it is an Integer in
# Julia, but `-si /x=true` is not something the C++ side will parse.
function override_flags(key, value)
    ptr = pointer(key)
    if value isa Bool
        return ["-sb", "$ptr=$(value ? "true" : "false")"]
    elseif value isa Integer
        return ["-si", "$ptr=$value"]
    elseif value isa Real
        return ["-sd", "$ptr=$value"]
    elseif value isa AbstractString || value isa Symbol
        return ["-ss", "$ptr=$value"]
    end
    throw(ArgumentError(
        "cannot pass $(repr(key)) = $(repr(value)) ($(typeof(value))) to a DiFfRG app; " *
        "overrides must be Bool, Integer, Real, String or Symbol"))
end

"""
    override_flags(params) -> Vector{String}

Command-line flags setting every entry of `params`, which may be a `NamedTuple`,
a `Dict` or any collection of pairs.
"""
function override_flags(params)
    flags = String[]
    for (k, v) in kvpairs(params)
        append!(flags, override_flags(k, v))
    end
    return flags
end

"""
    nest(params) -> Dict

Expand flat pointer-keyed parameters into the nested tree a parameter file
holds, so the same `(T = 0.1,)` spelling works whether it is written to a file
or passed on the command line.
"""
function nest(params)
    root = Dict{String,Any}()
    for (k, v) in kvpairs(params)
        parts = split(pointer(k), '/'; keepempty = false)
        node = root
        for p in parts[1:end-1]
            child = get!(node, String(p), Dict{String,Any}())
            child isa AbstractDict ||
                throw(ArgumentError("parameter path /$(join(parts, '/')) runs through " *
                                    "/$(p), which is already a value"))
            node = child
        end
        node[String(parts[end])] = v isa Symbol ? String(v) : v
    end
    return root
end

"""
    merge_tree(base, over) -> Dict

`base` with `over` merged into it, recursing into nested sections rather than
replacing them, so overriding one key of `/physical` keeps the rest.
"""
function merge_tree(base::AbstractDict, over::AbstractDict)
    out = Dict{String,Any}(String(k) => v for (k, v) in base)
    for (k, v) in over
        key = String(k)
        prev = get(out, key, nothing)
        out[key] = (prev isa AbstractDict && v isa AbstractDict) ? merge_tree(prev, v) : v
    end
    return out
end

"""
    read_config(path) -> Dict

Read a parameter file. TOML is used for a `.toml`/`.tml` extension and JSON
otherwise, matching how `ConfigTree` picks a parser.
"""
function read_config(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext in (".toml", ".tml")
        return TOML.parsefile(path)
    end
    return JSON3.read(read(path, String), Dict{String,Any})
end

"""
    write_config(path, tree)

Write a parameter tree, in the format implied by the file extension.
"""
function write_config(path::AbstractString, tree::AbstractDict)
    mkpath(dirname(abspath(path)))
    ext = lowercase(splitext(path)[2])
    if ext in (".toml", ".tml")
        open(path, "w") do io
            TOML.print(io, tree; sorted = true)
        end
    else
        open(path, "w") do io
            JSON3.pretty(io, tree)
        end
    end
    return path
end
