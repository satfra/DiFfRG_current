"""
The scheme table: which C++ types a discretization and timestepper are spelled
with, which headers they need, and which combinations are legal.

The table itself is `frontend/schemes.toml` in the DiFfRG install, not in this
package, so that adding a discretization to the library updates this frontend in
the same commit.
"""

"""
    schemes_path(install) -> String

Where the scheme table lives.

Installed at `<prefix>/frontend/schemes.toml`. The source tree is also accepted,
so the package works against a checkout that has not been reinstalled yet.
"""
function schemes_path(install::DiFfRGInstall)
    installed = joinpath(install.prefix, "frontend", "schemes.toml")
    isfile(installed) && return installed
    # A development checkout: this file is DiFfRG/julia/src/schemes.jl, so the
    # table is two levels up.
    insource = normpath(joinpath(@__DIR__, "..", "..", "frontend", "schemes.toml"))
    isfile(insource) && return insource
    error("""
          No frontend/schemes.toml found. Looked in:
            $installed
            $insource
          It ships with DiFfRG; reinstall the library to pick it up.
          """)
end

"""
    load_schemes(install = find_install()) -> Dict

Read the scheme table.
"""
load_schemes(install::DiFfRGInstall = find_install()) = TOML.parsefile(schemes_path(install))

"""
    Scheme

A resolved choice of discretization and everything it implies: the concrete C++
type strings, the headers to include in order, and the bundle entries the model
may read.

Build one with [`resolve_scheme`](@ref).
"""
struct Scheme
    name::String
    main_shape::String
    mesh::String
    discretization::String
    assembler::String
    stepper::String
    flowing_variables::String
    flowing_variables_arg::String
    adaptivity::String
    adaptivity_construct::String
    headers::Vector{String}
    solution::Vector{String}
    takes_adaptor::Bool
end

function Base.show(io::IO, s::Scheme)
    print(io, "Scheme(", s.name, ", ", s.stepper, ")")
end

_fill(template, subs) = replace(template, ("{$k}" => v for (k, v) in subs)...)

"""
    resolve_scheme(; model, dim, discretization, stepper=..., ...) -> Scheme

Resolve a scheme into concrete C++ types, rejecting illegal combinations.

Checks that the table can actually express what was asked for: that the mesh
suits the discretization (LDG needs a serial triangulation), that the adaptivity
does (KT needs `NoAdaptivity`), and that a stepper which has no default for its
linear solver or precision was given one.
"""
function resolve_scheme(; model::AbstractString,
                        dim::Integer,
                        discretization::AbstractString,
                        stepper::AbstractString = "SUNDIALS_IDA",
                        mesh = nothing,
                        adaptivity = nothing,
                        linear_solver = nothing,
                        limiter = "MinMod",
                        reconstructor = "TVD",
                        wave_speed = "MaxEigenvalue",
                        prec = nothing,
                        install::DiFfRGInstall = find_install(),
                        table = nothing)
    tbl = table === nothing ? load_schemes(install) : table

    haskey(tbl["discretizations"], discretization) ||
        throw(ArgumentError("unknown discretization $(repr(discretization)); " *
                            "have $(join(sort(collect(keys(tbl["discretizations"]))), ", "))"))
    haskey(tbl["steppers"], stepper) ||
        throw(ArgumentError("unknown timestepper $(repr(stepper)); " *
                            "have $(join(sort(collect(keys(tbl["steppers"]))), ", "))"))

    disc = tbl["discretizations"][discretization]
    step = tbl["steppers"][stepper]
    headers = String[]
    append!(headers, String.(disc["headers"]))

    subs = Dict("dim" => string(dim), "model" => String(model))

    # -- mesh ---------------------------------------------------------------
    mesh_type = ""
    if disc["main_shape"] != "variables"
        mesh_name = mesh === nothing ? disc["default_mesh"] : String(mesh)
        haskey(tbl["meshes"], mesh_name) ||
            throw(ArgumentError("unknown mesh $(repr(mesh_name))"))
        entry = tbl["meshes"][mesh_name]
        req = get(disc, "mesh_requirement", nothing)
        if req == "serial" && entry["kind"] != "serial"
            throw(ArgumentError(
                "$discretization requires a serial triangulation, but $mesh_name is " *
                "$(entry["kind"]); use RectangularMeshSerial"))
        end
        mesh_type = _fill(entry["type"], subs)
        append!(headers, String.(entry["headers"]))
        subs["mesh"] = mesh_type
    end

    # -- finite-volume policies ---------------------------------------------
    if occursin("{reconstructor}", disc["assembler"])
        rec = tbl["reconstructors"][String(reconstructor)]
        if get(rec, "takes_limiter", false)
            lim = tbl["limiters"][String(limiter)]
            subs["limiter"] = lim["type"]
            append!(headers, String.(lim["headers"]))
        end
        subs["reconstructor"] = _fill(rec["type"], subs)
        append!(headers, String.(rec["headers"]))
        ws = tbl["wave_speeds"][String(wave_speed)]
        subs["wave_speed"] = ws["type"]
        append!(headers, String.(ws["headers"]))
    end

    # -- discretization and assembler ---------------------------------------
    disc_type = ""
    if haskey(disc, "discretization")
        disc_type = _fill(disc["discretization"], subs)
        subs["discretization"] = disc_type
    end
    assembler = _fill(disc["assembler"], subs)
    subs["assembler"] = assembler

    # -- linear solver -------------------------------------------------------
    solver_mode = step["linear_solver"]
    if solver_mode == "required" && linear_solver === nothing
        throw(ArgumentError("$stepper has no default linear solver; pass linear_solver="))
    end
    if solver_mode != "none" && linear_solver !== nothing
        sol = tbl["linear_solvers"][String(linear_solver)]
        subs["linear_solver"] = sol["type"]
        append!(headers, String.(sol["headers"]))
    end

    # -- precision -----------------------------------------------------------
    if get(step, "prec", "none") == "required"
        subs["prec"] = string(prec === nothing ? step["default_prec"] : prec)
    end

    # Omitting the solver is not the same as naming the default one: they are
    # different types, so the table carries a separate spelling for that case.
    step_template = (solver_mode == "optional" && linear_solver === nothing) ?
                    step["type_no_solver"] : step["type"]
    stepper_type = _fill(step_template, subs)
    append!(headers, String.(step["headers"]))

    # -- adaptivity ----------------------------------------------------------
    adapt_name, adapt_type, adapt_construct = "", "", ""
    if disc["main_shape"] != "variables"
        adapt_name = adaptivity === nothing ? disc["default_adaptivity"] : String(adaptivity)
        required = get(disc, "adaptivity_requirement", nothing)
        if required !== nothing && adapt_name != required
            throw(ArgumentError("$discretization requires $required, not $adapt_name"))
        end
        entry = tbl["adaptivity"][adapt_name]
        adapt_type = entry["type"]
        adapt_construct = replace(entry["construct"], "{name}" => "mesh_adaptor")
        append!(headers, String.(entry["headers"]))
    end

    return Scheme(discretization, disc["main_shape"], mesh_type, disc_type, assembler,
                  stepper_type, _fill(disc["flowing_variables"], subs),
                  String(disc["flowing_variables_arg"]), adapt_name, adapt_construct,
                  _ordered_unique(headers),
                  String.(disc["solution"]), Bool(step["adaptor"]))
end

# Duplicates are dropped but order is kept. The discretization's own headers are
# appended first, which is what puts the FV/KT headers ahead of anything pulling
# in DiFfRG::Quadrature<NT>.
function _ordered_unique(headers)
    out = String[]
    for h in headers
        h in out || push!(out, h)
    end
    return out
end
