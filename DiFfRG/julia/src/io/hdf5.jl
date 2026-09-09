"""
Reading the HDF5 a run writes.

The layout is fixed by `DiFfRG::HDF5Output`:

    /config/<section>[/<subsection>]      the run's parameters, as attributes
    /FE/<NNNNNN>/{nodes,<field>}          finite-element frames, attr `time`
    /<group>/<NNNNNN>/...                 further named field groups
    /scalars/<name>                       one appendable series per scalar
    /maps/<name>/<NNNNNN>/{data,coordinates}
    /coordinates/<name>

Because `/config` is mirrored into the file, a run is self-describing: the
parameters that produced it can be recovered from the output alone.
"""

"""
    SimulationData

A run's HDF5 output, read eagerly.

# Fields
- `path`: the file it came from.
- `config`: the parameters the run used, nested as in the parameter file.
- `scalars`: name => series, one entry per scalar readout.
- `maps`: name => `(times, coordinates, data)`, one entry per map.
- `coordinates`: name => the standalone coordinate sets.
- `fe`: group => `(times, nodes, fields)` for the finite-element frames.
"""
struct SimulationData
    path::String
    config::Dict{String,Any}
    scalars::Dict{String,Vector{Float64}}
    maps::Dict{String,NamedTuple}
    coordinates::Dict{String,Any}
    fe::Dict{String,NamedTuple}
end

function Base.show(io::IO, d::SimulationData)
    print(io, "SimulationData(", basename(d.path),
          ", ", length(d.scalars), " scalars",
          ", ", length(d.maps), " maps",
          ", ", length(d.fe), " FE groups)")
end

# Frames are named by a zero-padded counter, and HDF5 hands them back in
# whatever order the file lists them, so they are sorted before use -- otherwise
# a time series silently comes out shuffled.
_frame_order(names) = sort(collect(names))

"""
    read_h5(path) -> SimulationData

Read a DiFfRG output file.
"""
function read_h5(path::AbstractString)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    HDF5.h5open(path, "r") do file
        SimulationData(abspath(path),
                       haskey(file, "config") ? _read_config_group(file["config"]) : Dict{String,Any}(),
                       haskey(file, "scalars") ? _read_scalars(file["scalars"]) : Dict{String,Vector{Float64}}(),
                       haskey(file, "maps") ? _read_maps(file["maps"]) : Dict{String,NamedTuple}(),
                       haskey(file, "coordinates") ? _read_coordinates(file["coordinates"]) : Dict{String,Any}(),
                       _read_fe_groups(file))
    end
end

# The config is written as attributes on nested groups; booleans arrive as 0/1
# ints and arrays as serialized JSON, which is left as written rather than
# guessed at here.
function _read_config_group(group)
    out = Dict{String,Any}()
    for name in keys(HDF5.attrs(group))
        out[String(name)] = HDF5.attrs(group)[name]
    end
    for name in keys(group)
        child = group[name]
        child isa HDF5.Group && (out[String(name)] = _read_config_group(child))
    end
    return out
end

function _read_scalars(group)
    out = Dict{String,Vector{Float64}}()
    for name in keys(group)
        out[String(name)] = vec(Float64.(read(group[name])))
    end
    return out
end

function _read_maps(group)
    out = Dict{String,NamedTuple}()
    for name in keys(group)
        frames = _frame_order(keys(group[name]))
        isempty(frames) && continue
        node = group[name]
        out[String(name)] = (
            times = [_frame_time(node[f]) for f in frames],
            coordinates = [read(node[f]["coordinates"]) for f in frames],
            data = [read(node[f]["data"]) for f in frames],
        )
    end
    return out
end

function _read_coordinates(group)
    out = Dict{String,Any}()
    for name in keys(group)
        out[String(name)] = read(group[name])
    end
    return out
end

_frame_time(frame) = haskey(HDF5.attrs(frame), "time") ?
                     Float64(HDF5.attrs(frame)["time"]) : NaN

# A field group is any top-level group whose children are numbered frames; that
# covers /FE and any further named groups a model writes, without hard-coding
# the names a particular model happens to use.
const _NON_FIELD_GROUPS = ("config", "scalars", "maps", "coordinates")

function _read_fe_groups(file)
    out = Dict{String,NamedTuple}()
    for name in keys(file)
        String(name) in _NON_FIELD_GROUPS && continue
        node = file[name]
        node isa HDF5.Group || continue
        frames = _frame_order(keys(node))
        (isempty(frames) || !(node[frames[1]] isa HDF5.Group)) && continue

        fields = Dict{String,Vector{Array{Float64}}}()
        for f in frames, ds in keys(node[f])
            String(ds) == "nodes" && continue
            push!(get!(fields, String(ds), Vector{Array{Float64}}()),
                  Float64.(read(node[f][ds])))
        end
        out[String(name)] = (
            times = [_frame_time(node[f]) for f in frames],
            nodes = [haskey(node[f], "nodes") ? read(node[f]["nodes"]) : nothing
                     for f in frames],
            fields = fields,
        )
    end
    return out
end

"""
    read_h5(result) -> Vector{SimulationData}

Read every HDF5 file a run produced.
"""
read_h5(r::RunResult) = [read_h5(f) for f in h5_files(r)]
