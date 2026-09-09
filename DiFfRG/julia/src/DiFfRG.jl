"""
    DiFfRG

A Julia interface to [DiFfRG](https://github.com/satfra/DiFfRG), a discretization
framework for functional Renormalization Group flows.

This layer drives applications that already exist: it locates a DiFfRG install,
builds an app against it, runs it over a parameter grid, and reads the results
back. Because every parameter is settable on the command line, a sweep is one
build and one process per point -- nothing is recompiled between points.

```julia
using DiFfRG

app = App("Examples/ONfiniteT"; target = "CG")
res = run(app; params = (T = 0.1,))
data = read_h5(res)

sweep = scan(app; T = 0.01:0.01:0.2)
```
"""
module DiFfRG

import Dates
import HDF5
import JSON3
import Downloads
import SHA
import Scratch
import TOML

export App, DiFfRGInstall, RunResult, SimulationData
export build!, csv_files, executable_path, find_install, h5_files
export artifact_status, bundle_variant, check_bundle_runnable
export install_diffrg!, install_prefix, uninstall_diffrg!, which_install
export glibc_version, macos_version, supports_x86_64_v3
export read_csv, read_h5, run_app, scan
export Scheme, load_schemes, resolve_scheme
export Component, CxxBody, FlowLibrary, FunctionND, Model, Scalar
export @cxx_str, @julia, @variables_flow, emit_cmake, emit_main, emit_model, generate

include("platform.jl")
include("install.jl")
include("bootstrap.jl")
include("config.jl")
include("schemes.jl")
include("app.jl")
include("run.jl")
include("model.jl")
include("dsl.jl")
include("flowdsl.jl")
include("generate.jl")
include("io/hdf5.jl")
include("io/csv.jl")

end # module
