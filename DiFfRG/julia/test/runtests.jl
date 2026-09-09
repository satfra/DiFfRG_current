using DiFfRG
using Test

const D = DiFfRG

# The integration tests compile a real app against a real install, which takes
# about a minute each. They are opt-in so the unit tests stay a fast check.
const RUN_INTEGRATION = get(ENV, "DIFFRG_JL_INTEGRATION", "0") == "1"
const TUT1 = abspath(joinpath(@__DIR__, "..", "..", "..", "Tutorials", "tut1"))

@testset "DiFfRG.jl" begin

    @testset "kvpairs" begin
        # A tuple of pairs must not be read through `pairs`, which would yield
        # index => pair and name the parameter `1`.
        @test collect(D.kvpairs(("/physical/T" => 0.1,))) == ["/physical/T" => 0.1]
        @test collect(D.kvpairs((T = 0.1,))) == [:T => 0.1]
    end

    @testset "parameter pointers" begin
        @test D.pointer(:T) == "/physical/T"
        @test D.pointer("T") == "/physical/T"
        @test D.pointer("/output/name") == "/output/name"
    end

    @testset "override flags" begin
        @test D.override_flags(:T, 0.1) == ["-sd", "/physical/T=0.1"]
        @test D.override_flags(:Nc, 3) == ["-si", "/physical/Nc=3"]
        @test D.override_flags("/output/name", "run") == ["-ss", "/output/name=run"]

        # Bool is an Integer in Julia, but `-si /x=true` is not parseable on the
        # C++ side, so it must be matched before the Integer branch.
        @test D.override_flags(:flag, true) == ["-sb", "/physical/flag=true"]
        @test D.override_flags(:flag, false) == ["-sb", "/physical/flag=false"]

        @test D.override_flags((T = 0.1, Nc = 3)) ==
              ["-sd", "/physical/T=0.1", "-si", "/physical/Nc=3"]
        @test_throws ArgumentError D.override_flags(:bad, [1, 2])
    end

    @testset "nesting and merging" begin
        @test D.nest((T = 0.1,)) == Dict("physical" => Dict("T" => 0.1))
        @test D.nest(("/a/b/c" => 1,)) == Dict("a" => Dict("b" => Dict("c" => 1)))

        # Merging must recurse: overriding one key of a section keeps the others.
        base = Dict("physical" => Dict("T" => 0.1, "Lambda" => 1.0))
        got = D.merge_tree(base, D.nest((T = 0.5,)))
        @test got["physical"]["T"] == 0.5
        @test got["physical"]["Lambda"] == 1.0

        @test_throws ArgumentError D.nest(Dict("/a" => 1, "/a/b" => 2))
    end

    @testset "config round-trip" begin
        tree = Dict("physical" => Dict("T" => 0.1, "Nc" => 3),
                    "output" => Dict("name" => "run"))
        mktempdir() do dir
            for name in ("parameter.json", "parameter.toml")
                path = D.write_config(joinpath(dir, name), tree)
                back = D.read_config(path)
                @test back["physical"]["T"] == 0.1
                @test back["physical"]["Nc"] == 3
                @test back["output"]["name"] == "run"
            end
        end
    end

    @testset "scan naming" begin
        # Matches what the Python package writes and parses, so a sweep produced
        # here stays readable by the existing tooling.
        @test D.scan_name((T = 0.1, muq = 0.3)) == "T:0.1_muq:0.3"
        @test D.scan_name(("/physical/T" => 0.1,)) == "T:0.1"
    end

    @testset "binary detection" begin
        mktempdir() do dir
            elf = joinpath(dir, "fake_exe")
            write(elf, UInt8[0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01])
            text = joinpath(dir, "parameter.toml")
            write(text, "[physical]\nT = 0.1\n")
            # Both are 0777 here, as they are in a checkout on a permissive
            # filesystem; only the magic number separates them.
            chmod(elf, 0o777)
            chmod(text, 0o777)
            @test D._is_native_binary(elf)
            @test !D._is_native_binary(text)
            @test D._executables(dir) == [elf]
        end
    end

    @testset "csv reading" begin
        mktempdir() do dir
            path = joinpath(dir, "out.csv")
            write(path, "t,kGeV,m2\n0.0,1.0,0.5\n0.1,0.9,0.4\n")
            cols = read_csv(path)
            @test cols["t"] == [0.0, 0.1]
            @test cols["m2"] == [0.5, 0.4]
        end
    end

    @testset "App construction" begin
        @test_throws ArgumentError App(mktempdir())          # no CMakeLists.txt
        @test_throws ArgumentError App("/nonexistent/path/xyz")

        if isdir(TUT1)
            app = App(TUT1)
            @test app.source == TUT1
            @test D.base_parameter_file(app) == joinpath(TUT1, "parameter.json")
            @test_throws ArgumentError App(TUT1; parameter_file = "nope.toml")
            @test D.base_parameter_file(App(TUT1; parameter_file = "parameter.toml")) ==
                  joinpath(TUT1, "parameter.toml")
        end

        # A driver with its own default parameter file name must get that file,
        # not the project's shared one.
        ON = abspath(joinpath(@__DIR__, "..", "..", "..", "Examples", "ONfiniteT"))
        if isdir(ON)
            @test D.base_parameter_file(App(ON; target = "KT")) ==
                  joinpath(ON, "parameter_KT.toml")
            @test D.base_parameter_file(App(ON; target = "KT_sigma")) ==
                  joinpath(ON, "parameter_KT_sigma.toml")
            @test D.base_parameter_file(App(ON; target = "CG")) ==
                  joinpath(ON, "parameter.toml")
            # The build key must follow the sources, not the clock.
            inst = find_install().prefix
            @test D.build_key(TUT1, "Release", String[], inst) ==
                  D.build_key(TUT1, "Release", String[], inst)
            @test D.build_key(TUT1, "Debug", String[], inst) !=
                  D.build_key(TUT1, "Release", String[], inst)
            # A tree built against one install is not valid against another.
            @test D.build_key(TUT1, "Release", String[], inst) !=
                  D.build_key(TUT1, "Release", String[], "/some/other/prefix")
            # Targets share a tree, so the generated flow library is built once
            # for a project rather than once per target.
            @test App(TUT1; target = "a").builddir == App(TUT1; target = "b").builddir
        end
    end

    @testset "scheme table" begin
        table = load_schemes()
        @test table["version"] == 1
        for section in ("meshes", "discretizations", "steppers", "linear_solvers")
            @test !isempty(table[section])
        end

        # Every header the table names must exist in the install.
        inc = joinpath(find_install().prefix, "include")
        src = normpath(joinpath(@__DIR__, "..", "..", "include"))
        for (_, entries) in table
            entries isa AbstractDict || continue   # `version` is a plain Int
            for (_, entry) in entries
                entry isa AbstractDict || continue
                for h in get(entry, "headers", String[])
                    @test isfile(joinpath(inc, h)) || isfile(joinpath(src, h))
                end
            end
        end
    end

    @testset "scheme resolution" begin
        # CG + IDA must reproduce what Tutorials/tut1 writes by hand.
        cg = resolve_scheme(model = "Tut1", dim = 1, discretization = "CG")
        @test cg.discretization == "CG::Discretization<Tut1, RectangularMesh<1>>"
        @test cg.assembler == "CG::Assembler<CG::Discretization<Tut1, RectangularMesh<1>>>"
        @test cg.stepper == "TimeStepperSUNDIALS_IDA<$(cg.assembler)>"
        @test cg.flowing_variables == "FE::FlowingVariables"

        # dDG takes the DG discretization but the dDG assembler.
        ddg = resolve_scheme(model = "M", dim = 1, discretization = "dDG")
        @test startswith(ddg.discretization, "DG::Discretization")
        @test startswith(ddg.assembler, "dDG::Assembler")

        # Naming a solver is a different type from omitting it, even when the
        # default resolves to the same solver.
        with = resolve_scheme(model = "M", dim = 1, discretization = "CG",
                              linear_solver = "UMFPack")
        @test occursin("UMFPack", with.stepper)
        @test with.stepper != cg.stepper

        # Variables has no mesh and no discretization at all.
        var = resolve_scheme(model = "M", dim = 0, discretization = "Variables",
                             stepper = "BoostABM")
        @test var.main_shape == "variables"
        @test isempty(var.mesh)
        @test isempty(var.discretization)
        @test var.assembler == "Variables::Assembler<M>"
        @test var.flowing_variables == "FlowingVariables"
        @test var.solution == ["variables"]

        # The bundle differs by assembler, which is what a generator has to
        # check a model body against.
        @test "fe_hessians" in cg.solution
        @test "fe_hessians" ∉ resolve_scheme(model = "M", dim = 1,
                                             discretization = "DG").solution

        # KT keeps its own headers first: they must precede anything pulling in
        # DiFfRG::Quadrature<NT>.
        kt = resolve_scheme(model = "M", dim = 1, discretization = "KT")
        @test first(kt.headers) == "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
        @test occursin("MinModLimiter", kt.assembler)
        @test occursin("MaxEigenvalueWaveSpeed", kt.assembler)
        @test kt.adaptivity == "NoAdaptivity"
    end

    @testset "scheme constraints" begin
        # LDG needs a serial triangulation. In a serial build the two mesh types
        # are the same, so this can only be caught here, not by the compiler.
        @test_throws ArgumentError resolve_scheme(model = "M", dim = 1,
                                                  discretization = "LDG",
                                                  mesh = "RectangularMeshParallel")
        @test resolve_scheme(model = "M", dim = 1, discretization = "LDG").mesh ==
              "RectangularMeshSerial<1>"

        # KT cannot be refined.
        @test_throws ArgumentError resolve_scheme(model = "M", dim = 1,
                                                  discretization = "KT",
                                                  adaptivity = "HAdaptivity")

        # These aliases have no default for their extra template parameters.
        @test_throws ArgumentError resolve_scheme(model = "M", dim = 1,
                                                  discretization = "CG",
                                                  stepper = "SUNDIALS_IDA_BoostRK")
        @test occursin("0>", resolve_scheme(model = "M", dim = 1, discretization = "CG",
                                            stepper = "SUNDIALS_IDA_BoostRK",
                                            linear_solver = "UMFPack").stepper)

        @test_throws ArgumentError resolve_scheme(model = "M", dim = 1,
                                                  discretization = "nope")
        @test_throws ArgumentError resolve_scheme(model = "M", dim = 1,
                                                  discretization = "CG", stepper = "nope")
    end

    @testset "model emission" begin
        m = Model("M"; dim = 1, fe = [Scalar("u")],
                  params = (Lambda = 1.0, Nc = 3, tag = "x"),
                  initial_condition = cxx"""values[idxf("u")] = 0.;""",
                  flux = cxx"""flux[idxf("u")][0] = 0.;""")
        table = load_schemes()
        scheme = resolve_scheme(model = "M", dim = 1, discretization = "CG", table = table)
        hh = emit_model(m, scheme; table = table)

        # A parameter is named once in Julia and reaches C++ with the right
        # typed getter; `Nc = 3` must not be read as a double.
        @test occursin("double Lambda;", hh)
        @test occursin("int Nc;", hh)
        @test occursin("std::string tag;", hh)
        @test occursin("Lambda = config.get_double(\"/physical/Lambda\");", hh)
        @test occursin("Nc = config.get_int(\"/physical/Nc\");", hh)

        @test occursin("using FEFunctionDesc = FEFunctionDescriptor<Scalar<\"u\">>;", hh)
        @test occursin("class M : public def::AbstractModel<M, Components>,", hh)
        @test occursin("def::fRG(config.get_double(\"/physical/Lambda\"))", hh)
        # Only members that were given are emitted; the rest keep their defaults.
        @test occursin("void flux(", hh)
        @test !occursin("void source(", hh)

        cc = emit_main(m, scheme)
        @test occursin("using Assembler = CG::Assembler<", cc)
        @test occursin("TimeStepper time_stepper(config, assembler, data_out, mesh_adaptor);", cc)
        @test occursin("FE::FlowingVariables initial_condition(discretization);", cc)
        # get_json()/JSONValue are deprecated spellings; never emit them.
        @test !occursin("get_json", cc)

        # FunctionND and the positional ComponentDescriptor.
        m2 = Model("V"; dim = 1, fe = [Scalar("u")],
                   variables = [FunctionND("Z", 8)], extractors = [Scalar("e")],
                   initial_condition = cxx"""values[idxf("u")] = 0.;""")
        hh2 = emit_model(m2, scheme; table = table)
        @test occursin("FunctionND<\"Z\", 8>", hh2)
        @test occursin("ComponentDescriptor<FEFunctionDesc, VariableDesc, ExtractorDesc>", hh2)

        # Extractors without variables still need the empty slot spelled out.
        m3 = Model("W"; dim = 1, fe = [Scalar("u")], extractors = [Scalar("e")],
                   initial_condition = cxx"""values[idxf("u")] = 0.;""")
        @test occursin("ComponentDescriptor<FEFunctionDesc, VariableDescriptor<>, ExtractorDesc>",
                       emit_model(m3, scheme; table = table))
    end

    @testset "variables main shape" begin
        table = load_schemes()
        m = Model("V"; dim = 0, variables = [Scalar("A")],
                  mixins = ["fRG", "NoJacobians"],
                  dt_variables = cxx"""residual[idxv("A")] = 0.;""")
        scheme = resolve_scheme(model = "V", dim = 0, discretization = "Variables",
                                stepper = "BoostABM", table = table)
        cc = emit_main(m, scheme)
        # No field space at all: no mesh, no discretization, no adaptor.
        @test !occursin("Mesh", cc)
        @test !occursin("Discretization", cc)
        @test !occursin("mesh_adaptor", cc)
        @test occursin("Assembler assembler(model, config);", cc)
        @test occursin("FlowingVariables initial_condition;", cc)
    end

    @testset "default mixins per scheme" begin
        table = load_schemes()
        # CG needs no numerical flux; the face-based schemes do.
        @test DiFfRG._default_mixins(table, "CG") == ["fRG", "AD", "FlowBoundaries"]
        for d in ("DG", "dDG", "LDG")
            @test "LLFFlux" in DiFfRG._default_mixins(table, d)
        end
        # FlowBoundaries is a FEM face-flux boundary; a finite-volume scheme
        # needs one of the fv_boundaries.hh stencils instead.
        kt = DiFfRG._default_mixins(table, "KT")
        @test "FVDefaultBoundaries" in kt
        @test "FlowBoundaries" ∉ kt
        @test "LLFFlux" ∉ kt
        # No field space: nothing to apply boundaries or FE jacobians to.
        @test DiFfRG._default_mixins(table, "Variables") == ["fRG", "NoJacobians"]
    end

    @testset "model emission errors" begin
        @test_throws ArgumentError Model("M"; nonsense_member = cxx"""x""")

        table = load_schemes()
        # Two mixins for the same role would be an ambiguous base.
        bad = Model("M"; dim = 1, fe = [Scalar("u")], mixins = ["AD", "FE_AD"],
                    initial_condition = cxx"""values[idxf("u")] = 0.;""")
        @test_throws ArgumentError DiFfRG.resolve_mixins(bad, "CG"; table = table)
        @test_throws ArgumentError DiFfRG.resolve_mixins(
            Model("M"; mixins = ["NoSuchMixin"]), "CG"; table = table)
        # A constraint mixin needs the component it constrains.
        @test_throws ArgumentError DiFfRG.resolve_mixins(
            Model("M"; mixins = ["ConstrainOriginBoundaryPointToZero"]), "CG"; table = table)
        @test occursin("<\"u\", M>", first(DiFfRG.resolve_mixins(
            Model("M"; mixins = ["ConstrainOriginBoundaryPointToZero"],
                  constraint_component = "u"), "CG"; table = table))[1])

        # AbstractModel deletes the default initial_condition.
        @test_throws ArgumentError generate(Model("M"; dim = 1, fe = [Scalar("u")]),
                                            mktempdir())
    end

    @testset "flow library detection" begin
        flows_dir = abspath(joinpath(@__DIR__, "..", "..", "..", "Tutorials", "tut3", "flows"))
        if isdir(flows_dir)
            fl = FlowLibrary(flows_dir)
            # Both names are read out of the generated files, not assumed.
            @test fl.class == "ONFiniteTFlows"
            @test fl.target == "ONFiniteTFlows"
            @test fl.member == "flow_equations"

            m = Model("F"; dim = 1, fe = [Scalar("u")], flows = fl,
                      initial_condition = cxx"""values[idxf("u")] = 0.;""")
            @test occursin("mutable ONFiniteTFlows flow_equations;",
                           emit_model(m, resolve_scheme(model = "F", dim = 1,
                                                        discretization = "CG");
                                      table = load_schemes()))
            @test occursin("ONFiniteTFlows", emit_cmake(m))
            @test occursin("add_subdirectory(flows)", emit_cmake(m))
        end
        @test_throws ArgumentError FlowLibrary(mktempdir())
    end

    @testset "expression DSL" begin
        comps = Dict("fe_functions" => ["u"], "variables" => String[],
                     "extractors" => String[])
        CG = ["fe_functions", "fe_derivatives", "fe_hessians", "extractors", "variables"]
        DG = ["fe_functions", "extractors", "variables"]
        render(b, avail = CG) = DiFfRG.render_body(b, "flux";
                                                   components = comps, available = avail)

        code = render(@julia((F, x, sol) -> begin
            u  = sol.fe_functions[:u]
            du = sol.fe_derivatives[:u][1]
            F[:u][1] = 0.5 * u^2 + x[1] * du
        end))
        # The bundle binding is emitted once per entry actually read.
        @test occursin("const auto &fe_functions = get<\"fe_functions\">(sol);", code)
        @test occursin("const auto &fe_derivatives = get<\"fe_derivatives\">(sol);", code)
        @test !occursin("fe_hessians", code)
        # The user's own names survive into the C++.
        @test occursin("const auto u = fe_functions[idxf(\"u\")];", code)
        # 1-based in Julia, 0-based on the way out.
        @test occursin("fe_derivatives[idxf(\"u\")][0]", code)
        @test occursin("pos[0]", code)
        @test occursin("flux[idxf(\"u\")][0] =", code)
        # An integer power is powr<n>, not a call to pow.
        @test occursin("powr<2>(u)", code)

        # Every numeric literal is a double: `1 / 2` must not become integer
        # division, which C++ would evaluate to zero.
        @test occursin("(1. / 2.)", render(@julia((F, x, sol) -> begin
            F[:u][1] = 1 / 2
        end)))
        # A template parameter still has to be an integer.
        @test occursin("tF::lB<0>(", render(@julia((F, x, sol) -> begin
            F[:u][1] = lB{0}(sol.fe_functions[:u], k, T, 4)
        end)))
        @test occursin("M_PI", render(@julia((F, x, sol) -> begin
            F[:u][1] = pi
        end)))
        @test occursin(" ? ", render(@julia((F, x, sol) -> begin
            F[:u][1] = ifelse(sol.fe_functions[:u] > 0, 1.0, 0.0)
        end)))
    end

    @testset "expression DSL rejections" begin
        comps = Dict("fe_functions" => ["u"], "variables" => String[],
                     "extractors" => String[])
        CG = ["fe_functions", "fe_derivatives", "fe_hessians", "extractors", "variables"]
        DG = ["fe_functions", "extractors", "variables"]
        render(b, avail) = DiFfRG.render_body(b, "flux";
                                              components = comps, available = avail)

        # The whole point of mirroring the bundle: a mismatch is caught here,
        # with a readable message, not as a template error in the compiler.
        @test_throws DiFfRG.DSLError render(@julia((F, x, sol) -> begin
            F[:u][1] = sol.fe_hessians[:u][1][1]
        end), DG)
        # ... and the same body is fine under an assembler that provides it.
        @test occursin("fe_hessians", render(@julia((F, x, sol) -> begin
            F[:u][1] = sol.fe_hessians[:u][1][1]
        end), CG))

        @test_throws DiFfRG.DSLError render(@julia((F, x, sol) -> begin
            F[:u][1] = sol.fe_functions[:nope]
        end), CG)
        @test_throws DiFfRG.DSLError render(@julia((F, x, sol) -> begin
            F[:u][1] = sol.not_an_entry[:u]
        end), CG)
        # Branching on a value does not survive AD or a device instantiation.
        @test_throws DiFfRG.DSLError render(@julia((F, x, sol) -> begin
            u = sol.fe_functions[:u]
            if u > 0
                F[:u][1] = u
            end
        end), CG)
        @test_throws DiFfRG.DSLError render(@julia((F, x, sol) -> begin
            for i in 1:3
                F[:u][1] = 1.0
            end
        end), CG)
        # A fixed number type would break the AD instantiations.
        @test_throws DiFfRG.DSLError render(@julia((F, x, sol) -> begin
            u::Float64 = sol.fe_functions[:u]
            F[:u][1] = u
        end), CG)
        # Threshold functions carry their level as a template parameter.
        @test_throws DiFfRG.DSLError render(@julia((F, x, sol) -> begin
            F[:u][1] = lB(sol.fe_functions[:u], k, T, 4)
        end), CG)
    end

    @testset "flow DSL batching" begin
        vars = ["A", "B", "C"]
        render(b) = DiFfRG.render_flow(b, vars, "fe")

        # Independent launches share one DeferredMaps scope.
        one = render(@variables_flow(begin
            r[:A] = flows.A.map(co, args)
            r[:B] = flows.B.map(co, args)
            r[:C] = flows.C.map(co, args)
        end))
        @test count(x -> occursin("DeferredMaps defer;", x), split(one, '\n')) == 1
        @test occursin("fe.A.map(&residual[idxv(\"A\")], co, args);", one)

        # A launch that reads an earlier destination has to wait for it, so the
        # batch closes and the first lands on its own. Getting this wrong is not
        # a compile error but wrong numbers, which is why it is derived.
        dep = render(@variables_flow(begin
            r[:A] = flows.A.map(co, args)
            r[:B] = flows.B.map(co, r[:A])
            r[:C] = flows.C.map(co, args)
        end))
        lines = split(dep, '\n')
        a_line = findfirst(l -> occursin("fe.A.map", l), lines)
        scope = findfirst(l -> occursin("DeferredMaps defer;", l), lines)
        @test a_line < scope           # A is outside, and before, the batch
        @test count(x -> occursin("DeferredMaps defer;", x), lines) == 1

        # Anything that is not a launch closes the batch too.
        upd = render(@variables_flow(begin
            r[:A] = flows.A.map(co, args)
            @update dtA from r[:A]
            r[:B] = flows.B.map(co, args)
        end))
        @test !occursin("DeferredMaps", upd)
        @test occursin("dtA.update(&residual[idxv(\"A\")]);", upd)

        # A single launch is not wrapped: outside a scope, map() keeps its
        # original contract that the result is there when it returns.
        @test !occursin("DeferredMaps", render(@variables_flow(begin
            r[:A] = flows.A.map(co, args)
        end)))
    end

    @testset "flow DSL structure" begin
        vars = ["ZA", "Zc"]
        code = DiFfRG.render_flow(@variables_flow(begin
            @guard ran_away(x) => FlowAbort(t)
            @update ZA, Zc
            args = @tie(k, ZA, Zc)
            @fixedpoint (dtZA, dtZc) size=N tol=eps maxiter=imax begin
                r[:ZA] = flows.ZA.map(co1, args)
                r[:Zc] = flows.Zc.map(co1, args)
                @update dtZA from r[:ZA]
                @update dtZc from r[:Zc]
            end
        end), vars, "flow_equations")
        lines = split(code, '\n')

        # The guard must precede every launch: the assembler fences only after
        # dt_variables returns, so throwing past an in-flight kernel is what its
        # placement defends against.
        guard = findfirst(l -> occursin("throw FlowAbort", l), lines)
        launch = findfirst(l -> occursin(".map(", l), lines)
        @test guard < launch

        @test occursin("const auto args = device::tie(k, ZA, Zc);", code)
        @test occursin("ZA.update(&variables.data()[idxv(\"ZA\")]);", code)

        # The fixed-point bookkeeping is generated, not written by hand.
        @test occursin("std::vector<double> old_dtZA(N);", code)
        @test occursin("while (!converged)", code)
        @test occursin("std::abs(dtZA[i] - old_dtZA[i])", code)
        @test occursin("if (dist < eps || n_iter >= imax) converged = true;", code)

        # The two launches inside the loop are independent, so they batch.
        @test occursin("DeferredMaps defer;", code)
    end

    @testset "flow DSL rejections" begin
        @test_throws DiFfRG.DSLError DiFfRG.render_flow(@variables_flow(begin
            r[:nope] = flows.X.map(co, args)
        end), ["A"], "fe")
        @test_throws DiFfRG.DSLError DiFfRG.render_flow(@variables_flow(begin
            @guard something
        end), ["A"], "fe")
        @test_throws DiFfRG.DSLError DiFfRG.render_flow(@variables_flow(begin
            @fixedpoint (a) tol=1 begin
                r[:A] = flows.A.map(co, args)
            end
        end), ["A"], "fe")
        @test_throws DiFfRG.DSLError DiFfRG.render_flow(@variables_flow(begin
            @notamacro x
        end), ["A"], "fe")
    end

    @testset "platform gate" begin
        # Julia's platform triplet says nothing about instruction-set level or
        # glibc version, so these are the checks Pkg cannot make for us.
        @test supports_x86_64_v3() isa Union{Bool,Nothing}
        @test glibc_version() isa Union{VersionNumber,Nothing}
        @test bundle_variant() isa String
        @test occursin("platform", artifact_status())

        if Sys.islinux() && glibc_version() !== nothing
            # A floor no system meets must be refused, with the reason.
            @test_throws ErrorException check_bundle_runnable(glibc_floor = v"99.0")
            # ... and reported rather than thrown when asked.
            @test check_bundle_runnable(glibc_floor = v"99.0", strict = false) == false
        end

        withenv("DIFFRG_VARIANT" => "linux-x86_64-v3-cuda12") do
            @test bundle_variant() == "linux-x86_64-v3-cuda12"
        end
    end

    @testset "default_jobs" begin
        j = DiFfRG.default_jobs()
        @test 1 <= j <= 8
        @test j <= Sys.CPU_THREADS
        # The lower bound comes from the clamp, so a machine with almost no free
        # memory still gets one job rather than zero.
        @test clamp(min(Sys.CPU_THREADS, 0), 1, 8) == 1
    end

    @testset "install precedence" begin
        cands = DiFfRG.candidate_prefixes()
        @test cands == unique(cands)          # DIFFRG_DIR may repeat a default
        conventional = joinpath(homedir(), ".local", "share", "DiFfRG")
        scratch = DiFfRG.install_prefix()
        if conventional in cands && scratch in cands
            # A hand-maintained install must not be shadowed by the fallback
            # this package installs for itself.
            @test findfirst(==(conventional), cands) < findfirst(==(scratch), cands)
        end
        # An explicit DIFFRG_DIR outranks everything.
        withenv("DIFFRG_DIR" => "/tmp/some-diffrg") do
            @test first(DiFfRG.candidate_prefixes()) == "/tmp/some-diffrg"
        end
        @test occursin("->", sprint(which_install)) ||
              occursin("none usable", sprint(which_install))
    end

    @testset "bootstrap" begin
        # Package-owned, so it survives updates and goes away with the package.
        @test occursin("scratchspaces", DiFfRG.install_prefix())
        @test isfile(DiFfRG.wizard_script())
        # The install this package makes for itself must be discoverable.
        @test DiFfRG.install_prefix() in DiFfRG.candidate_prefixes()
    end

    @testset "install discovery" begin
        inst = find_install()
        @test isdir(inst.prefix)
        @test !isempty(D.config_file(inst.prefix))
        @test_throws ErrorException find_install(mktempdir())
    end

    if RUN_INTEGRATION
        @testset "integration: every scheme compiles" begin
            # The strongest check on the table: a wrong type string or a missing
            # header becomes a compile error instead of sitting there unnoticed.
            # (Deliberately corrupting an entry does make this fail -- verified.)
            include(joinpath(@__DIR__, "gen_probe.jl"))
            dir = mktempdir(; prefix = "diffrg_probe_")
            combos = generate_probe(dir)
            @test combos > 40
            app = App(dir)
            build!(app; verbose = false)
            @test isfile(executable_path(app))
        end

        @testset "integration: generated tut1 matches the hand-written one" begin
            m = Model("Tut1"; dim = 1, fe = [Scalar("u")],
                      params = (Lambda = 1.0, a = 0.0, b = 1.0, c = 0.0, d = 0.0),
                      initial_condition = cxx"""
                          values[idxf("u")] = prm.a + prm.b * pos[0] +
                                              prm.c * powr<2>(pos[0]) + prm.d * powr<3>(pos[0]);
                          """,
                      flux = cxx"""
                          flux[idxf("u")][0] = 0.5 * powr<2>(get<"fe_functions">(sol)[idxf("u")]);
                          """)
            gen = generate(m, mktempdir(; prefix = "diffrg_gen_"))
            build!(gen; verbose = false)

            hand = App(TUT1)
            hc = DiFfRG.read_config(DiFfRG.base_parameter_file(hand))
            cfg = ("/timestepping/final_time" => 0.5,
                   "/discretization" => hc["discretization"])
            p = (a = 0.0, b = 1.0, c = 0.0, d = 0.0)

            gu = only(read_h5(run(gen; params = p, config = cfg, verbose = false))).fe["FE"].fields["u"][end]
            hu = only(read_h5(run(hand; params = p, config = cfg, verbose = false))).fe["FE"].fields["u"][end]

            @test length(gu) == length(hu)
            # Same model, same settings: the generated app must agree exactly,
            # not merely to run tolerance.
            @test gu == hu
            # And both must match the exact Burgers solution u = x/(1+t).
            @test isapprox(maximum(gu), 1 / 1.5; rtol = 1e-3)
        end

        @testset "integration: tut1 from the expression DSL" begin
            # The same comparison as above, but with no C++ written at all.
            m = Model("Tut1"; dim = 1, fe = [Scalar("u")],
                      params = (Lambda = 1.0, a = 0.0, b = 1.0, c = 0.0, d = 0.0),
                      initial_condition = @julia((values, pos) -> begin
                          x = pos[1]
                          values[:u] = prm.a + prm.b * x + prm.c * x^2 + prm.d * x^3
                      end),
                      flux = @julia((F, x, sol) -> begin
                          u = sol.fe_functions[:u]
                          F[:u][1] = 0.5 * u^2
                      end))
            gen = generate(m, mktempdir(; prefix = "diffrg_dsl_"))
            build!(gen; verbose = false)

            hand = App(TUT1)
            hc = DiFfRG.read_config(DiFfRG.base_parameter_file(hand))
            cfg = ("/timestepping/final_time" => 0.5,
                   "/discretization" => hc["discretization"])
            p = (a = 0.0, b = 1.0, c = 0.0, d = 0.0)
            gu = only(read_h5(run(gen; params = p, config = cfg, verbose = false))).fe["FE"].fields["u"][end]
            hu = only(read_h5(run(hand; params = p, config = cfg, verbose = false))).fe["FE"].fields["u"][end]
            @test gu == hu
        end

        @testset "integration: graph flow body" begin
            m = Model("GraphFlow"; dim = 0,
                      variables = [Scalar("A"), Scalar("B")],
                      params = (Lambda = 1.0, rate = 2.0),
                      mixins = ["fRG", "NoJacobians"],
                      initial_condition_variables = cxx"""
                          values[idxv("A")] = 1.; values[idxv("B")] = 1.;
                          """,
                      dt_variables = @variables_flow(begin
                          r[:A] = prm.rate * variables[idxv("A")]
                          r[:B] = 2. * prm.rate * variables[idxv("B")]
                      end),
                      readouts = cxx"""
                          auto out = output.table("data.csv");
                          out.value("A", get<"variables">(sol)[idxv("A")]);
                          out.value("B", get<"variables">(sol)[idxv("B")]);
                          """)
            app = generate(m, mktempdir(; prefix = "diffrg_graph_");
                           discretization = "Variables", stepper = "BoostABM")
            build!(app; verbose = false)
            res = run(app; config = ("/timestepping/final_time" => 1.0,), verbose = false)
            @test res.success
            cols = read_csv(first(csv_files(res)))
            @test isapprox(last(cols["A"]), exp(-2); rtol = 1e-4)
            @test isapprox(last(cols["B"]), exp(-4); rtol = 1e-4)
        end

        @testset "integration: generated model with a flow library" begin
            tut3 = abspath(joinpath(@__DIR__, "..", "..", "..", "Tutorials", "tut3"))
            if isdir(joinpath(tut3, "flows"))
                m = Model("ON_finiteT"; dim = 1, fe = [Scalar("m2")],
                          params = (Lambda = 1.0, N = 2.0, T = 0.001,
                                    lambda2 = -0.1, lambda4 = 50.0, lambda6 = 0.0),
                          flows = FlowLibrary(joinpath(tut3, "flows")),
                          initial_condition = cxx"""
                              values[idxf("m2")] = prm.lambda2 + prm.lambda4 * pos[0] +
                                                   prm.lambda6 * powr<2>(pos[0]);
                              """,
                          set_time = cxx"""
                              t = t_; k = std::exp(-t) * prm.Lambda; flow_equations.set_k(k);
                              """,
                          flux = cxx"""
                              const auto rho = pos[0];
                              const auto &fe_functions = get<"fe_functions">(sol);
                              const auto &fe_derivatives = get<"fe_derivatives">(sol);
                              const auto m2Pi = fe_functions[idxf("m2")];
                              const auto m2Sigma = m2Pi + 2. * rho * fe_derivatives[idxf("m2")][0];
                              flow_equations.V.get(flux[idxf("m2")][0], k, prm.N, prm.T, m2Pi, m2Sigma);
                              """,
                          cell_indicator = cxx"""
                              indicator = get<"fe_hessians">(sol)[idxf("m2")][0][0];
                              """)
                hand = App(tut3)
                hc = DiFfRG.read_config(DiFfRG.base_parameter_file(hand))
                gen = generate(m, mktempdir(; prefix = "diffrg_t3_");
                               config = ("/discretization" => hc["discretization"],
                                         "/integration" => get(hc, "integration", Dict())))
                build!(gen; verbose = false)
                build!(hand; verbose = false)

                cfg = ("/timestepping/final_time" => 2.0,)
                g = only(read_h5(run(gen; config = cfg, verbose = false))).fe["FE"].fields["m2"][end]
                h = only(read_h5(run(hand; config = cfg, verbose = false))).fe["FE"].fields["m2"][end]
                @test length(g) == length(h)
                @test g == h
            end
        end

        @testset "integration: generated variables model" begin
            # dt_variables fills the residual and the assembler takes
            # A_dot = -residual, so a decaying variable needs a positive one.
            m = Model("Decay"; dim = 0, variables = [Scalar("A")],
                      params = (Lambda = 1.0, rate = 2.0),
                      mixins = ["fRG", "NoJacobians"],
                      initial_condition_variables = cxx"""values[idxv("A")] = 1.;""",
                      dt_variables = cxx"""
                          residual[idxv("A")] = prm.rate * get<"variables">(sol)[idxv("A")];
                          """,
                      readouts = cxx"""
                          auto out = output.table("data.csv");
                          out.value("A", get<"variables">(sol)[idxv("A")]);
                          """)
            app = generate(m, mktempdir(; prefix = "diffrg_var_");
                           discretization = "Variables", stepper = "BoostABM")
            build!(app; verbose = false)
            res = run(app; config = ("/timestepping/final_time" => 1.0,), verbose = false)
            @test res.success
            A = read_csv(first(csv_files(res)))["A"]
            @test isapprox(last(A), exp(-2); rtol = 1e-4)
        end

        @testset "integration: build, run, read" begin
            app = App(TUT1)
            build!(app; verbose = false)
            @test isfile(executable_path(app))

            # tut1 is Burgers' equation with u(0,x) = a + b x + c x^2 + d x^3.
            # With b as the only non-zero coefficient the initial condition is
            # exactly b*x on x in [0,1], which pins both the override plumbing
            # and the reader to a known answer.
            res = run(app; params = (b = 2.0,),
                      config = ("/timestepping/final_time" => 0.1,), verbose = false)
            @test res.success
            @test !isempty(h5_files(res))

            data = only(read_h5(res))
            @test data.config["physical"]["b"] == 2.0
            u0 = data.fe["FE"].fields["u"][1]
            @test extrema(u0) == (0.0, 2.0)
        end

        @testset "integration: scan varies the parameter" begin
            app = App(TUT1)
            sweep = scan(app; b = [0.5, 1.0],
                         config = ("/timestepping/final_time" => 0.1,))
            @test length(sweep) == 2
            @test all(r -> r.success, sweep)
            maxima = [maximum(only(read_h5(r)).fe["FE"].fields["u"][1]) for r in sweep]
            @test maxima ≈ [0.5, 1.0]
        end
    end
end
