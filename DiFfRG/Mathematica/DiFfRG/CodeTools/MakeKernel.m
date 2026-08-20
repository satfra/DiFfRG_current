(* ::Package:: *)

(* Exported symbols added here with SymbolName::usage *)

BeginPackage["DiFfRG`CodeTools`MakeKernel`"]

GetStandardKernelDefinitions::usage = "GetStandardKernelDefinitions[] returns a list of standard kernel definitions used in DiFfRG."

MakeKernel::usage = "MakeKernel[kernel_Association, parameterList_List,integrandFlow_,constantFlow_:0., integrandDefinitions_String:\"\", constantDefinitions_String:\"\"]
Make a kernel from a given flow equation, parmeter list and kernel. The kernel must be a valid specification of an integration kernel.
This Function creates an integrator that evaluates (constantFlow + \[Integral]integrandFlow). One can prepend additional c++ definitions to the flow equation by using the integrandDefinitions and constantDefinitions parameters.
These are prepended to the respective methods of the integration kernel, allowing one to e.g. define specific angles one needs for the flow code.

The options \"KernelReturnTransform\" and \"ConstantReturnTransform\" (default Identity) accept a Mathematica function applied to the optimized expression before code generation, which lets you wrap the return value, e.g. \"KernelReturnTransform\" -> Re renders the kernel return as real(...).";

MakeKernel::Invalid = "The given arguments are invalid. See MakeKernel::usage";

MakeKernel::InvalidSpec = "The given kernel specification is invalid.";

MakeKernel::InvalidSpec = "The parameters given to MakeKernel are not valid.";

MakeKernel::MissingKey = "The key \"`1`\" is missing.";

MakeKernel::MissingType = "Parameter \"`1`\" has no Type specified, defaulting to double.";

MakeKernel::InvalidKey = "The key \"`1`\" is invalid: `2`";

MakeKernel::exportFailed = "Export of sources.m to `1` failed.";

MakeKernel::notEven = "MatsubaraEven requested for kernel \"`1`\" but it is not even in \"`2`\"; emitting the standard kernel (the integrator keeps the explicit kernel(+f0)+kernel(-f0) form). This is expected when the loop contains fermionic dressings evaluated at f0-shifted arguments.";

Begin["`Private`"]

Needs["DiFfRG`CodeTools`Utils`"]

Needs["DiFfRG`CodeTools`Directory`"]

Needs["DiFfRG`CodeTools`Export`"]

Needs["DiFfRG`CodeTools`TemplateParameterGeneration`"]

Needs["DiFfRG`CodeTools`Regulator`"]

$ADReplacements = {"double" -> "autodiff::real", "DiFfRG::complex<double>" -> "cxreal"};

$AD2Replacements = {"double" -> "autodiff::Real<2, double>", "DiFfRG::complex<double>" -> "cxReal<2, double>"};

$ADSpecializations = {<|"Suffix" -> "AD", "Replacements" -> $ADReplacements|>, <|"Suffix" -> "AD2", "Replacements" -> $AD2Replacements|>};

$PredefRegFunc = {"RB", "RF", "RBdot", "RFdot", "dq2RB", "dq2RF"};

$StandardKernelDefinitions = Map[FunKit`MakeCppFunction["Name" -> #, "Body" -> "return Regulator::" <> # <> "(k2, p2);", "Prefix" -> "static KOKKOS_INLINE_FUNCTION", "Suffix" -> "", "Parameters" -> {"k2", "p2"}]&, $PredefRegFunc];

CheckKey[kernel_Association, name_String, test_, msg_String] :=
    Module[{valid},
        If[Not @ KeyExistsQ[kernel, name],
            Message[MakeKernel::MissingKey, name];
            Return[False]
        ];
        If[Not @ test[kernel[name]],
            Message[MakeKernel::InvalidKey, name, msg];
            Return[False]
        ];
        Return[True];
    ];

KernelSpecQ[spec_Association] :=
    Module[{validKeys, validKeyTypes},
        validKeys = CheckKey[spec, "Name", StringQ[#] && StringLength[#] > 0&, "Cannot be empty"] && CheckKey[spec, "Integrator", StringQ[#] && StringLength[#] > 0&, "Cannot be empty"] && CheckKey[spec, "d", IntegerQ[#] && # >= 0&, "Must be an Integer >= 0"] && CheckKey[spec, "AD", BooleanQ, "Must be a Boolean"] && CheckKey[spec, "Device", MemberQ[{"Threads", "TBB", "GPU"}, #]&, "Must be Threads, TBB or GPU."] && CheckKey[spec, "Type", StringQ[#] && StringLength[#] > 0&, "Cannot be empty"];
        Return[validKeys];
    ];

GetStandardKernelDefinitions[] :=
    $StandardKernelDefinitions

(* Internal functions added here with Internal`*::usage *)

Options[MakeKernel] = {"Coordinates" -> {}, "CoordinateArguments" -> {}, "IntegrationVariables" -> {}, "KernelDefinitions" -> $StandardKernelDefinitions, "Regulator" -> "DiFfRG::PolynomialExpRegulator", "RegulatorOpts" -> {"", ""}, "KernelBody" -> "", "KernelReturnType" -> "auto", "KernelReturnTransform" -> Identity, "ConstantBody" -> "", "ConstantReturnType" -> "auto", "ConstantReturnTransform" -> Identity, "Parameters" -> {}, "Name" -> "", "d" -> -1, "Integrator" -> "", "AD" -> False, "ctype" -> "double", "Device" -> "TBB", "Type" -> "double", "SplitKernel" -> False, "SeparateLookups" -> False, "Decorator" -> "static KOKKOS_FUNCTION", "MatsubaraEven" -> False};

MakeKernel[__] :=
    (
        Message[MakeKernel::Invalid];
        Abort[]
    );

MakeKernel[kernelExpr_, OptionsPattern[]] :=
    MakeKernel @@ (Join[{kernelExpr, 0}, Thread[Rule @@ {#, OptionValue[MakeKernel, #]}]& @ Keys[Options[MakeKernel]]]);

MakeKernel[kernelExpr_, constExpr_, OptionsPattern[]] :=
    Module[{expr, const, exec, kernel, constant, kernelClass, kernelHeader, integratorHeader, integratorCpp, integratorTemplateParams, tparams = <|"Name" -> "...t", "Type" -> "auto&&", "Reference" -> False, "Const" -> False|>, kernelDefs = OptionValue["KernelDefinitions"], coordinates = OptionValue["Coordinates"], getArgs = OptionValue["CoordinateArguments"], intVariables = OptionValue["IntegrationVariables"], preArguments, regulator, params, adSpecs, explParamAD, arguments, outputPath, sources, returnType, returnTypePointer, spec, parameters, parametersKernel, matsubaraEvenTrait},
        spec = Association @@ Thread[Rule @@ {#, OptionValue[MakeKernel, #]}]& @ Keys[Options[MakeKernel]];
        If[Not @ KernelSpecQ[spec],
            Message[MakeKernel::InvalidSpec];
            Abort[]
        ];
        expr = kernelExpr;
        While[ListQ[expr], expr = Plus @@ expr];
(* Matsubara evenness: the finite-T integrator sums kernel(+xt)+kernel(-xt) over
   the Matsubara frequency (the last integration variable). If the kernel is
   GENUINELY even in that variable (kernel(+f0) == kernel(-f0)), the integrator
   can evaluate it ONCE and double the result -- a real saving, with NO change to
   the kernel body. We do NOT symbolically fold to the even part: for rational
   integrands the even part is a ratio-of-products that BLOATS the kernel 2-4x.
   Instead we leave the kernel untouched and only set the `matsubara_even` trait
   when evenness is verified symbolically. Note: fermionic dressings evaluated at
   f0-shifted arguments (e.g. ZQ[f0+p0, ...]) generally BREAK evenness, so most
   finite-T kernels are NOT even and keep the explicit two-call form. *)
        matsubaraEvenTrait = {};
        If[TrueQ[OptionValue["MatsubaraEven"]] && Length[OptionValue["IntegrationVariables"]] > 0,
            Module[{mVar, diff, isEven},
                mVar = Last[OptionValue["IntegrationVariables"]];
                mVar =
                    If[StringQ[mVar],
                        mVar
                        ,
                        mVar["Name"]
                    ];
                diff = expr - (expr /. s_Symbol /; SymbolName[s] === mVar :> -s);
                isEven = TrueQ[Quiet @ TimeConstrained[PossibleZeroQ[Simplify[diff]], 180, False]];
                If[isEven,
                    matsubaraEvenTrait = {"static constexpr bool matsubara_even = true;"};
                    ,
                    Message[MakeKernel::notEven, OptionValue["Name"], mVar];
                ];
            ]
        ];
        const = constExpr;
        While[ListQ[const], const = Plus @@ const];
        intVariables = FunKit`Private`prepParam /@ intVariables;
        intVariables = Map[Append[#, "Type" -> "double"]&, intVariables];
        getArgs = FunKit`Private`prepParam /@ getArgs;
        getArgs = Map[Append[#, "Type" -> "double"]&, getArgs];
        (********************************************************************)
        (* First, the kernel itself *)
        (********************************************************************)
        parametersKernel =
            Map[
                Which[
                    #["AD"] === True,
                        Merge[{#, <|"Type" -> "auto"|>}, Last]
                    ,
                    KeyFreeQ[#, "Type"],
                        Message[MakeKernel::MissingType, #["Name"]];
                        Merge[{#, <|"Type" -> "auto"|>}, Last]
                    ,
                    True,
                        #
                ]&
                ,
                spec["Parameters"]
            ];
        kernel =
            If[TrueQ[OptionValue["SplitKernel"]] || TrueQ[OptionValue["SeparateLookups"]],
                FunKit`MakeCppFunctionSplit[expr, "Name" -> "kernel", "Return" -> OptionValue["KernelReturnType"], "Suffix" -> "", "Prefix" -> "static KOKKOS_INLINE_FUNCTION", "Decorator" -> OptionValue["Decorator"], "SeparateLookups" -> OptionValue["SeparateLookups"], "Parameters" -> Join[intVariables, getArgs, parametersKernel], "Body" -> StringTemplate["using namespace DiFfRG;using namespace DiFfRG::compute;\n`1`"][OptionValue["KernelBody"]], "ReturnTransform" -> OptionValue["KernelReturnTransform"]]
                ,
                FunKit`MakeCppFunction[expr, "Name" -> "kernel", "Return" -> OptionValue["KernelReturnType"], "Suffix" -> "", "Prefix" -> "static KOKKOS_INLINE_FUNCTION", "Parameters" -> Join[intVariables, getArgs, parametersKernel], "Body" -> StringTemplate["using namespace DiFfRG;using namespace DiFfRG::compute;\n`1`"][OptionValue["KernelBody"]], "ReturnTransform" -> OptionValue["KernelReturnTransform"]]
            ];
        constant = FunKit`MakeCppFunction[constExpr, "Name" -> "constant", "Return" -> OptionValue["ConstantReturnType"], "Suffix" -> "", "Prefix" -> "static KOKKOS_INLINE_FUNCTION", "Parameters" -> Join[getArgs, parametersKernel], "Body" -> StringTemplate["using namespace DiFfRG;using namespace DiFfRG::compute;\n`1`"][OptionValue["ConstantBody"]], "ReturnTransform" -> OptionValue["ConstantReturnTransform"]];
        kernelClass = FunKit`MakeCppClass["TemplateTypes" -> {"_Regulator"}, "Name" -> OptionValue["Name"] <> "_kernel", "MembersPublic" -> Join[{"using Regulator = _Regulator;"}, matsubaraEvenTrait, {kernel, constant}], "MembersPrivate" -> kernelDefs];
        kernelHeader = FunKit`MakeCppHeader["Includes" -> {"DiFfRG/physics/interpolation.hh", "DiFfRG/physics/physics.hh"}, "Body" -> {"namespace DiFfRG {", kernelClass, StringTemplate["} using DiFfRG::`1`_kernel;"][spec["Name"]]}];
        (********************************************************************)
        (* Next, the corresponding class holding the map and get functions *)
        (********************************************************************)
        (*We set up lists of parameters for the map/get functions, depending on their AD setting*)
        parameters = spec["Parameters"];
        params = FunKit`Private`prepParam /@ parameters;
        params = First @ processParameters[params, $ADReplacements];
        arguments = StringRiffle[Map[#["Name"]&, params], ", "];
        getArgs = FunKit`Private`prepParam /@ getArgs;
        getArgs = Map[Append[#, "Type" -> "double"]&, getArgs];
        getArgs = First @ processParameters[getArgs, $ADReplacements];
        preArguments = StringRiffle[Map[#["Name"]&, getArgs], ", "];
        If[preArguments =!= "",
            preArguments = preArguments <> ", "
        ];
        (* Choose the execution space. Default is TBB, as only TBB is compatible with the FEM assemblers. *)
        exec =
            If[KeyFreeQ[spec, "Device"] || FreeQ[{"GPU", "Threads"}, spec["Device"]],
                "DiFfRG::TBB_exec"
                ,
                "DiFfRG::" <> spec["Device"] <> "_exec"
            ];
        integratorTemplateParams = TemplateParameterGeneration[spec];
        integratorTemplateParams = StringRiffle[integratorTemplateParams, ", "];
        returnType = spec["ctype"];
        returnTypePointer = StringTemplate["`1`*"][returnType];
        adSpecs =
            If[spec["AD"],
                Map[Merge[{#, <|"IntegratorTemplateParams" -> StringRiffle[TemplateParameterGeneration[spec, #["Replacements"]], ", "], "ReturnType" -> spec["ctype"] /. #["Replacements"], "Params" -> Last @ processParameters[params, #["Replacements"]]|>}, Last]&, $ADSpecializations]
                ,
                {}
            ];
        (* Now, we create the header which holds the class with the integrators and the map/get methods *)
        integratorHeader =
            FunKit`MakeCppHeader[
(* kernel.hh, NOT a forward declaration of <Name>_kernel. The integrator member below is
   Integrator_*<..., <Name>_kernel<Regulator>, ...>, and completing that type evaluates the
   integrator's kernel-trait detectors (`requires { requires K::matsubara_even; }` and friends). On
   an INCOMPLETE K every such detector is a substitution failure, i.e. silently false -- so a TU
   that had not seen kernel.hh built a DIFFERENT integrator class than one that had. That is an ODR
   violation, and it is not benign: flows.cc instantiates set_T() -> refresh_matsubara() with the
   traits off while the CT_*.cc TUs instantiate map() with them on, and the two disagree about
   which Matsubara rule is on the axis. It produced a wrong RHS (a factor ~2 in dtZA), not a
   diagnostic. Costs compile time in every consumer of <Name>.hh; correctness first. *)
                "Includes" -> {"DiFfRG/physics/integration.hh", "DiFfRG/physics/physics.hh", "DiFfRG/physics/interpolation.hh", "kernel.hh"}
                ,
                "Body" ->
                    {
                        "namespace DiFfRG {\n"
                        ,
                        FunKit`MakeCppClass[
                            "Name" -> StringTemplate["`Name`_integrator"][spec]
                            ,
                            "MembersPublic" ->
                                Join[
                                    {FunKit`MakeCppFunction["Name" -> StringTemplate["`Name`_integrator"][spec], "Parameters" -> {<|"Type" -> "DiFfRG::QuadratureProvider", "Reference" -> True, "Const" -> False, "Name" -> "quadrature_provider"|>, <|"Type" -> "DiFfRG::ConfigTree", "Reference" -> True, "Const" -> True, "Name" -> "config"|>}, "Body" -> None, "Return" -> ""], getRegulator[OptionValue["Regulator"], OptionValue["RegulatorOpts"]], StringTemplate["`1`<`2`> integrator;"][spec["Integrator"], integratorTemplateParams]}
                                    ,
                                    Map[StringTemplate["`1`<`2`> integrator_`3`;"][spec["Integrator"], #["IntegratorTemplateParams"], #["Suffix"]]&, adSpecs]
                                    ,
                                    Map[FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Body" -> None, "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> returnTypePointer, "Const" -> False, "Reference" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> #, "Const" -> True|>}, params]]&, coordinates]
                                    ,
                                    If[Length[coordinates] > 0,
                                            #
                                            ,
                                            {}
                                        ]& @ {FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Body" -> "return device::apply([&](const auto...t){return map(dest, coordinates, t...);}, args);", "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> "IT*", "Reference" -> False, "Const" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> "C", "Const" -> True|>, <|"Name" -> "args", "Type" -> "device::tuple<T...>", "Reference" -> True, "Const" -> True|>}], "Templates" -> {"IT", "C", "...T"}]}
                                    ,
                                    Flatten[
                                        Map[
                                            With[{ad = #},
                                                Map[FunKit`MakeCppFunction["Name" -> "map", "Return" -> "void", "Body" -> None, "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> StringTemplate["`1`*"][ad["ReturnType"]], "Const" -> False, "Reference" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> #, "Const" -> True|>}, ad["Params"]]]&, coordinates]
                                            ]&
                                            ,
                                            adSpecs
                                        ]
                                    ]
                                    ,
                                    {FunKit`MakeCppFunction["Name" -> "get", "Return" -> "void", "Body" -> None, "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> returnType, "Reference" -> True, "Const" -> False|>}, getArgs, params]], FunKit`MakeCppFunction["Name" -> "get", "Return" -> "void", "Body" -> "device::apply([&](const auto...t){get(dest, " <> preArguments <> "t...);}, args);", "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> "IT", "Reference" -> True, "Const" -> False|>}, getArgs, {<|"Name" -> "args", "Type" -> "device::tuple<T...>", "Reference" -> True, "Const" -> True|>}], "Templates" -> {"IT", "...T"}]}
                                    ,
                                    Map[FunKit`MakeCppFunction["Name" -> "get", "Return" -> "void", "Body" -> None, "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> #["ReturnType"], "Reference" -> True, "Const" -> False|>}, getArgs, #["Params"]]]&, adSpecs]
                                ]
                            ,
                            "MembersPrivate" -> {"DiFfRG::QuadratureProvider& quadrature_provider;"}
                        ]
                        ,
                        "}\nusing DiFfRG::" <> spec["Name"] <> "_integrator;"
                    }
            ];
        (* Finally, the code fo rall methods of the class. we will save them to different files, so they can all be compiled in separate units.*)
        integratorCpp["Constructor"] =
            FunKit`MakeCppBlock[
                "Includes" -> {"../kernel.hh"}
                ,
                "Body" ->
                    {
                        StringTemplate["#include \"../`Name`.hh\"\n"][spec]
                        ,
                        FunKit`MakeCppFunction[
                            "Name" -> StringTemplate["`Name`_integrator"][spec]
                            ,
                            "Class" -> StringTemplate["`Name`_integrator"][spec]
                            ,
                            "Suffix" ->
                                If[Length[adSpecs] > 0,
                                    ": integrator(quadrature_provider, config), " <> StringRiffle[Map[StringTemplate["integrator_`1`(quadrature_provider, config)"][#["Suffix"]]&, adSpecs], ", "] <> ", quadrature_provider(quadrature_provider)"
                                    ,
                                    ": integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)"
                                ]
                            ,
                            "Body" -> ""
                            ,
                            "Parameters" -> {<|"Type" -> "DiFfRG::QuadratureProvider", "Reference" -> True, "Const" -> False, "Name" -> "quadrature_provider"|>, <|"Type" -> "DiFfRG::ConfigTree", "Reference" -> True, "Const" -> True, "Name" -> "config"|>}
                            ,
                            "Return" -> ""
                        ]
                    }
            ];
        integratorCpp["CT", "get"] = FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> {StringTemplate["#include \"../`Name`.hh\"\n"][spec], FunKit`MakeCppFunction["Name" -> "get", "Class" -> StringTemplate["`Name`_integrator"][spec], "Body" -> StringTemplate["integrator.get(dest, `1` `2`);"][preArguments, arguments], "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> returnType, "Reference" -> True, "Const" -> False|>}, getArgs, params], "Return" -> "void"]}];
        integratorCpp["AD", "get"] = FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> Join[{StringTemplate["#include \"../`Name`.hh\"\n"][spec]}, Map[FunKit`MakeCppFunction["Name" -> "get", "Class" -> StringTemplate["`Name`_integrator"][spec], "Body" -> StringTemplate["integrator_`1`.get(dest, `2` `3`);"][#["Suffix"], preArguments, arguments], "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> #["ReturnType"], "Reference" -> True, "Const" -> False|>}, getArgs, #["Params"]], "Return" -> "void"]&, adSpecs]]];
        integratorCpp["CT", "map"] = Map[FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> {StringTemplate["#include \"../`Name`.hh\"\n"][spec], FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Class" -> StringTemplate["`Name`_integrator"][spec], "Body" -> StringTemplate["return integrator.map(dest, coordinates, `1`);"][arguments], "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> returnTypePointer, "Const" -> False, "Reference" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> #, "Const" -> True|>}, params]]}]&, coordinates];
        integratorCpp["AD", "map"] =
            Map[
                With[{coordinate = #},
                    FunKit`MakeCppBlock["Includes" -> {"../kernel.hh"}, "Body" -> Join[{StringTemplate["#include \"../`Name`.hh\"\n"][spec]}, Map[FunKit`MakeCppFunction["Name" -> "map", "Return" -> exec, "Class" -> spec["Name"] <> "_integrator", "Body" -> StringTemplate["return integrator_`1`.map(dest, coordinates, `2`);"][#["Suffix"], arguments], "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> StringTemplate["`1`*"][#["ReturnType"]], "Const" -> False, "Reference" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> coordinate, "Const" -> True|>}, #["Params"]]]&, adSpecs]]]
                ]&
                ,
                coordinates
            ];
        outputPath = FileNameJoin[flowDir, spec["Name"]];
        ExportCode[FileNameJoin[outputPath, spec["Name"] <> ".hh"], integratorHeader];
        ExportCode[FileNameJoin[outputPath, "kernel.hh"], kernelHeader];
        sources = {FileNameJoin[outputPath, "src", "constructor.cc"]};
        ExportCode[sources[[-1]], integratorCpp["Constructor"]];
        AppendTo[sources, FileNameJoin[outputPath, "src", "CT_get.cc"]];
        ExportCode[sources[[-1]], integratorCpp["CT", "get"]];
        Do[
            AppendTo[sources, FileNameJoin[outputPath, "src", StringTemplate["CT_map_`1`.cc"][i]]];
            ExportCode[sources[[-1]], integratorCpp["CT", "map"][[i]]]
            ,
            {i, 1, Length[coordinates]}
        ];
        If[spec["AD"],
            AppendTo[sources, FileNameJoin[outputPath, "src", "AD_get.cc"]];
            ExportCode[sources[[-1]], integratorCpp["AD", "get"]];
            Do[
                AppendTo[sources, FileNameJoin[outputPath, "src", StringTemplate["AD_map_`1`.cc"][i]]];
                ExportCode[sources[[-1]], integratorCpp["AD", "map"][[i]]]
                ,
                {i, 1, Length[coordinates]}
            ];
        ];
        sources = Map[StringReplace[#, outputPath -> StringTemplate["${CMAKE_CURRENT_SOURCE_DIR}/`Name`"][spec]]&, sources];
        If[Export[FileNameJoin[outputPath, "sources.m"], sources] === $Failed,
            Message[MakeKernel::exportFailed, FileNameJoin[outputPath, "sources.m"]];
            Abort[]
        ];
        Print["Please run UpdateFlows[] to export an up-to-date CMakeLists.txt"];
    ];

End[]

EndPackage[]
