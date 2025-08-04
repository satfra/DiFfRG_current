(* Exported symbols added here with SymbolName::usage *)

MakeKernel::usage = "MakeKernel[args] computes the kernel with given arguments.";

MakeKernel::Invalid = "The given arguments are invalid. See MakeKernel::usage";

MakeKernel::InvalidSpec = "The given kernel specification is invalid.";

Begin["`Private`"]

ClearAll[MakeKernel]

ClearAll[appendDefaultAssociation]

appendDefaultAssociation[config_] :=
    Merge[{config, <|"Type" -> "double", "Reference" -> True, "Const"
         -> True|>}, First];

appendDefaultAssociationAD[config_] :=
    Merge[{config, <|"Type" -> "double", "Reference" -> True, "Const"
         -> True|>}, First];

Options[MakeKernel] = {"Coordinates" -> {}, "IntegrationVariables" ->
     {}, "KernelDefinitions" -> $StandardKernelDefinitions, "Regulator" ->
     "DiFfRG::PolynomialExpRegulator", "RegulatorOpts" -> {"", ""}};

$ADReplacements = {};

$ADReplacementsDirect = {"double" -> "autodiff::real", "complex" -> "complex<autodiff::real>"
    };

MakeKernel[__] :=
    (
        Message[MakeKernel::Invalid];
        Abort[]
    );

MakeKernel[kernelExpr_, spec_Association, parameters_List, OptionsPattern[
    ]] :=
    MakeKernel @@ (Join[{kernelExpr, 0, spec, parameters}, Thread[Rule
         @@ {#, OptionValue[MakeKernel, #]}]& @ Keys[Options[MakeKernel]]]);

MakeKernel[kernelExpr_, constExpr_, spec_Association, parameters_List,
     OptionsPattern[]] :=
    Module[{expr, const, kernel, constant, kernelClass, kernelHeader,
         integratorHeader, integratorCpp, integratorTemplateParams, integratorADTemplateParams,
         tparams = <|"Name" -> "...t", "Type" -> "auto&&", "Reference" -> False,
         "Const" -> False|>, kernelDefs = OptionValue["KernelDefinitions"], coordinates
         = OptionValue["Coordinates"], regulator, params, paramsAD, i, arguments,
         outputPath, sources, appendDefaultAssociation},
        If[Not @ KernelSpecQ[spec],
            Message[MakeKernel::InvalidSpec];
            Abort[]
        ];
        expr = kernelExpr;
        While[ListQ[expr], expr = Plus @@ expr];
        const = constExpr;
        While[ListQ[const], const = Plus @@ const];
        kernel = FunKit`MakeCppFunction[expr, "Name" -> "kernel", "Suffix"
             -> "", "Prefix" -> "static KOKKOS_FORCEINLINE_FUNCTION", "Parameters"
             -> Join[OptionValue["IntegrationVariables"], parameters], "Body" -> 
            "using namespace DiFfRG;using namespace DiFfRG::compute;"];
        constant = FunKit`MakeCppFunction[constExpr, "Name" -> "constant",
             "Suffix" -> "", "Prefix" -> "static KOKKOS_FORCEINLINE_FUNCTION", "Parameters"
             -> parameters, "Body" -> "using namespace DiFfRG;using namespace DiFfRG::compute;"
            ];
        kernelClass = FunKit`MakeCppClass["TemplateTypes" -> {"_Regulator"
            }, "Name" -> spec["Name"] <> "_kernel", "MembersPublic" -> {"using Regulator = _Regulator;",
             kernel, constant}, "MembersPrivate" -> kernelDefs];
        kernelHeader = FunKit`MakeCppHeader["Includes" -> {"DiFfRG/physics/utils.hh"
            }, "Body" -> {"namespace DiFfRG {", kernelClass, "} using DiFfRG::" <>
             spec["Name"] <> "_kernel;"}];
        params = FunKit`Private`prepParam /@ parameters;
        appendDefaultAssociation[params];
        paramsAD = params;
        paramsAD["Type"] = paramsAD["Type"] /. $ADReplacements;
        arguments = StringRiffle[Map[#["Name"]&, params], ", "];
        integratorTemplateParams = {};
        If[KeyExistsQ[spec, "d"] && spec["d"] =!= None,
            AppendTo[integratorTemplateParams, ToString[spec["d"]]]
        ];
        If[KeyExistsQ[spec, "Type"],
            AppendTo[integratorTemplateParams, ToString[spec["Type"]]
                ]
            ,
            AppendTo[integratorTemplateParams, "double"]
        ];
        AppendTo[integratorTemplateParams, spec["Name"] <> "_kernel<Regulator>"
            ];
        AppendTo[
            integratorTemplateParams
            ,
            If[KeyFreeQ[spec, "Device"] || FreeQ[{"GPU", "Threads"}, 
                spec["Device"]],
                "DiFfRG::TBB_exec"
                ,
                "DiFfRG::" <> spec["Device"] <> "_exec"
            ]
        ];
        integratorTemplateParams = StringRiffle[integratorTemplateParams,
             ", "];
        integratorADTemplateParams = {};
        If[KeyExistsQ[spec, "d"] && spec["d"] =!= None,
            AppendTo[integratorADTemplateParams, ToString[spec["d"]]]
                
        ];
        If[KeyExistsQ[spec, "Type"],
            AppendTo[integratorADTemplateParams, StringReplace[ToString[
                spec["Type"]], $ADReplacementsDirect]]
            ,
            AppendTo[integratorADTemplateParams, "autodiff::real"]
        ];
        AppendTo[integratorADTemplateParams, spec["Name"] <> "_kernel<Regulator>"
            ];
        AppendTo[
            integratorADTemplateParams
            ,
            If[KeyFreeQ[spec, "Device"] || FreeQ[{"GPU", "Threads"}, 
                spec["Device"]],
                "DiFfRG::TBB_exec"
                ,
                "DiFfRG::" <> spec["Device"] <> "_exec"
            ]
        ];
        integratorADTemplateParams = StringRiffle[integratorADTemplateParams,
             ", "];
        integratorHeader =
            FunKit`MakeCppHeader[
                "Includes" -> {"DiFfRG/physics/interpolation.hh", "DiFfRG/physics/integration.hh",
                     "DiFfRG/physics/physics.hh"}
                ,
                "Body" ->
                    {
                        "#include \"./kernel.hh\"\n"
                        ,
                        FunKit`MakeCppClass[
                            "Name" -> spec["Name"] <> "_integrator"
                            ,
                            "MembersPublic" ->
                                Join[
                                    {
                                        FunKit`MakeCppFunction["Name"
                                             -> spec["Name"] <> "_integrator", "Parameters" -> {<|"Type" -> "DiFfRG::QuadratureProvider",
                                             "Reference" -> True, "Const" -> False, "Name" -> "quadrature_provider"
                                            |>, <|"Type" -> "DiFfRG::JSONValue", "Reference" -> True, "Const" -> 
                                            True, "Name" -> "json"|>}, "Body" -> None, "Return" -> ""]
                                        ,
                                        If[Length[coordinates] > 0,
                                            FunKit`MakeCppFunction["Name"
                                                 -> "map", "Templates" -> {"NT=double"}, "Parameters" -> {tparams}, "Body"
                                                 -> "static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>, \"Unknown type requested of "
                                                 <> spec["Name"] <> "_integrator::get\");
if constexpr (std::is_same_v<NT, double>)
  map_CT(std::forward<decltype(t)>(t)...);
else if constexpr (std::is_same_v<NT, autodiff::real>)
  map_AD(std::forward<decltype(t)>(t)...);",
                                                 "Return" -> "void"]
                                            ,
                                            ""
                                        ]
                                        ,
                                        getRegulator[OptionValue["Regulator"
                                            ], OptionValue["RegulatorOpts"]]
                                        ,
                                        spec["Integrator"] <> "<" <> 
                                            integratorTemplateParams <> "> integrator;"
                                        ,
                                        If[spec["AD"],
                                            spec["Integrator"] <> "<"
                                                 <> integratorADTemplateParams <> "> integrator_AD;"
                                            ,
                                            ""
                                        ]
                                    }
                                    ,
                                    Map[FunKit`MakeCppFunction["Name"
                                         -> "map", "Return" -> "void", "Body" -> None, "Parameters" -> Join[{
                                        <|"Name" -> "dest", "Type" -> "double*", "Const" -> False|>, <|"Name"
                                         -> "coordinates", "Reference" -> True, "Type" -> #, "Const" -> True|>
                                        }, params]]&, coordinates]
                                    ,
                                    If[spec["AD"],
                                            #
                                            ,
                                            {}
                                        ]& @ Map[FunKit`MakeCppFunction[
                                            "Name" -> "map", "Return" -> "void", "Body" -> None, "Parameters" -> 
                                            Join[{<|"Name" -> "dest", "Type" -> "autodiff::real*", "Const" -> False
                                            |>, <|"Name" -> "coordinates", "Reference" -> True, "Type" -> #, "Const"
                                             -> True|>}, paramsAD]]&, coordinates]
                                    ,
                                    {
                                        FunKit`MakeCppFunction["Name"
                                             -> "get", "Return" -> "void", "Body" -> None, "Parameters" -> Join[{
                                            <|"Name" -> "dest", "Type" -> "double", "Reference" -> True, "Const" 
                                            -> False|>}, params]]
                                        ,
                                        If[spec["AD"],
                                                #
                                                ,
                                                ""
                                            ]& @ FunKit`MakeCppFunction[
                                                "Name" -> "get", "Return" -> "void", "Body" -> None, "Parameters" -> 
                                                Join[{<|"Name" -> "dest", "Type" -> "autodiff::real", "Reference" -> 
                                                True, "Const" -> False|>}, paramsAD]]
                                    }
                                ]
                            ,
                            "MembersPrivate" -> {"DiFfRG::QuadratureProvider& quadrature_provider;"
                                }
                        ]
                    }
            ];
        (*Print[integratorHeader];*)
        integratorCpp["Constructor"] =
            FunKit`MakeCppBlock[
                "Includes" -> {"../" <> spec["Name"] <> ".hh"}
                ,
                "Body" ->
                    {
                        FunKit`MakeCppFunction[
                            "Name" -> spec["Name"] <> "_integrator"
                            ,
                            "Class" -> spec["Name"] <> "_integrator"
                            ,
                            "Suffix" ->
                                ": integrator(quadrature_provider, json), "
                                     <>
                                    If[spec["AD"],
                                        "integrator_AD(quadrature_provider, json), "
                                            
                                        ,
                                        ""
                                    ] <> "quadrature_provider(quadrature_provider)"
                                        
                            ,
                            "Body" -> ""
                            ,
                            "Parameters" -> {<|"Type" -> "DiFfRG::QuadratureProvider",
                                 "Reference" -> True, "Const" -> False, "Name" -> "quadrature_provider"
                                |>, <|"Type" -> "DiFfRG::JSONValue", "Reference" -> True, "Const" -> 
                                True, "Name" -> "json"|>}
                            ,
                            "Return" -> ""
                        ]
                    }
            ];
        integratorCpp["CT", "get"] = FunKit`MakeCppBlock["Includes" ->
             {"../" <> spec["Name"] <> ".hh"}, "Body" -> {FunKit`MakeCppFunction[
            "Name" -> "get", "Class" -> spec["Name"] <> "_integrator", "Body" -> 
            "integrator.get(dest, " <> arguments <> ");", "Parameters" -> Join[{<|
            "Name" -> "dest", "Type" -> "double", "Reference" -> True, "Const" ->
             False|>}, params], "Return" -> "void"]}];
        integratorCpp["AD", "get"] = FunKit`MakeCppBlock["Includes" ->
             {"../" <> spec["Name"] <> ".hh"}, "Body" -> {FunKit`MakeCppFunction[
            "Name" -> "get", "Class" -> spec["Name"] <> "_integrator", "Body" -> 
            "integrator_AD.get(dest, " <> arguments <> ");", "Parameters" -> Join[
            {<|"Name" -> "dest", "Type" -> "autodiff::real", "Reference" -> True,
             "Const" -> False|>}, paramsAD], "Return" -> "void"]}];
        integratorCpp["CT", "map"] = Map[FunKit`MakeCppBlock["Includes"
             -> {"../" <> spec["Name"] <> ".hh"}, "Body" -> {FunKit`MakeCppFunction[
            "Name" -> "map", "Return" -> "void", "Class" -> spec["Name"] <> "_integrator",
             "Body" -> "integrator.map(dest, coordinates, " <> arguments <> ");",
             "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> "double*", "Const"
             -> False|>, <|"Name" -> "coordinates", "Reference" -> True, "Type" ->
             #, "Const" -> True|>}, params]]}]&, coordinates];
        integratorCpp["AD", "map"] = Map[FunKit`MakeCppBlock["Includes"
             -> {"../" <> spec["Name"] <> ".hh"}, "Body" -> {FunKit`MakeCppFunction[
            "Name" -> "map", "Return" -> "void", "Class" -> spec["Name"] <> "_integrator",
             "Body" -> "integrator_AD.map(dest, coordinates, " <> arguments <> ");",
             "Parameters" -> Join[{<|"Name" -> "dest", "Type" -> "autodiff::real*",
             "Const" -> False|>, <|"Name" -> "coordinates", "Reference" -> True, 
            "Type" -> #, "Const" -> True|>}, paramsAD]]}]&, coordinates];
        outputPath = flowDir <> spec["Name"] <> "/";
        ExportCode[outputPath <> spec["Name"] <> ".hh", integratorHeader
            ];
        ExportCode[outputPath <> "kernel.hh", kernelHeader];
        sources = {outputPath <> "src/constructor.cc"};
        ExportCode[sources[[-1]], integratorCpp["Constructor"]];
        AppendTo[sources, outputPath <> "src/CT_get.cc"];
        ExportCode[sources[[-1]], integratorCpp["CT", "get"]];
        Do[
            AppendTo[sources, outputPath <> "src/CT_map_" <> ToString[
                i] <> ".cc"];
            ExportCode[sources[[-1]], integratorCpp["CT", "map"][[i]]
                ]
            ,
            {i, 1, Length[coordinates]}
        ];
        If[spec["AD"],
            AppendTo[sources, outputPath <> "src/AD_get.cc"];
            ExportCode[sources[[-1]], integratorCpp["AD", "get"]];
            Do[
                AppendTo[sources, outputPath <> "src/AD_map_" <> ToString[
                    i] <> ".cc"];
                ExportCode[sources[[-1]], integratorCpp["AD", "map"][[
                    i]]]
                ,
                {i, 1, Length[coordinates]}
            ];
        ];
        sources = Map[StringReplace[#, outputPath -> "${CMAKE_CURRENT_SOURCE_DIR}/"
             <> spec["Name"] <> "/"]&, sources];
        Export[outputPath <> "sources.m", sources];
        Message[MakeKernel::info, "Please run UpdateFlows[] to export an up-to-date CMakeLists.txt"
            ];
    ];

End[]
