(* Exported symbols added here with SymbolName::usage *)

BeginPackage["DiFfRG`CodeTools`MakeKernel`"]

MakeKernel::usage = "MakeKernel[args] computes the kernel with given arguments.";

MakeKernel::Invalid = "The given arguments are invalid. See MakeKernel::usage";

MakeKernel::InvalidSpec = "The given kernel specification is invalid.";

Begin["`Private`"]

Needs["DiFfRG`CodeTools`Utils`"]

Needs["DiFfRG`CodeTools`Directory`"]

Needs["DiFfRG`CodeTools`Export`"]

Needs["DiFfRG`CodeTools`TemplateParameterGeneration`"]

Needs["DiFfRG`CodeTools`Regulator`"]

ClearAll[makeMapParams]
makeFuncParams[type_String, otherParams_List] := Join[
    {
        <|"Name" -> "dest", "Type" -> type, "Reference" -> True, "Const" -> False|>
    },
    otherParams
];
makeMapParams[type_String, coord_String, otherParams_List] := Join[
    {
        <|"Name" -> "dest", "Type" -> type, "Const" -> False (*, Reference -> False *)|>,
        <|"Name" -> "coordinates", "Reference" -> True, "Type" -> coord, "Const" -> True|>
    },
    otherParams
];

ClearAll[MakeKernel]

Options[MakeKernel] = {"Coordinates" -> {}, "IntegrationVariables" ->
     {}, "KernelDefinitions" -> $StandardKernelDefinitions, "Regulator" ->
     "DiFfRG::PolynomialExpRegulator", "RegulatorOpts" -> {"", ""}};


$ADReplacements = {"double" -> "autodiff::real", "complex" -> "complex<autodiff::real>"};

MakeKernel[__] :=
    (
        Message[MakeKernel::Invalid];
        Abort[]
    );

MakeKernel[kernelExpr_, spec_Association, parameters_List, OptionsPattern[
    ]] :=
    MakeKernel @@ (Join[{kernelExpr, 0, spec, parameters}, Thread[Rule @@ {#, OptionValue[MakeKernel, #]}]& @ Keys[Options[MakeKernel]]]);

MakeKernel[kernelExpr_, constExpr_, spec_Association, parameters_List,
     OptionsPattern[]] :=
    Module[{expr, const, kernel, constant, kernelClass, kernelHeader,
        integratorHeader, integratorCpp, integratorTemplateParams, integratorADTemplateParams,
        tparams = <|"Name" -> "...t", "Type" -> "auto&&", "Reference" -> False, "Const" -> False|>, 
        kernelDefs = OptionValue["KernelDefinitions"], coordinates = OptionValue["Coordinates"], 
        params, paramsAD, arguments, outputPath, sources, returnType, returnTypeAD, returnTypePointer, returnTypePointerAD},
        If[Not @ KernelSpecQ[spec],
            Message[MakeKernel::InvalidSpec];
            Abort[]
        ];
        expr = kernelExpr;
        While[ListQ[expr], expr = Plus @@ expr];
        const = constExpr;
        While[ListQ[const], const = Plus @@ const];
        kernel = FunKit`MakeCppFunction[expr, 
            "Name" -> "kernel", 
            "Suffix" -> "", 
            "Prefix" -> "static KOKKOS_FORCEINLINE_FUNCTION", 
            "Parameters" -> Join[OptionValue["IntegrationVariables"], parameters], 
            "Body" -> "using namespace DiFfRG;using namespace DiFfRG::compute;"
        ];
        constant = FunKit`MakeCppFunction[constExpr, 
            "Name" -> "constant",
            "Suffix" -> "", 
            "Prefix" -> "static KOKKOS_FORCEINLINE_FUNCTION", 
            "Parameters" -> parameters, 
            "Body" -> "using namespace DiFfRG;using namespace DiFfRG::compute;"
        ];
        kernelClass = FunKit`MakeCppClass[
            "TemplateTypes" -> {"_Regulator"}, 
            "Name" -> StringTemplate["`1`_kernel"][spec["Name"]],
            "MembersPublic" -> {"using Regulator = _Regulator;", kernel, constant}, 
            "MembersPrivate" -> kernelDefs
        ];
        kernelHeader = FunKit`MakeCppHeader[
            "Includes" -> {"DiFfRG/physics/utils.hh"}, 
            "Body" -> {"namespace DiFfRG {", kernelClass, StringTemplate["} using DiFfRG::`1`_kernel;"][spec["Name"]]}
        ];


        params = FunKit`Private`prepParam /@ parameters;
        {params, paramsAD} = processParameters[params, $ADReplacements];
        arguments = StringRiffle[Map[#["Name"]&, params], ", "];

        integratorTemplateParams = TemplateParameterGeneration[spec];
        integratorTemplateParams = StringRiffle[integratorTemplateParams, ", "];
        integratorADTemplateParams = TemplateParameterGeneration[spec, $ADReplacements];
        integratorADTemplateParams = StringRiffle[integratorADTemplateParams, ", "];
        returnType = spec["Type"];
        returnTypePointer = StringTemplate["`1`*"][returnType];
        returnTypeAD = spec["Type"] /. $ADReplacements;
        returnTypePointerAD = StringTemplate["`1`*"][returnTypeAD];

        integratorHeader =
            FunKit`MakeCppHeader[
                "Includes" -> {"DiFfRG/physics/interpolation.hh", "DiFfRG/physics/integration.hh",
                    "DiFfRG/physics/physics.hh"},
                "Body" -> {
                    "#include \"./kernel.hh\"\n",
                    FunKit`MakeCppClass[
                        "Name" -> StringTemplate["`1`_integrator"][spec["Name"]],
                        "MembersPublic" ->
                        Join[{
                            FunKit`MakeCppFunction["Name" -> StringTemplate["`1`_integrator"][spec["Name"]],
                                "Parameters" -> {
                                    <|
                                        "Type" -> "DiFfRG::QuadratureProvider",
                                        "Reference" -> True, "Const" -> False, 
                                        "Name" -> "quadrature_provider"
                                    |>, 
                                    <|
                                        "Type" -> "DiFfRG::JSONValue", 
                                        "Reference" -> True, 
                                        "Const" -> True, 
                                        "Name" -> "json"
                                    |>
                                }, 
                                "Body" -> None, 
                                "Return" -> ""
                            ],
                                        If[Length[coordinates] > 0,
                                            FunKit`MakeCppFunction[
                                                "Name" -> "map", 
                                                "Templates" -> {StringTemplate["NT=`1`"][returnType]}, 
                                                "Parameters" -> {tparams}, 
                                                "Body" -> StringTemplate[
(* If this formatting looks weird to you, you might not be so wrong. But to have the correct indentation in the C++ code, this needs to be this way, sorry :( )*)
"static_assert(std::is_same_v<NT, `1`> || std::is_same_v<NT, `2`>, \"Unknown type requested of `3`_integrator::get\");
if constexpr (std::is_same_v<NT, `1`>)
  map_CT(std::forward<decltype(t)>(t)...);
else if constexpr (std::is_same_v<NT, `2`>)
  map_AD(std::forward<decltype(t)>(t)...);"][returnType, returnTypeAD, spec["Name"]],
                                                 "Return" -> "void"]
                                            ,
                                            ""
                                        ],
                                        FunKit`FormatCode[getRegulator[OptionValue["Regulator"], OptionValue["RegulatorOpts"]]],
                                        StringTemplate["`1`<`2`> integrator;"][spec["Integrator"], integratorTemplateParams],
                                        If[spec["AD"],
                                            StringTemplate["`1`<`2`> integrator_AD;"][spec["Integrator"], integratorADTemplateParams],
                                            ""
                                        ]
                                    },
                                    Map[FunKit`MakeCppFunction[
                                        "Name" -> "map", "Return" -> "void", "Body" -> None, 
                                         "Parameters" -> makeMapParams[returnTypePointer, #, params]]&,
                                    coordinates],
                                    If[spec["AD"], 
                                        Map[FunKit`MakeCppFunction[
                                            "Name" -> "map", "Return" -> "void", "Body" -> None, 
                                            "Parameters" -> makeMapParams[returnTypePointerAD, #, paramsAD]]&, 
                                        coordinates], 
                                        {}
                                    ],
                                    {
                                        FunKit`MakeCppFunction["Name"
                                             -> "get", "Return" -> "void", "Body" -> None, "Parameters" -> 
                                            makeFuncParams[returnType, params]
                                        ],
                                        If[spec["AD"],
                                            FunKit`MakeCppFunction[
                                                "Name" -> "get", "Return" -> "void", "Body" -> None, 
                                                "Parameters" -> 
                                                makeFuncParams[returnTypeAD, paramsAD]
                                            ],
                                            ""
                                        ]
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
                "Includes" -> {StringTemplate["../`Name`.hh"][spec]},
                "Body" ->
                    {
                        FunKit`MakeCppFunction[
                            "Name" -> StringTemplate["`Name`_integrator"][spec],
                            "Class" -> StringTemplate["`Name`_integrator"][spec],
                            "Suffix" ->
                                If[spec["AD"],
                                    ": integrator(quadrature_provider, json), integrator_AD(quadrature_provider, json), quadrature_provider(quadrature_provider)",
                                    ": integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)"
                                ],
                            "Body" -> "",
                            "Parameters" -> {
                                <|
                                    "Type" -> "DiFfRG::QuadratureProvider",
                                    "Reference" -> True, 
                                    "Const" -> False, 
                                    "Name" -> "quadrature_provider"
                                |>, 
                                <|
                                    "Type" -> "DiFfRG::JSONValue", 
                                    "Reference" -> True, 
                                    "Const" -> True, 
                                    "Name" -> "json"
                                |>
                            },
                            "Return" -> ""
                        ]
                    }
            ];
        integratorCpp["CT", "get"] = 
            FunKit`MakeCppBlock[
                "Includes" ->{StringTemplate["../`Name`.hh"][spec]}, 
                "Body" -> 
                    {
                        FunKit`MakeCppFunction[
                            "Name" -> "get", 
                            "Class" -> StringTemplate["`Name`_integrator"][spec],
                            "Body" -> StringTemplate["integrator.get(dest, `1`);"][arguments], 
                            "Parameters" -> makeFuncParams[returnType, params], 
                            "Return" -> "void"
                        ]
                    }
            ];
        integratorCpp["AD", "get"] = FunKit`MakeCppBlock[
            "Includes" -> {StringTemplate["../`Name`.hh"][spec]}, 
            "Body" -> {FunKit`MakeCppFunction[
                "Name" -> "get", 
                "Class" -> StringTemplate["`Name`_integrator"][spec], 
                "Body" -> StringTemplate["integrator_AD.get(dest, `1`);"][arguments], 
                "Parameters" -> makeFuncParams[returnTypeAD, paramsAD], 
                "Return" -> "void"]
                }
            ];
        integratorCpp["CT", "map"] = Map[
            FunKit`MakeCppBlock[
                "Includes" -> {StringTemplate["../`Name`.hh"][spec]}, 
                "Body" -> {FunKit`MakeCppFunction[
                    "Name" -> "map", 
                    "Return" -> "void", 
                    "Class" -> StringTemplate["`Name`_integrator"][spec],
                    "Body" -> StringTemplate["integrator.map(dest, coordinates, `1`);"][arguments],
                    "Parameters" -> makeFuncParams[returnTypePointer, #, params]
                        ]
                    }]&, 
                coordinates
            ];
        integratorCpp["AD", "map"] = Map[FunKit`MakeCppBlock[
            "Includes" -> {StringTemplate["../`Name`.hh"][spec]}, 
            "Body" -> {FunKit`MakeCppFunction[
                "Name" -> "map", 
                "Return" -> "void", 
                "Class" -> StringTemplate["`Name`_integrator"][spec],
                "Body" -> StringTemplate["integrator_AD.map(dest, coordinates, `1`);"][arguments],
                "Parameters" -> makeFuncParams[returnTypePointerAD, #, paramsAD]
                        ]
                    }]&, 
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
            AppendTo[sources, FileNameJoin[outputPath, "src", StringTemplate["CT_map_`1`.cc"][i]] ];
            ExportCode[sources[[-1]], integratorCpp["CT", "map"][[i]]],
            {i, 1, Length[coordinates]}
        ];
        If[spec["AD"],
            AppendTo[sources, FileNameJoin[outputPath, "src", "AD_get.cc"]];
            ExportCode[sources[[-1]], integratorCpp["AD", "get"]];
            Do[
                AppendTo[sources, FileNameJoin[outputPath, "src", StringTemplate["AD_map_`1`.cc"][i]]];
                ExportCode[sources[[-1]], integratorCpp["AD", "map"][[i]]],
                {i, 1, Length[coordinates]}
            ];
        ];
        sources = Map[StringReplace[#, 
                        outputPath -> StringTemplate["${CMAKE_CURRENT_SOURCE_DIR}/`Name`"][spec]
                        ]&, 
                    sources];
        Export[FileNameJoin[outputPath, "sources.m"], sources];
        (* :!CodeAnalysis::BeginBlock:: *)
        (* :!CodeAnalysis::Disable::SuspiciousSessionSymbol:: *)
        Print["Please run UpdateFlows[] to export an up-to-date CMakeLists.txt"];
        (* :!CodeAnalysis::EndBlock:: *)
    ];

End[]

EndPackage[]

(* Internal functions added here with Internal`*::usage *)
