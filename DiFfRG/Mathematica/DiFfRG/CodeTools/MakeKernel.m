(* ::Package:: *)

(* Exported symbols added here with SymbolName::usage *)

BeginPackage["DiFfRG`CodeTools`MakeKernel`"]

GetStandardKernelDefinitions::usage="GetStandardKernelDefinitions[] returns a list of standard kernel definitions used in DiFfRG."

MakeKernel::usage = "MakeKernel[kernel_Association, parameterList_List,integrandFlow_,constantFlow_:0., integrandDefinitions_String:\"\", constantDefinitions_String:\"\"]
Make a kernel from a given flow equation, parmeter list and kernel. The kernel must be a valid specification of an integration kernel.
This Function creates an integrator that evaluates (constantFlow + \[Integral]integrandFlow). One can prepend additional c++ definitions to the flow equation by using the integrandDefinitions and constantDefinitions parameters. 
These are prepended to the respective methods of the integration kernel, allowing one to e.g. define specific angles one needs for the flow code.";
MakeKernel::Invalid = "The given arguments are invalid. See MakeKernel::usage";
MakeKernel::InvalidSpec = "The given kernel specification is invalid.";
MakeKernel::InvalidSpec="The parameters given to MakeKernel are not valid.";
MakeKernel::MissingKey="The key \"`1`\" is missing.";
MakeKernel::InvalidKey="The key \"`1`\" is invalid: `2`";


Begin["`Private`"]

Needs["DiFfRG`CodeTools`Utils`"]

Needs["DiFfRG`CodeTools`Directory`"]

Needs["DiFfRG`CodeTools`Export`"]

Needs["DiFfRG`CodeTools`TemplateParameterGeneration`"]

Needs["DiFfRG`CodeTools`Regulator`"]

$ADReplacements = {"double" -> "autodiff::real", "DiFfRG::complex<double>" -> "cxreal"};

$PredefRegFunc={"RB","RF","RBdot","RFdot","dq2RB","dq2RF"};
$StandardKernelDefinitions=Map[
FunKit`MakeCppFunction["Name"->#,"Body"->"return Regulator::"<>#<>"(k2, p2);","Prefix"->"static KOKKOS_FORCEINLINE_FUNCTION","Suffix"->"","Parameters"->{"k2","p2"}]&,
$PredefRegFunc];


CheckKey[kernel_Association,name_String,test_,msg_String]:=Module[{valid},
    If[Not@KeyExistsQ[kernel,name],Message[MakeKernel::MissingKey,name];Return[False]];
    If[Not@test[kernel[name]],Message[MakeKernel::InvalidKey,name,msg];Return[False]];
    Return[True];
];

KernelSpecQ[spec_Association]:=Module[{validKeys,validKeyTypes},
    validKeys=CheckKey[spec,"Name",StringQ[#]&&StringLength[#]>0&,"Cannot be empty"]&&
        CheckKey[spec,"Integrator",StringQ[#]&&StringLength[#]>0&,"Cannot be empty"]&&
        CheckKey[spec,"d",IntegerQ[#]&&#>=0&,"Must be an Integer >= 0"]&&
        CheckKey[spec,"AD",BooleanQ,"Must be a Boolean"]&&
        CheckKey[spec,"Device",MemberQ[{"Threads","TBB","GPU"},#]&,"Must be Threads, TBB or GPU."]&&
        CheckKey[spec,"Type",StringQ[#]&&StringLength[#]>0&,"Cannot be empty"];
    Return[validKeys];
];

GetStandardKernelDefinitions[]:=$StandardKernelDefinitions

(* Internal functions added here with Internal`*::usage *)
Options[MakeKernel]={
"Coordinates"->{},
"CoordinateArguments"->{},
"IntegrationVariables"->{},
"KernelDefinitions"->$StandardKernelDefinitions,
"Regulator"->"DiFfRG::PolynomialExpRegulator",
"RegulatorOpts"->{"",""},
"KernelBody"->"",
"ConstantBody"->"",
"Parameters"->{},
"Name"->"",
"d"->-1,
"Integrator"->"",
"AD"->False,
"ctype"->"double",
"Device"->"TBB",
"Type"->"double"
};

MakeKernel[__]:=(Message[MakeKernel::Invalid];Abort[]);
MakeKernel[kernelExpr_,OptionsPattern[]]:=MakeKernel@@(Join[{kernelExpr,0},Thread[Rule@@{#,OptionValue[MakeKernel,#]}]&@Keys[Options[MakeKernel]]]);
MakeKernel[kernelExpr_,constExpr_,OptionsPattern[]]:=Module[{
    expr, const, exec, kernel, constant, kernelClass, kernelHeader, integratorHeader, 
    integratorCpp, integratorTemplateParams, integratorADTemplateParams,
    tparams=<|"Name"->"...t","Type"->"auto&&","Reference"->False,"Const"->False|>,
    kernelDefs=OptionValue["KernelDefinitions"], coordinates=OptionValue["Coordinates"],
    getArgs=OptionValue["CoordinateArguments"], intVariables=OptionValue["IntegrationVariables"], preArguments, regulator, params, paramsAD, explParamAD,
    arguments, outputPath, sources, returnType, returnTypeAD, returnTypePointer, returnTypePointerAD,
    spec,parameters,parametersKernel
},
spec=Association@@Thread[Rule@@{#,OptionValue[MakeKernel,#]}]&@Keys[Options[MakeKernel]];

If[Not@KernelSpecQ[spec],Message[MakeKernel::InvalidSpec];Abort[]];

expr=kernelExpr;
While[ListQ[expr],expr=Plus@@expr];
const=constExpr;
While[ListQ[const],const=Plus@@const];

intVariables=FunKit`Private`prepParam/@intVariables;
intVariables=Map[Append[#,"Type"->"double"]&,intVariables];

getArgs=FunKit`Private`prepParam/@getArgs;
getArgs=Map[Append[#,"Type"->"double"]&,getArgs];

(********************************************************************)
(* First, the kernel itself *)
(********************************************************************)

parametersKernel=Map[
        If[#["AD"]===True, Merge[{#, <|"Type" -> "auto"|>}, Last],#]&,
        spec["Parameters"]
        ];

kernel=FunKit`MakeCppFunction[
    expr,
    "Name"->"kernel",
    "Suffix"->"",
    "Prefix"->"static KOKKOS_FORCEINLINE_FUNCTION",
    "Parameters"->Join[intVariables,getArgs,parametersKernel],
    "Body"->StringTemplate["using namespace DiFfRG;using namespace DiFfRG::compute;\n`1`"][OptionValue["KernelBody"]]
];

constant=FunKit`MakeCppFunction[
    constExpr,
    "Name"->"constant",
    "Suffix"->"",
    "Prefix"->"static KOKKOS_FORCEINLINE_FUNCTION",
    "Parameters"->Join[getArgs,parametersKernel],
    "Body"->StringTemplate["using namespace DiFfRG;using namespace DiFfRG::compute;\n`1`"][OptionValue["ConstantBody"]]
];

kernelClass=FunKit`MakeCppClass[
    "TemplateTypes"->{"_Regulator"},
    "Name"->OptionValue["Name"]<>"_kernel",
    "MembersPublic"->{"using Regulator = _Regulator;",kernel,constant},
    "MembersPrivate"->kernelDefs
];

kernelHeader=FunKit`MakeCppHeader[
    "Includes"->{"DiFfRG/physics/interpolation.hh","DiFfRG/physics/physics.hh"},
    "Body" -> {"namespace DiFfRG {", kernelClass, StringTemplate["} using DiFfRG::`1`_kernel;"][spec["Name"]]}
];

(********************************************************************)
(* Next, the corresponding class holding the map and get functions *)
(********************************************************************)

(*We set up lists of parameters for the map/get functions, depending on their AD setting*)
parameters=spec["Parameters"];
params=FunKit`Private`prepParam/@parameters;
{params, paramsAD} = processParameters[params, $ADReplacements];
arguments=StringRiffle[Map[#["Name"]&,params],", "];

getArgs=FunKit`Private`prepParam/@getArgs;
getArgs=Map[Append[#,"Type"->"double"]&,getArgs];
getArgs = First @ processParameters[getArgs, $ADReplacements];
preArguments=StringRiffle[Map[#["Name"]&,getArgs],", "];
If[preArguments=!="",preArguments=preArguments<>", "];

(* Choose the execution space. Default is TBB, as only TBB is compatible with the FEM assemblers. *)
exec=If[KeyFreeQ[spec,"Device"]||FreeQ[{"GPU","Threads"},spec["Device"]],"DiFfRG::TBB_exec","DiFfRG::"<>spec["Device"]<>"_exec"];

integratorTemplateParams=TemplateParameterGeneration[spec];
integratorTemplateParams=StringRiffle[integratorTemplateParams,", "];
integratorADTemplateParams=TemplateParameterGeneration[spec, $ADReplacements];
integratorADTemplateParams=StringRiffle[integratorADTemplateParams,", "];
returnType = spec["ctype"];
returnTypePointer = StringTemplate["`1`*"][returnType];
returnTypeAD = spec["ctype"] /. $ADReplacements;
returnTypePointerAD = StringTemplate["`1`*"][returnTypeAD];

(* Now, we create the header which holds the class with the integrators and the map/get methods *)
integratorHeader=FunKit`MakeCppHeader[
    "Includes"->{"DiFfRG/physics/integration.hh","DiFfRG/physics/physics.hh","DiFfRG/physics/interpolation.hh"},
    "Body"->{StringTemplate["namespace DiFfRG { template<typename> class `Name`_kernel;\n"][spec],
    FunKit`MakeCppClass[
        "Name" -> StringTemplate["`Name`_integrator"][spec],
        "MembersPublic"->
            Join[
                {
                    FunKit`MakeCppFunction[
                        "Name" -> StringTemplate["`Name`_integrator"][spec],
                        "Parameters"->{<|"Type"->"DiFfRG::QuadratureProvider","Reference"->True,"Const"->False,"Name"->"quadrature_provider"|>,<|"Type"->"DiFfRG::JSONValue","Reference"->True,"Const"->True,"Name"->"json"|>},
                        "Body"->None,
                        "Return"->""
                    ],
                    getRegulator[OptionValue["Regulator"],OptionValue["RegulatorOpts"]],
                    StringTemplate["`1`<`2`> integrator;"][spec["Integrator"], integratorTemplateParams],
                    If[spec["AD"], StringTemplate["`1`<`2`> integrator_AD;"][spec["Integrator"], integratorADTemplateParams], "" ]
                },
            Map[
                FunKit`MakeCppFunction["Name"->"map","Return"->exec,"Body"->None,"Parameters"->Join[{<|"Name"->"dest","Type"->returnTypePointer,"Const"->False,"Reference"->False|>,<|"Name"->"coordinates","Reference"->True,"Type"->#,"Const"->True|>},params]]&,
            coordinates],
            If[Length[coordinates]>0,#,{}]&@{
                FunKit`MakeCppFunction["Name"->"map","Return"->exec,"Body"->"return device::apply([&](const auto...t){return map(dest, coordinates, t...);}, args);","Parameters"->Join[
                {<|"Name"->"dest","Type"->"IT*","Reference"->False,"Const"->False|>,
                <|"Name"->"coordinates","Reference"->True,"Type"->"C","Const"->True|>,
                <|"Name"->"args","Type"->"device::tuple<T...>","Reference"->True,"Const"->True|>}],
                "Templates"->{"IT","C","...T"}]
            },
            If[spec["AD"],#,{}]&@Map[
                FunKit`MakeCppFunction["Name"->"map","Return"->"void","Body"->None,"Parameters"->Join[{<|"Name"->"dest","Type"->returnTypePointerAD,"Const"->False,"Reference"->False|>,<|"Name"->"coordinates","Reference"->True,"Type"->#,"Const"->True|>},paramsAD]]&,
                coordinates],
            {
                FunKit`MakeCppFunction["Name"->"get","Return"->"void","Body"->None,"Parameters"->Join[{<|"Name"->"dest","Type"->returnType,"Reference"->True,"Const"->False|>},getArgs,params]],
                FunKit`MakeCppFunction["Name"->"get","Return"->"void","Body"->"device::apply([&](const auto...t){get(dest, "<>preArguments<>"t...);}, args);","Parameters"->Join[
                {<|"Name"->"dest","Type"->"IT","Reference"->True,"Const"->False|>},
                getArgs,
                {<|"Name"->"args","Type"->"device::tuple<T...>","Reference"->True,"Const"->True|>}],
                "Templates"->{ "IT","...T"}],
                If[spec["AD"],#,""]&@FunKit`MakeCppFunction["Name"->"get","Return"->"void","Body"->None,"Parameters"->Join[{<|"Name"->"dest","Type"->returnTypeAD,"Reference"->True,"Const"->False|>},getArgs,paramsAD]]
            }
        ]
    ,
    "MembersPrivate"->{"DiFfRG::QuadratureProvider& quadrature_provider;"}

    ],"}\nusing DiFfRG::"<>spec["Name"]<>"_integrator;"}
];

(* Finally, the code fo rall methods of the class. we will save them to different files, so they can all be compiled in separate units.*)

integratorCpp["Constructor"]=FunKit`MakeCppBlock[
    "Includes"->{"../kernel.hh"},
    "Body"->{
        StringTemplate["#include \"../`Name`.hh\"\n"][spec],
        FunKit`MakeCppFunction[
            "Name" -> StringTemplate["`Name`_integrator"][spec],
            "Class" -> StringTemplate["`Name`_integrator"][spec],
            "Suffix"-> If[spec["AD"],
                ": integrator(quadrature_provider, json), integrator_AD(quadrature_provider, json), quadrature_provider(quadrature_provider)",
                ": integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)"
            ],
            "Body"->"",
            "Parameters"->{<|"Type"->"DiFfRG::QuadratureProvider","Reference"->True,"Const"->False,"Name"->"quadrature_provider"|>,<|"Type"->"DiFfRG::JSONValue","Reference"->True,"Const"->True,"Name"->"json"|>},
            "Return"->""
        ]
    }
];
integratorCpp["CT","get"]=FunKit`MakeCppBlock[
    "Includes"->{"../kernel.hh"},
    "Body"->{
    StringTemplate["#include \"../`Name`.hh\"\n"][spec],
    FunKit`MakeCppFunction[
        "Name"->"get",
        "Class" -> StringTemplate["`Name`_integrator"][spec],
        "Body" -> StringTemplate["integrator.get(dest, `1` `2`);"][preArguments, arguments], 
        "Parameters"->Join[{<|"Name"->"dest","Type"->returnType,"Reference"->True,"Const"->False|>},getArgs,params],
        "Return"->"void"
    ]
}];
integratorCpp["AD","get"]=FunKit`MakeCppBlock[
    "Includes"->{"../kernel.hh"},
    "Body"->{
    StringTemplate["#include \"../`Name`.hh\"\n"][spec],
    FunKit`MakeCppFunction[
        "Name"->"get",
        "Class" -> StringTemplate["`Name`_integrator"][spec],
        "Body" -> StringTemplate["integrator_AD.get(dest, `1` `2`);"][preArguments, arguments], 
        "Parameters"->Join[{<|"Name"->"dest","Type"->returnTypeAD,"Reference"->True,"Const"->False|>},getArgs,paramsAD],
        "Return"->"void"
    ]
}];
integratorCpp["CT","map"]=Map[FunKit`MakeCppBlock[
    "Includes"->{"../kernel.hh"},
    "Body"->{
    StringTemplate["#include \"../`Name`.hh\"\n"][spec],
    FunKit`MakeCppFunction[
        "Name"->"map",
        "Return"->exec,
        "Class" -> StringTemplate["`Name`_integrator"][spec],
        "Body"->StringTemplate["return integrator.map(dest, coordinates, `1`);"][arguments],
        "Parameters"->Join[{<|"Name"->"dest","Type"->returnTypePointer,"Const"->False,"Reference"->False|>,<|"Name"->"coordinates","Reference"->True,"Type"->#,"Const"->True|>},params]
    ]
}]&,coordinates];
integratorCpp["AD","map"]=Map[FunKit`MakeCppBlock[
    "Includes"->{"../kernel.hh"},
    "Body"->{
    StringTemplate["#include \"../`Name`.hh\"\n"][spec],
    FunKit`MakeCppFunction[
        "Name"->"map",
        "Return"->exec,
        "Class"->spec["Name"]<>"_integrator",
        "Body"->StringTemplate["return integrator_AD.map(dest, coordinates, `1`);"][arguments],
        "Parameters"->Join[{<|"Name"->"dest","Type"->returnTypePointerAD,"Const"->False,"Reference"->False|>,<|"Name"->"coordinates","Reference"->True,"Type"->#,"Const"->True|>},paramsAD]
    ]
}]&,coordinates];

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
Print["Please run UpdateFlows[] to export an up-to-date CMakeLists.txt"];
];

End[]

EndPackage[]
