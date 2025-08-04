(* ::Package:: *)

(* ::Title:: *)

(*DiFfRG: Code Tools package*)

(* ::Chapter::Closed:: *)

(*Package Setup*)

(* ::Section::Closed:: *)

(*Setup and exports*)

BeginPackage["DiFfRG`CodeTools`"];

Unprotect["DiFfRG`CodeTools`*"];

ClearAll["DiFfRG`CodeTools`*"];

ClearAll["DiFfRG`CodeTools`Private`*"];

UpdateFlows::usage = "";

FlowKernel::usage = "FlowKernel[expr_,name_String,NT_String:\"auto\",addprefix_String:\"\"]
Makes an equation into a lambda expression - of limited usefulness, but can be used together with LoopIntegrals::integrate and similar functions.";

ExportCode::usage = "ExportCode[fileName_String,expression_String]
Writes the given expression to disk and runs clang-format on it.
";

CodeForm::usage = "CodeForm[expr_]
Obtain properly formatted and processed C++ code from an expression.";

SetCppNames::usage = "SetCppNames[rules___]
Set additional replacement rules used when invoking CodeForm[expr].

Example Call: SetCppNames[\"k\"->\"k_bosonic\", \"Arccos(\"->\"std::arccos(\"]";

JuliaForm::usage = "CodeForm[expr_]
Obtain properly formatted and processed Julia code from an expression.";

UnicodeClip::usage = "UnicodeClip[expr_String]
Copy a string as unicode into the clipboard. Useful when exporting to Julia.";

MakeCMakeFile::usage = "MakeCMakeFile[kernels_List]
Creates a CMakeLists.txt inside the 'flowDir' which you can set using SetFlowDir[dir_String]. This CMake file contains references to all kernels specified in the List 'kernels'. Make sure you have created all kernels before compiling!
If so, simply add the flow directory in the parent directory of the flow directory: \n add_subdirectory(flows) \n Note that this CMakeLists.txt exports its source files into parent scope as $flow_sources
Thus, to compile the flows, simply add them as source files:
    add_executable(QCD QCD.cc ${flow_sources})";

SetFlowDirectory::usage = "SetFlowDirectory[dir]
Set the current flow directory, i.e. where all generated files are saved. Default is ./flows/";

ShowFlowDirectory::usage = "ShowFlowDirectory[]
Show the current flow directory, i.e. where all generated files are saved. Default is ./flows/";

MakeFlowClass::usage = "MakeFlowClass[name_String,kernels_List]
This creates a file flows.hh inside the flow directory, containing a class with the specified name, as well as several other files. All defined kernels have a corresponding integrator object in this class.
Automatically calls MakeCMakeFile with the passed list of kernels.";

MakeFlowClassFiniteT::usage = "MakeFlowClassFiniteT[name_String,kernels_List]
This creates a file flows.hh inside the flow directory, containing a class with the specified name, as well as several other files. All defined kernels have a corresponding integrator object in this class.
Automatically calls MakeCMakeFile wiht the passed list of kernels.";

(* MakeKernel::usage = "MakeKernel[kernel_Association, parameterList_List,integrandFlow_,constantFlow_:0., integrandDefinitions_String:\"\", constantDefinitions_String:\"\"]
Make a kernel from a given flow equation, parmeter list and kernel. The kernel must be a valid specification of an integration kernel.
This Function creates an integrator that evaluates (constantFlow + \[Integral]integrandFlow). One can prepend additional c++ definitions to the flow equation by using the integrandDefinitions and constantDefinitions parameters. 
These are prepended to the respective methods of the integration kernel, allowing one to e.g. define specific angles one needs for the flow code."; *)

DeclareSymmetricPoints4DP4::usage = "DeclareSymmetricPoints4DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 4D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints4DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints4DP3::usage = "DeclareSymmetricPoints4DP3[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 4D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints4DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints3DP4::usage = "DeclareSymmetricPoints3DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 3D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints3DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints3DP3::usage = "DeclareSymmetricPoints3DP3[]
Obtain C++ code declaring angles for a three-point symmetric configuration in 3D.
The angles will have the names cosp1q, cosp2q and cosp3q.
DeclareSymmetricPoints3DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints2DP4::usage = "DeclareSymmetricPoints2DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 2D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints2DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints2DP3::usage = "DeclareSymmetricPoints2DP3[]
Obtain C++ code declaring angles for a three-point symmetric configuration in 2D.
The angles will have the names cosp1q, cosp2q and cosp3q.
DeclareSymmetricPoints2DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareAnglesP34Dpqr::usage = "DeclareAnglesP34Dpqr[q,p,r]
Obtain C++ code declaring angles for the angles in a full three-point function in 4D.
The angles will have the names cospq and cosqr, where q,p,r are replaced by the given Symbol names and ordered alphabetically.
DeclareAnglesP34Dpqr[q,p,r,computeType]
Set the type of the declared C++ variables (should be double or float).";

SafeFiniteTFunctions::usage = "";

ShowTypes::usage = "ShowTypes[]
Show a list of all types available for use in a parameter list to make a kernel.";

AddCodeOptimizeFunctions::usage = "AddCodeOptimizeFunctions[func1[__], func2[__],...]
Add the functions func1, func2, ... to the functions to be optimized out, i.e. calculated only once in auto-generated kernel code.";

ShowCodeOptimizeFunctions::usage = "ShowCodeOptimizeFunctions[]
Show a list of all functions which DiFfRG will try to optimize out when generating C++ code.";

ClearCodeOptimizeFunctions::usage = "ClearCodeOptimizeFunctions[]
Remove all optimization functions from the internal list";

UseKernelOptimizations::usage = "UseKernelOptimizations[True]
Turn on code optimizations in auto-generation of C++ code.
UseKernelOptimizations[False]
Turn off code optimizations in auto-generation of C++ code.
";

SetKernelDefinitions::usage = "SetKernelDefinitions[definitionCode_String] sets the kernel definitions to definitionCode.
SetKernelDefinitions[] resets the kernel definitions to the standard.";

ShowKernelDefinitions::usage = "ShowKernelDefinitions[]
Show the currently specified kernel definitions code.";

AddParameterType::usage = "AddParameterType[name,cppType,cppTypeAD,Reference,computeTypeName]
Add a recognized parameter to the list of useable kernel parameter types.
";

SetCodeParser::usage = "";

$DiFfRGDirectory = SelectFirst[Join[{FileNameJoin[{$UserBaseDirectory,
   "Applications", "DiFfRG"}], FileNameJoin[{$BaseDirectory, "Applications",
   "DiFfRG"}], FileNameJoin[{$InstallationDirectory, "AddOns", "Applications",
   "DiFfRG"}], FileNameJoin[{$InstallationDirectory, "AddOns", "Packages",
   "DiFfRG"}], FileNameJoin[{$InstallationDirectory, "AddOns", "ExtraPackages",
   "DiFfRG"}]}, Select[$Path, StringContainsQ[#, "DiFfRG"]&]], DirectoryQ[
  #]&] <> "/";

Get[FileNameJoin[{$DiFfRGDirectory, "modules", "MakeKernel.m"}]]

Begin["`Private`"];

(* ::Chapter::Closed:: *)

(*Safe replacements for finite T*)

SafeFiniteTFunctions[expr_, T_] :=
  Module[{a},
    expr //. {Tanh[a_ / (2 T)] :> Symbol["TanhFiniteT"][a, T], Tanh[a_
       / T] :> Symbol["TanhFiniteT"][a, 2 T], Tanh[a_ / (2 T)] ^ n_ :> Symbol[
      "TanhFiniteT"][a, T] ^ n, Tanh[a_ / T] ^ n_ :> Symbol["TanhFiniteT"][
      a, 2 T] ^ n, Coth[a_ / (2 T)] :> Symbol["CothFiniteT"][a, T], Coth[a_
       / T] :> Symbol["CothFiniteT"][a, 2 T], Coth[a_ / (2 T)] ^ n_ :> Symbol[
      "CothFiniteT"][a, T] ^ n, Coth[a_ / T] ^ n_ :> Symbol["CothFiniteT"][
      a, 2 T] ^ n, Csch[a_ / (2 T)] :> Symbol["CschFiniteT"][a, T], Csch[a_
       / T] :> Symbol["CschFiniteT"][a, 2 T], Csch[a_ / (2 T)] ^ n_ :> Symbol[
      "CschFiniteT"][a, T] ^ n, Csch[a_ / T] ^ n_ :> Symbol["CschFiniteT"][
      a, 2 T] ^ n, Sech[a_ / (2 T)] :> Symbol["SechFiniteT"][a, T], Sech[a_
       / T] :> Symbol["SechFiniteT"][a, 2 T], Sech[a_ / (2 T)] ^ n_ :> Symbol[
      "SechFiniteT"][a, T] ^ n, Sech[a_ / T] ^ n_ :> Symbol["SechFiniteT"][
      a, 2 T] ^ n}
  ];

(* ::Chapter::Closed:: *)

(*General Definitions and structural methods*)

(* ::Section::Closed:: *)

(*Folder setup*)

(* ::Input::Initialization:: *)

flowName = "flows";

flowDir :=
  If[$Notebooks,
      NotebookDirectory[]
      ,
      Directory[]
    ] <> flowName <> "/";

SetFlowName[name_String] :=
  Module[{},
    flowName = name
  ]

ShowFlowDirectory[] :=
  Print[                   "\!\(\*
StyleBox[\"DiFfRG\",\nFontWeight->\"Bold\"]\)\!\(\*
StyleBox[\" \",\nFontWeight->\"Bold\"]\)\!\(\*
StyleBox[\"CodeTools\",\nFontWeight->\"Bold\"]\)\!\(\*
StyleBox[\":\",\nFontWeight->\"Bold\"]\) Flow output directory is set to \n        "
     <> flowDir <> "\nThis can be modified by using \!\(\*
StyleBox[\"SetFlowName\",\nFontColor->RGBColor[1, 0.5, 0]]\)[\"YourNewName\"]"
    ]

ShowFlowDirectory[]

(* ::Section:: *)

(*Safety Checks*)

(* ::Section::Closed:: *)

(*Momentum Configurations*)

(* ::Subsection::Closed:: *)

(*4D*)

DeclareAnglesP34Dpqr[q_, p_, r_, computeType_String:"double"] :=
  Module[{vec4, Vectorp, Vectorr, Vectorq, cos, Resultcospq, Resultcosqr,
     namecospq, namecosqr, code, file},
    vec4[\[CapitalTheta]1_, \[CapitalTheta]2_, \[Phi]_] := {Cos[\[CapitalTheta]1
      ], Sin[\[CapitalTheta]1] Cos[\[CapitalTheta]2], Sin[\[CapitalTheta]1]
       Sin[\[CapitalTheta]2] Cos[\[Phi]], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2
      ] Sin[\[Phi]]};
    SetAttributes[cos, Orderless];
    Vectorp = vec4[0, 0, 0];
    Vectorr = vec4[ArcCos[cos[p, r]], 0, 0];
    Vectorq = vec4[ArcCos[Symbol["cos1"]], ArcCos[Symbol["cos2"]], Symbol[
      "phi"]];
    Resultcospq = Vectorq . Vectorp //. cos[a_, b_] :> Symbol["cos" <>
       ToString[a] <> ToString[b]] // FullSimplify;
    Resultcosqr = Vectorq . Vectorr //. cos[a_, b_] :> Symbol["cos" <>
       ToString[a] <> ToString[b]] // FullSimplify;
    namecospq = cos[p, q] //. cos[a_, b_] :> Symbol["cos" <> ToString[
      a] <> ToString[b]];
    namecosqr = cos[q, r] //. cos[a_, b_] :> Symbol["cos" <> ToString[
      a] <> ToString[b]];
    code = "const " <> computeType <> " " <> ToString[namecospq] <> " = "
       <> CodeForm[Resultcospq, computeType] <> ";\n" <> "const " <> computeType
       <> " " <> ToString[namecosqr] <> " = " <> CodeForm[Resultcosqr, computeType
      ] <> ";";
    Return[code];
  ];

(* ::Input::Initialization:: *)

DeclareSymmetricPoints4DP4[computeType_String:"double"] :=
  Module[{vec4, sanity, Vectors4DSP4, SymmetricPoint4DP4cosp1, SymmetricPoint4DP4cosp2,
     SymmetricPoint4DP4cosp3, SymmetricPoint4DP4cosp4, SymmetricPoint4DP4Code
    },
    vec4[\[CapitalTheta]1_, \[CapitalTheta]2_, \[Phi]_] := {Cos[\[CapitalTheta]1
      ], Sin[\[CapitalTheta]1] Cos[\[CapitalTheta]2], Sin[\[CapitalTheta]1]
       Sin[\[CapitalTheta]2] Cos[\[Phi]], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2
      ] Sin[\[Phi]]};
    Vectors4DSP4 = {vec4[\[Pi] / 2, 0, 0], vec4[\[Pi] / 2, ArcCos[-(1
      /3)], 0], vec4[\[Pi] / 2, ArcCos[-(1/3)], 1 (2 \[Pi]) / 3], vec4[\[Pi]
       / 2, ArcCos[-(1/3)], 2 (2 \[Pi]) / 3]};
    sanity = Map[Vectors4DSP4[[#[[1]]]] . Vectors4DSP4[[#[[2]]]]&, Subsets[
      {1, 2, 3, 4}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/3) // N)&],
      Print["Sanity check failed!"];
      Abort[];
    ];
    SymmetricPoint4DP4cosp1 = vec4[ArcCos[Symbol["cos1"]], ArcCos[Symbol[
      "cos2"]], Symbol["phi"]] . Vectors4DSP4[[1]] // FullSimplify;
    SymmetricPoint4DP4cosp2 = vec4[ArcCos[Symbol["cos1"]], ArcCos[Symbol[
      "cos2"]], Symbol["phi"]] . Vectors4DSP4[[2]] // FullSimplify;
    SymmetricPoint4DP4cosp3 = vec4[ArcCos[Symbol["cos1"]], ArcCos[Symbol[
      "cos2"]], Symbol["phi"]] . Vectors4DSP4[[3]] // FullSimplify;
    SymmetricPoint4DP4cosp4 = vec4[ArcCos[Symbol["cos1"]], ArcCos[Symbol[
      "cos2"]], Symbol["phi"]] . Vectors4DSP4[[4]] // FullSimplify;
    SymmetricPoint4DP4Code = "const " <> computeType <> " cosp1q = " 
      <> CodeForm[SymmetricPoint4DP4cosp1, computeType] <> ";\n" <> "const "
       <> computeType <> " cosp2q = " <> CodeForm[SymmetricPoint4DP4cosp2, 
      computeType] <> ";\n" <> "const " <> computeType <> " cosp3q = " <> CodeForm[
      SymmetricPoint4DP4cosp3, computeType] <> ";\n" <> "const " <> computeType
       <> " cosp4q = " <> CodeForm[SymmetricPoint4DP4cosp4, computeType] <>
       ";";
    Return[SymmetricPoint4DP4Code];
  ];

DeclareSymmetricPoints4DP3[computeType_String:"double"] :=
  Module[{vec4, Vectors4DSP3, sanity, SymmetricPoint4DP3cosp1, SymmetricPoint4DP3cosp2,
     SymmetricPoint4DP3cosp3, SymmetricPoint4DP3Code},
    vec4[\[CapitalTheta]1_, \[CapitalTheta]2_, \[Phi]_] := {Cos[\[CapitalTheta]1
      ], Sin[\[CapitalTheta]1] Cos[\[CapitalTheta]2], Sin[\[CapitalTheta]1]
       Sin[\[CapitalTheta]2] Cos[\[Phi]], Sin[\[CapitalTheta]1] Sin[\[CapitalTheta]2
      ] Sin[\[Phi]]};
    Vectors4DSP3 = {vec4[0, 0, 0], vec4[(2 \[Pi]) / 3, 0, 0], vec4[(2
       \[Pi]) / 3, \[Pi], 0]};
    sanity = Map[Vectors4DSP3[[#[[1]]]] . Vectors4DSP3[[#[[2]]]]&, Subsets[
      {1, 2, 3}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/2) // N)&],
      Print["Sanity check failed!"];
      Abort[];
    ];
    SymmetricPoint4DP3cosp1 = vec4[ArcCos[Symbol["cos1"]], ArcCos[Symbol[
      "cos2"]], Symbol["phi"]] . Vectors4DSP3[[1]] // FullSimplify;
    SymmetricPoint4DP3cosp2 = vec4[ArcCos[Symbol["cos1"]], ArcCos[Symbol[
      "cos2"]], Symbol["phi"]] . Vectors4DSP3[[2]] // FullSimplify;
    SymmetricPoint4DP3cosp3 = vec4[ArcCos[Symbol["cos1"]], ArcCos[Symbol[
      "cos2"]], Symbol["phi"]] . Vectors4DSP3[[3]] // FullSimplify;
    SymmetricPoint4DP3Code = "const " <> computeType <> " cosp1q = " 
      <> CodeForm[SymmetricPoint4DP3cosp1, computeType] <> ";\n" <> "const "
       <> computeType <> " cosp2q = " <> CodeForm[SymmetricPoint4DP3cosp2, 
      computeType] <> ";\n" <> "const " <> computeType <> " cosp3q = " <> CodeForm[
      SymmetricPoint4DP3cosp3, computeType] <> ";";
    Return[SymmetricPoint4DP3Code];
  ];

(* ::Subsection::Closed:: *)

(*3D*)

(* ::Input::Initialization:: *)

DeclareSymmetricPoints3DP3[computeType_String:"double"] :=
  Module[{vec3, Vectors3DSP3, sanity, SymmetricPoint3DP3cosp1, SymmetricPoint3DP3cosp2,
     SymmetricPoint3DP3cosp3, SymmetricPoint3DP3Code},
    vec3[\[CapitalTheta]_, \[Phi]_] := {Sin[\[CapitalTheta]] Cos[\[Phi]
      ], Sin[\[CapitalTheta]] Sin[\[Phi]], Cos[\[CapitalTheta]]};
    Vectors3DSP3 = {vec3[\[Pi] / 2, 0], vec3[\[Pi] / 2, 1 (2 \[Pi]) /
       3], vec3[\[Pi] / 2, 2 (2 \[Pi]) / 3]};
    sanity = Map[Vectors3DSP3[[#[[1]]]] . Vectors3DSP3[[#[[2]]]]&, Subsets[
      {1, 2, 3}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/2) // N)&],
      Print["Sanity check failed!"];
      Abort[];
    ];
    SymmetricPoint3DP3cosp1 = vec3[ArcCos[Symbol["cos1"]], Symbol["phi"
      ]] . Vectors3DSP3[[1]] // FullSimplify;
    SymmetricPoint3DP3cosp2 = vec3[ArcCos[Symbol["cos1"]], Symbol["phi"
      ]] . Vectors3DSP3[[2]] // FullSimplify;
    SymmetricPoint3DP3cosp3 = vec3[ArcCos[Symbol["cos1"]], Symbol["phi"
      ]] . Vectors3DSP3[[3]] // FullSimplify;
    SymmetricPoint3DP3Code = "const " <> computeType <> " cosp1q = " 
      <> CodeForm[SymmetricPoint3DP3cosp1, computeType] <> ";\n" <> "const "
       <> computeType <> " cosp2q = " <> CodeForm[SymmetricPoint3DP3cosp2, 
      computeType] <> ";\n" <> "const " <> computeType <> " cosp3q = " <> CodeForm[
      SymmetricPoint3DP3cosp3, computeType] <> ";";
    Return[SymmetricPoint3DP3Code];
  ];

DeclareSymmetricPoints3DP4[computeType_String:"double"] :=
  Module[{vec3, sanity, Vectors3DSP4, SymmetricPoint3DP4cosp1, SymmetricPoint3DP4cosp2,
     SymmetricPoint3DP4cosp3, SymmetricPoint3DP4cosp4, SymmetricPoint3DP4Code
    },
    vec3[\[CapitalTheta]_, \[Phi]_] := {Sin[\[CapitalTheta]] Cos[\[Phi]
      ], Sin[\[CapitalTheta]] Sin[\[Phi]], Cos[\[CapitalTheta]]};
    Vectors3DSP4 = {vec3[0, 0], vec3[ArcCos[-(1/3)], 0], vec3[ArcCos[
      -(1/3)], 1 (2 \[Pi]) / 3], vec3[ArcCos[-(1/3)], 2 (2 \[Pi]) / 3]};
    sanity = Map[Vectors3DSP4[[#[[1]]]] . Vectors3DSP4[[#[[2]]]]&, Subsets[
      {1, 2, 3, 4}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/3) // N)&],
      Print["Sanity check failed!"];
      Abort[];
    ];
    SymmetricPoint3DP4cosp1 = vec3[ArcCos[Symbol["cos1"]], Symbol["phi"
      ]] . Vectors3DSP4[[1]] // FullSimplify;
    SymmetricPoint3DP4cosp2 = vec3[ArcCos[Symbol["cos1"]], Symbol["phi"
      ]] . Vectors3DSP4[[2]] // FullSimplify;
    SymmetricPoint3DP4cosp3 = vec3[ArcCos[Symbol["cos1"]], Symbol["phi"
      ]] . Vectors3DSP4[[3]] // FullSimplify;
    SymmetricPoint3DP4cosp4 = vec3[ArcCos[Symbol["cos1"]], Symbol["phi"
      ]] . Vectors3DSP4[[4]] // FullSimplify;
    SymmetricPoint3DP4Code = "const " <> computeType <> " cosp1q = " 
      <> CodeForm[SymmetricPoint3DP4cosp1, computeType] <> ";\n" <> "const "
       <> computeType <> " cosp2q = " <> CodeForm[SymmetricPoint3DP4cosp2, 
      computeType] <> ";\n" <> "const " <> computeType <> " cosp3q = " <> CodeForm[
      SymmetricPoint3DP4cosp3, computeType] <> ";\n" <> "const " <> computeType
       <> " cosp4q = " <> CodeForm[SymmetricPoint3DP4cosp4, computeType] <>
       ";";
    Return[SymmetricPoint3DP4Code];
  ];

(* ::Subsection::Closed:: *)

(*2D*)

(* ::Input::Initialization:: *)

DeclareSymmetricPoints2DP3[computeType_String:"double"] :=
  Module[{vec2, Vectors3DSP3, sanity, SymmetricPoint3DP3cosp1, SymmetricPoint3DP3cosp2,
     SymmetricPoint3DP3cosp3, SymmetricPoint3DP3Code},
    vec2[\[CapitalTheta]_] := {Cos[\[CapitalTheta]], Sin[\[CapitalTheta]
      ]};
    Vectors3DSP3 = {vec2[0], vec2[1 (2 \[Pi]) / 3], vec2[2 (2 \[Pi]) 
      / 3]};
    sanity = Map[Vectors3DSP3[[#[[1]]]] . Vectors3DSP3[[#[[2]]]]&, Subsets[
      {1, 2, 3}, {2}]];
    If[Not @ AllTrue[(sanity // N), # == (-(1/2) // N)&],
      Print["Sanity check failed!"];
      Abort[];
    ];
    SymmetricPoint3DP3cosp1 = vec2[ArcCos[Symbol["cos1"]]] . Vectors3DSP3
      [[1]] // FullSimplify;
    SymmetricPoint3DP3cosp2 = vec2[ArcCos[Symbol["cos1"]]] . Vectors3DSP3
      [[2]] // FullSimplify;
    SymmetricPoint3DP3cosp3 = vec2[ArcCos[Symbol["cos1"]]] . Vectors3DSP3
      [[3]] // FullSimplify;
    SymmetricPoint3DP3Code = "const " <> computeType <> " cosp1q = " 
      <> CodeForm[SymmetricPoint3DP3cosp1, computeType] <> ";\n" <> "const "
       <> computeType <> " cosp2q = " <> CodeForm[SymmetricPoint3DP3cosp2, 
      computeType] <> ";\n" <> "const " <> computeType <> " cosp3q = " <> CodeForm[
      SymmetricPoint3DP3cosp3, computeType] <> ";";
    Return[SymmetricPoint3DP3Code];
  ];

DeclareSymmetricPoints2DP4[computeType_String:"double"] :=
  Module[{vec2, sanity, Vectors3DSP4, SymmetricPoint3DP4cosp1, SymmetricPoint3DP4cosp2,
     SymmetricPoint3DP4cosp3, SymmetricPoint3DP4cosp4, SymmetricPoint3DP4Code
    },
    vec2[\[CapitalTheta]_] := {Cos[\[CapitalTheta]], Sin[\[CapitalTheta]
      ]};
    Vectors3DSP4 = {vec2[0], vec2[1 (2 \[Pi]) / 4], vec2[2 (2 \[Pi]) 
      / 4], vec2[3 (2 \[Pi]) / 4]};
    sanity = Map[Vectors3DSP4[[#[[1]]]] . Vectors3DSP4[[#[[2]]]]&, Subsets[
      {1, 2, 3, 4}, {2}]];
    SymmetricPoint3DP4cosp1 = vec2[ArcCos[Symbol["cos1"]]] . Vectors3DSP4
      [[1]] // FullSimplify;
    SymmetricPoint3DP4cosp2 = vec2[ArcCos[Symbol["cos1"]]] . Vectors3DSP4
      [[2]] // FullSimplify;
    SymmetricPoint3DP4cosp3 = vec2[ArcCos[Symbol["cos1"]]] . Vectors3DSP4
      [[3]] // FullSimplify;
    SymmetricPoint3DP4cosp4 = vec2[ArcCos[Symbol["cos1"]]] . Vectors3DSP4
      [[4]] // FullSimplify;
    SymmetricPoint3DP4Code = "const " <> computeType <> " cosp1q = " 
      <> CodeForm[SymmetricPoint3DP4cosp1, computeType] <> ";\n" <> "const "
       <> computeType <> " cosp2q = " <> CodeForm[SymmetricPoint3DP4cosp2, 
      computeType] <> ";\n" <> "const " <> computeType <> " cosp3q = " <> CodeForm[
      SymmetricPoint3DP4cosp3, computeType] <> ";\n" <> "const " <> computeType
       <> " cosp4q = " <> CodeForm[SymmetricPoint3DP4cosp4, computeType] <>
       ";";
    Return[SymmetricPoint3DP4Code];
  ];

(* ::Chapter:: *)

(*Flow code generation*)

(* ::Section:: *)

(*Kernel methods and classes for T = 0*)

(*Block[{Print},Get["FunKit`"]]*)

$PredefRegFunc = {"RB", "RF", "RBdot", "RFdot", "dq2RB", "dq2RF"};

$StandardKernelDefinitions = Map[FunKit`MakeCppFunction["Name" -> #, 
  "Body" -> "return Regulator::" <> # <> "(k2, p2);", "Prefix" -> "static KOKKOS_FORCEINLINE_FUNCTION",
   "Suffix" -> "", "Parameters" -> {"k2", "p2"}]&, $PredefRegFunc];

DiFfRG::MissingKey = "The key \"`1`\" is missing.";

CheckKey[kernel_Association, name_String, test_] :=
  Module[{valid},
    If[Not @ KeyExistsQ[kernel, name],
      Message[DiFfRG::MissingKey, name];
      Return[False]
    ];
    If[Not @ test[kernel[name]],
      Return[False]
    ];
    Return[True];
  ];

KernelSpecQ[spec_Association] :=
  Module[{validKeys, validKeyTypes},
    validKeys = CheckKey[spec, "Name", StringQ] && CheckKey[spec, "Integrator",
       StringQ] && CheckKey[spec, "d", IntegerQ] && CheckKey[spec, "AD", BooleanQ
      ] && CheckKey[spec, "Device", StringQ];
    Return[validKeys];
  ];

getRegulator[name_, {optName_, optDef_}] :=
  Module[{ret},
    ret = "";
    If[optName =!= "",
      ret = ret <> optDef <> "\n";
    ];
    ret = ret <> "using Regulator = " <> name <> "<";
    If[optName =!= "",
      ret = ret <> optName;
    ];
    ret = ret <> ">;";
    Return[FunKit`FormatCode[ret]];
  ];

ExportCode::WrongSyntax = "Incorrect arguments for ExportCode: `1`";

ExportCode[b___] :=
  (
    Message[ExportCode::WrongSyntax, {b}];
    Abort[]
  )

ExportCode[fileName_, content_] :=
  Module[{},
    If[FileExistsQ[fileName],
      If[Import[fileName, "Text"] === content,
        Print[fileName <> " unchanged"];
        Return[]
      ];
    ];
    Export[fileName, content, "Text"];
    Print["Exported to \"" <> fileName <> "\""];
  ];

updateCMake[varName_:"Flows"] :=
  Module[{folders, sources, cmake, fileName = flowDir <> "CMakeLists.txt",
     flowFolderName},
    flowFolderName = StringSplit[flowDir, "/"][[-1]];
    folders = Select[FileNames["*", flowDir, 1], DirectoryQ];
    sources = Flatten @ Join[Map[Get[# <> "/sources.m"]&, folders]];
    sources = "    " <> StringRiffle[sources, "\n    "];
    cmake = "set(" <> varName <> "_SOURCES 
" <> sources <> "
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library("
       <> varName <> " STATIC ${" <> varName <> "_SOURCES})
target_link_libraries("
       <> varName <> " DiFfRG::DiFfRG)
target_compile_options(
  " <> varName
       <> " PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wno-unused-parameter
                         -Wno-unused-variable >)
"
      ;
    ExportCode[fileName, cmake];
  ];

updateFlowClass[varName_:"Flows"] :=
  Module[{folders, flowFolderName, integrators, flowHeader, flowCpp, 
    integratorInitializations},
    flowFolderName = StringSplit[flowDir, "/"][[-1]];
    folders = Select[FileNames["*", flowDir, 1], DirectoryQ];
    integrators = Map[{StringSplit[#, "/"][[-1]], StringSplit[#, "/"]
      [[-1]] <> "_integrator", "./" <> StringSplit[#, "/"][[-1]] <> "/" <> 
      StringSplit[#, "/"][[-1]] <> ".hh"}&, folders];
    flowHeader = FunKit`MakeCppHeader["Includes" -> Join[{"DiFfRG/common/utils.hh",
       "DiFfRG/physics/integration.hh"}, integrators[[All, 3]]], "Body" -> 
      {FunKit`MakeCppClass["Name" -> varName, "MembersPublic" -> Join[{FunKit`MakeCppFunction[
      "Name" -> varName, "Parameters" -> {<|"Type" -> "DiFfRG::JSONValue", 
      "Reference" -> True, "Const" -> True, "Name" -> "json"|>}, "Body" -> 
      None, "Return" -> ""], FunKit`MakeCppFunction["Name" -> "set_k", "Parameters"
       -> {<|"Type" -> "double", "Const" -> True, "Name" -> "k", "Reference"
       -> False|>}, "Body" -> None, "Return" -> "void"], "DiFfRG::QuadratureProvider quadrature_provider;"
      }, Map[#[[2]] <> " " <> #[[1]] <> ";"&, integrators]]]}];
    integratorInitializations =
      If[Length[integrators] > 0,
        ": " <> StringRiffle[integrators[[All, 1]], "(quadrature_provider, json), "
          ] <> "(quadrature_provider, json)"
        ,
        ""
      ];
    flowCpp = FunKit`MakeCppBlock["Includes" -> {"./flows.hh"}, "Body"
       -> {FunKit`MakeCppFunction["Name" -> varName, "Class" -> varName, "Suffix"
       -> integratorInitializations, "Body" -> "", "Parameters" -> {<|"Type"
       -> "DiFfRG::JSONValue", "Reference" -> True, "Const" -> True, "Name"
       -> "json"|>}, "Return" -> ""], FunKit`MakeCppFunction["Name" -> "set_k",
       "Class" -> varName, "Parameters" -> {<|"Type" -> "double", "Const" ->
       True, "Name" -> "k", "Reference" -> False|>}, "Body" -> StringJoin[Map[
      "
if constexpr(DiFfRG::has_set_k<decltype(" <> #[[1]] <> ".integrator)>) "
       <> #[[1]] <> ".integrator.set_k(k);
if constexpr(DiFfRG::has_integrator_AD<decltype("
       <> #[[1]] <> ")>)
if constexpr(DiFfRG::has_set_k<decltype(" <> #[[1]]
       <> ".integrator_AD)>)" <> #[[1]] <> ".integrator_AD.set_k(k);" <> "
if constexpr(DiFfRG::has_set_T<decltype("
       <> #[[1]] <> ".integrator)>) " <> #[[1]] <> ".integrator.set_T(k);
if constexpr(DiFfRG::has_integrator_AD<decltype("
       <> #[[1]] <> ")>)
if constexpr(DiFfRG::has_set_T<decltype(" <> #[[1]]
       <> ".integrator_AD)>)" <> #[[1]] <> ".integrator_AD.set_T(k);"&, integrators
      ]], "Return" -> "void"]}];
    ExportCode[flowDir <> "flows.hh", flowHeader];
    ExportCode[flowDir <> "flows.cc", flowCpp];
  ];

UpdateFlows[varName_:flowName] :=
  Module[{},
    updateCMake[varName];
    updateFlowClass[varName];
  ];

(*MakeKernel[1,1,
<|"Name"->"AA","Integrator"->"DiFfRG::Integrator_p2","d"->4,"AD"->False,"Device"->"GPU","Type"->"double"|>,
{"a"},
"IntegrationVariables"->{"l1"}]
UpdateFlows["ONFiniteTFlows"]*)

(* Get["modules/MakeKernel.m"] *)

(* ::Chapter::Closed:: *)

(*Finishing package*)

Protect["DiFfRG`CodeTools`*"];

End[];

EndPackage[];
