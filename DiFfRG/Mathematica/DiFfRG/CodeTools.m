(* ::Package:: *)

(* ::Title:: *)
(*DiFfRG: Code Tools package*)


(* ::Chapter:: *)
(*Package Setup*)


(* ::Section:: *)
(*Setup and exports*)


BeginPackage["DiFfRG`CodeTools`"];
Unprotect["DiFfRG`CodeTools`*"];
ClearAll["DiFfRG`CodeTools`*"];
ClearAll["DiFfRG`CodeTools`Private`*"];


UpdateFlows::usage="";
GetStandardKernelDefinitions::usage=""


FlowKernel::usage = "FlowKernel[expr_,name_String,NT_String:\"auto\",addprefix_String:\"\"]
Makes an equation into a lambda expression - of limited usefulness, but can be used together with LoopIntegrals::integrate and similar functions.";

ExportCode::usage = "ExportCode[fileName_String,expression_String]
Writes the given expression to disk and runs clang-format on it.";


SetFlowDirectory::usage="SetFlowDirectory[dir]
Set the current flow directory, i.e. where all generated files are saved. Default is ./flows/";
ShowFlowDirectory::usage="ShowFlowDirectory[]
Show the current flow directory, i.e. where all generated files are saved. Default is ./flows/";
SetFlowName::usage="";

GetStandardKernelDefinitions::usage="";

MakeKernel::usage = "MakeKernel[kernel_Association, parameterList_List,integrandFlow_,constantFlow_:0., integrandDefinitions_String:\"\", constantDefinitions_String:\"\"]
Make a kernel from a given flow equation, parmeter list and kernel. The kernel must be a valid specification of an integration kernel.
This Function creates an integrator that evaluates (constantFlow + \[Integral]integrandFlow). One can prepend additional c++ definitions to the flow equation by using the integrandDefinitions and constantDefinitions parameters. 
These are prepended to the respective methods of the integration kernel, allowing one to e.g. define specific angles one needs for the flow code.";


SafeFiniteTFunctions::usage="";


DeclareSymmetricPoints4DP4::usage="DeclareSymmetricPoints4DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 4D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints4DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints4DP3::usage="DeclareSymmetricPoints4DP3[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 4D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints4DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints3DP4::usage="DeclareSymmetricPoints3DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 3D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints3DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints3DP3::usage="DeclareSymmetricPoints3DP3[]
Obtain C++ code declaring angles for a three-point symmetric configuration in 3D.
The angles will have the names cosp1q, cosp2q and cosp3q.
DeclareSymmetricPoints3DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints2DP4::usage="DeclareSymmetricPoints2DP4[]
Obtain C++ code declaring angles for a four-point symmetric configuration in 2D.
The angles will have the names cosp1q, cosp2q, cosp3q and cosp4q.
DeclareSymmetricPoints2DP4[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareSymmetricPoints2DP3::usage="DeclareSymmetricPoints2DP3[]
Obtain C++ code declaring angles for a three-point symmetric configuration in 2D.
The angles will have the names cosp1q, cosp2q and cosp3q.
DeclareSymmetricPoints2DP3[computeType]
Set the type of the declared C++ variables (should be double or float).";

DeclareAnglesP34Dpqr::usage="DeclareAnglesP34Dpqr[q,p,r]
Obtain C++ code declaring angles for the angles in a full three-point function in 4D.
The angles will have the names cospq and cosqr, where q,p,r are replaced by the given Symbol names and ordered alphabetically.
DeclareAnglesP34Dpqr[q,p,r,computeType]
Set the type of the declared C++ variables (should be double or float).";


Begin["`Private`"];


(* ::Chapter:: *)
(*General Definitions and structural methods*)


(* ::Section:: *)
(*Folder setup*)


(* ::Input::Initialization:: *)
flowName="flows";
SetFlowName[name_String]:=Module[{},flowName=name]

flowDir:=If[$Notebooks&&Not[Quiet[NotebookDirectory[]]===$Failed],NotebookDirectory[],Directory[]<>"/"]<>flowName<>"/";
SetFlowDirectory[name_String]:=Module[{dir},
dir=CreateDirectory[name<>flowName<>"/";];
If[dir=!=$Failed,
flowDir:=name<>flowName<>"/";,Abort[]
];
];
ShowFlowDirectory[]:=Print["\!\(\*
StyleBox[\"DiFfRG\",\nFontWeight->\"Bold\"]\)\!\(\*
StyleBox[\" \",\nFontWeight->\"Bold\"]\)\!\(\*
StyleBox[\"CodeTools\",\nFontWeight->\"Bold\"]\)\!\(\*
StyleBox[\":\",\nFontWeight->\"Bold\"]\) Flow output directory is set to \n        "<>flowDir<>"\nThis can be modified by using \!\(\*
StyleBox[\"SetFlowName\",\nFontColor->RGBColor[1, 0.5, 0]]\)[\"YourNewName\"]"]

ShowFlowDirectory[]


(* ::Section:: *)
(*Momentum Configurations*)


(* ::Subsection::Closed:: *)
(*4D*)


(* ::Input::Initialization:: *)
DeclareAnglesP34Dpqr[q_,p_,r_,cos1_:Symbol@"cos1",cos2_:Symbol@"cos2",phi_:Symbol@"phi",computeType_String:"double"]:=Module[
{vec4,Vectorp,Vectorr,Vectorq,cos,
Resultcospq,Resultcosqr,namecospq,namecosqr,
code,file},

vec4[\[CapitalTheta]1_,\[CapitalTheta]2_,\[Phi]_]:={Cos[\[CapitalTheta]1],Sin[\[CapitalTheta]1]Cos[\[CapitalTheta]2],Sin[\[CapitalTheta]1]Sin[\[CapitalTheta]2]Cos[\[Phi]],Sin[\[CapitalTheta]1]Sin[\[CapitalTheta]2]Sin[\[Phi]]};
SetAttributes[cos,Orderless];

Vectorp=vec4[0,0,0];
Vectorr=vec4[ArcCos[cos[p,r]],0,0];
Vectorq=vec4[ArcCos[Symbol@SymbolName@cos1],ArcCos[Symbol@SymbolName@cos2],Symbol@SymbolName@phi];

Resultcospq=Vectorq . Vectorp//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]]//FullSimplify;
Resultcosqr=Vectorq . Vectorr//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]]//FullSimplify;

namecospq=cos[p,q]//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]];
namecosqr=cos[q,r]//.cos[a_,b_]:>Symbol["cos"<>ToString[a]<>ToString[b]];

code="const "<>computeType<>" "<>ToString[namecospq]<>" = "<>FunKit`CppForm[Resultcospq]<>";\n"<>"const "<>computeType<>" "<>ToString[namecosqr]<>" = "<>FunKit`CppForm[Resultcosqr]<>";";
Return[code];
];


(* ::Input::Initialization:: *)
DeclareSymmetricPoints4DP4[q_,p_,{p1_,p2_,p3_,p4_},cos1_:Symbol@"cos1",cos2_:Symbol@"cos2",phi_:Symbol@"phi",computeType_String:"double"]:=Module[
{vec4,sanity,Vectors4DSP4,
SymmetricPoint4DP4cosp1,SymmetricPoint4DP4cosp2,SymmetricPoint4DP4cosp3,SymmetricPoint4DP4cosp4,SymmetricPoint4DP4Code
},

vec4[\[CapitalTheta]1_,\[CapitalTheta]2_,\[Phi]_]:={Cos[\[CapitalTheta]1],Sin[\[CapitalTheta]1]Cos[\[CapitalTheta]2],Sin[\[CapitalTheta]1]Sin[\[CapitalTheta]2]Cos[\[Phi]],Sin[\[CapitalTheta]1]Sin[\[CapitalTheta]2]Sin[\[Phi]]};

Vectors4DSP4={vec4[\[Pi]/2,0,0],vec4[\[Pi]/2,ArcCos[-(1/3)],0],vec4[\[Pi]/2,ArcCos[-(1/3)],1 (2\[Pi])/3],vec4[\[Pi]/2,ArcCos[-(1/3)],2 (2\[Pi])/3]};
sanity=Map[Vectors4DSP4[[#[[1]]]] . Vectors4DSP4[[#[[2]]]]&,Subsets[{1,2,3,4},{2}]];
If[Not@AllTrue[(sanity//N),#==(-(1/3)//N)&],Print["Sanity check failed!"];Abort[];];

SymmetricPoint4DP4cosp1=vec4[ArcCos[Symbol@SymbolName@cos1],ArcCos[Symbol@SymbolName@cos2],Symbol@SymbolName@phi] . Vectors4DSP4[[1]]//FullSimplify;
SymmetricPoint4DP4cosp2=vec4[ArcCos[Symbol@SymbolName@cos1],ArcCos[Symbol@SymbolName@cos2],Symbol@SymbolName@phi] . Vectors4DSP4[[2]]//FullSimplify;
SymmetricPoint4DP4cosp3=vec4[ArcCos[Symbol@SymbolName@cos1],ArcCos[Symbol@SymbolName@cos2],Symbol@SymbolName@phi] . Vectors4DSP4[[3]]//FullSimplify;
SymmetricPoint4DP4cosp4=vec4[ArcCos[Symbol@SymbolName@cos1],ArcCos[Symbol@SymbolName@cos2],Symbol@SymbolName@phi] . Vectors4DSP4[[4]]//FullSimplify;

SymmetricPoint4DP4Code=
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p1}])<>" = "<>FunKit`CppForm[SymmetricPoint4DP4cosp1]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p2}])<>" = "<>FunKit`CppForm[SymmetricPoint4DP4cosp2]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p3}])<>" = "<>FunKit`CppForm[SymmetricPoint4DP4cosp3]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p4}])<>" = "<>FunKit`CppForm[SymmetricPoint4DP4cosp4]<>";";
Return[SymmetricPoint4DP4Code];
];


(* ::Input::Initialization:: *)
DeclareSymmetricPoints4DP3[q_,p_,{p1_,p2_,p3_},cos1_:Symbol@"cos1",cos2_:Symbol@"cos2",phi_:Symbol@"phi",computeType_String:"double"]:=Module[
{vec4,Vectors4DSP3,sanity,
SymmetricPoint4DP3cosp1,SymmetricPoint4DP3cosp2,SymmetricPoint4DP3cosp3,SymmetricPoint4DP3Code
},

vec4[\[CapitalTheta]1_,\[CapitalTheta]2_,\[Phi]_]:={Cos[\[CapitalTheta]1],Sin[\[CapitalTheta]1]Cos[\[CapitalTheta]2],Sin[\[CapitalTheta]1]Sin[\[CapitalTheta]2]Cos[\[Phi]],Sin[\[CapitalTheta]1]Sin[\[CapitalTheta]2]Sin[\[Phi]]};

Vectors4DSP3={vec4[0,0,0],vec4[(2\[Pi])/3,0,0],vec4[(2\[Pi])/3,\[Pi],0]};
sanity=Map[Vectors4DSP3[[#[[1]]]] . Vectors4DSP3[[#[[2]]]]&,Subsets[{1,2,3},{2}]];
If[Not@AllTrue[(sanity//N),#==(-(1/2)//N)&],Print["Sanity check failed!"];Abort[];];

SymmetricPoint4DP3cosp1=vec4[ArcCos[Symbol@SymbolName@cos1],ArcCos[Symbol@SymbolName@cos2],Symbol@SymbolName@phi] . Vectors4DSP3[[1]]//FullSimplify;
SymmetricPoint4DP3cosp2=vec4[ArcCos[Symbol@SymbolName@cos1],ArcCos[Symbol@SymbolName@cos2],Symbol@SymbolName@phi] . Vectors4DSP3[[2]]//FullSimplify;
SymmetricPoint4DP3cosp3=vec4[ArcCos[Symbol@SymbolName@cos1],ArcCos[Symbol@SymbolName@cos2],Symbol@SymbolName@phi] . Vectors4DSP3[[3]]//FullSimplify;

SymmetricPoint4DP3Code=
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p1}])<>" = "<>FunKit`CppForm[SymmetricPoint4DP3cosp1]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p2}])<>" = "<>FunKit`CppForm[SymmetricPoint4DP3cosp2]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p3}])<>" = "<>FunKit`CppForm[SymmetricPoint4DP3cosp3]<>";";
Return[SymmetricPoint4DP3Code]
];


(* ::Subsection::Closed:: *)
(*3D*)


(* ::Input::Initialization:: *)
DeclareSymmetricPoints3DP3[q_,p_,{p1_,p2_,p3_},cos1_:Symbol@"cos1",phi_:Symbol@"phi",computeType_String:"double"]:=Module[
{vec3,Vectors3DSP3,sanity,
SymmetricPoint3DP3cosp1,SymmetricPoint3DP3cosp2,SymmetricPoint3DP3cosp3,SymmetricPoint3DP3Code
},

vec3[\[CapitalTheta]_,\[Phi]_]:={Sin[\[CapitalTheta]]Cos[\[Phi]],Sin[\[CapitalTheta]]Sin[\[Phi]],Cos[\[CapitalTheta]]};

Vectors3DSP3={vec3[\[Pi]/2,0],vec3[\[Pi]/2,1 (2\[Pi])/3],vec3[\[Pi]/2,2 (2\[Pi])/3]};
sanity=Map[Vectors3DSP3[[#[[1]]]] . Vectors3DSP3[[#[[2]]]]&,Subsets[{1,2,3},{2}]];
If[Not@AllTrue[(sanity//N),#==(-(1/2)//N)&],Print["Sanity check failed!"];Abort[];];

SymmetricPoint3DP3cosp1=vec3[ArcCos[Symbol@SymbolName@cos1],Symbol@SymbolName@phi] . Vectors3DSP3[[1]]//FullSimplify;
SymmetricPoint3DP3cosp2=vec3[ArcCos[Symbol@SymbolName@cos1],Symbol@SymbolName@phi] . Vectors3DSP3[[2]]//FullSimplify;
SymmetricPoint3DP3cosp3=vec3[ArcCos[Symbol@SymbolName@cos1],Symbol@SymbolName@phi] . Vectors3DSP3[[3]]//FullSimplify;

SymmetricPoint3DP3Code=
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p1}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP3cosp1]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p2}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP3cosp2]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p3}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP3cosp3]<>";";
Return[SymmetricPoint3DP3Code];
];


(* ::Input::Initialization:: *)
DeclareSymmetricPoints3DP4[q_,p_,{p1_,p2_,p3_,p4_},cos1_:Symbol@"cos1",phi_:Symbol@"phi",computeType_String:"double"]:=Module[
{vec3,sanity,Vectors3DSP4,
SymmetricPoint3DP4cosp1,SymmetricPoint3DP4cosp2,SymmetricPoint3DP4cosp3,SymmetricPoint3DP4cosp4,SymmetricPoint3DP4Code
},

vec3[\[CapitalTheta]_,\[Phi]_]:={Sin[\[CapitalTheta]]Cos[\[Phi]],Sin[\[CapitalTheta]]Sin[\[Phi]],Cos[\[CapitalTheta]]};

Vectors3DSP4={vec3[0,0],vec3[ArcCos[-(1/3)],0],vec3[ArcCos[-(1/3)],1 (2\[Pi])/3],vec3[ArcCos[-(1/3)],2 (2\[Pi])/3]};
sanity=Map[Vectors3DSP4[[#[[1]]]] . Vectors3DSP4[[#[[2]]]]&,Subsets[{1,2,3,4},{2}]];
If[Not@AllTrue[(sanity//N),#==(-(1/3)//N)&],Print["Sanity check failed!"];Abort[];];

SymmetricPoint3DP4cosp1=vec3[ArcCos[Symbol@SymbolName@cos1],Symbol@SymbolName@phi] . Vectors3DSP4[[1]]//FullSimplify;
SymmetricPoint3DP4cosp2=vec3[ArcCos[Symbol@SymbolName@cos1],Symbol@SymbolName@phi] . Vectors3DSP4[[2]]//FullSimplify;
SymmetricPoint3DP4cosp3=vec3[ArcCos[Symbol@SymbolName@cos1],Symbol@SymbolName@phi] . Vectors3DSP4[[3]]//FullSimplify;
SymmetricPoint3DP4cosp4=vec3[ArcCos[Symbol@SymbolName@cos1],Symbol@SymbolName@phi] . Vectors3DSP4[[4]]//FullSimplify;

SymmetricPoint3DP4Code=
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p1}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP4cosp1]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p2}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP4cosp2]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p3}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP4cosp3]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p4}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP4cosp4]<>";";
Return[SymmetricPoint3DP4Code];
];


(* ::Subsection::Closed:: *)
(*2D*)


(* ::Input::Initialization:: *)
DeclareSymmetricPoints2DP3[q_,p_,{p1_,p2_,p3_},cos1_:Symbol@"cos1",computeType_String:"double"]:=Module[
{vec2,Vectors3DSP3,sanity,
SymmetricPoint3DP3cosp1,SymmetricPoint3DP3cosp2,SymmetricPoint3DP3cosp3,SymmetricPoint3DP3Code
},

vec2[\[CapitalTheta]_]:={Cos[\[CapitalTheta]],Sin[\[CapitalTheta]]};

Vectors3DSP3={vec2[0],vec2[1 (2\[Pi])/3],vec2[2 (2\[Pi])/3]};
sanity=Map[Vectors3DSP3[[#[[1]]]] . Vectors3DSP3[[#[[2]]]]&,Subsets[{1,2,3},{2}]];
If[Not@AllTrue[(sanity//N),#==(-(1/2)//N)&],Print["Sanity check failed!"];Abort[];];

SymmetricPoint3DP3cosp1=vec2[ArcCos[Symbol@SymbolName@cos1]] . Vectors3DSP3[[1]]//FullSimplify;
SymmetricPoint3DP3cosp2=vec2[ArcCos[Symbol@SymbolName@cos1]] . Vectors3DSP3[[2]]//FullSimplify;
SymmetricPoint3DP3cosp3=vec2[ArcCos[Symbol@SymbolName@cos1]] . Vectors3DSP3[[3]]//FullSimplify;

SymmetricPoint3DP3Code=
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p1}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP3cosp1]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p2}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP3cosp2]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p3}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP3cosp3]<>";";
Return[SymmetricPoint3DP3Code];
];


(* ::Input::Initialization:: *)
DeclareSymmetricPoints2DP4[q_,p_,{p1_,p2_,p3_,p4_},cos1_:Symbol@"cos1",computeType_String:"double"]:=Module[
{vec2,sanity,Vectors3DSP4,
SymmetricPoint3DP4cosp1,SymmetricPoint3DP4cosp2,SymmetricPoint3DP4cosp3,SymmetricPoint3DP4cosp4,SymmetricPoint3DP4Code
},

vec2[\[CapitalTheta]_]:={Cos[\[CapitalTheta]],Sin[\[CapitalTheta]]};

Vectors3DSP4={vec2[0],vec2[1 (2\[Pi])/4],vec2[2 (2\[Pi])/4],vec2[3 (2\[Pi])/4]};
sanity=Map[Vectors3DSP4[[#[[1]]]] . Vectors3DSP4[[#[[2]]]]&,Subsets[{1,2,3,4},{2}]];

SymmetricPoint3DP4cosp1=vec2[ArcCos[Symbol@SymbolName@cos1]] . Vectors3DSP4[[1]]//FullSimplify;
SymmetricPoint3DP4cosp2=vec2[ArcCos[Symbol@SymbolName@cos1]] . Vectors3DSP4[[2]]//FullSimplify;
SymmetricPoint3DP4cosp3=vec2[ArcCos[Symbol@SymbolName@cos1]] . Vectors3DSP4[[3]]//FullSimplify;
SymmetricPoint3DP4cosp4=vec2[ArcCos[Symbol@SymbolName@cos1]] . Vectors3DSP4[[4]]//FullSimplify;

SymmetricPoint3DP4Code=
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p1}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP4cosp1]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p2}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP4cosp2]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p3}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP4cosp3]<>";\n"<>
"const "<>computeType<>" cos"<>StringJoin@(ToString/@Sort[{q,p4}])<>" = "<>FunKit`CppForm[SymmetricPoint3DP4cosp4]<>";";
Return[SymmetricPoint3DP4Code];
];


(* ::Section::Closed:: *)
(*Safe replacements for finite T*)


(* ::Input::Initialization:: *)
SafeFiniteTFunctions[expr_,T_]:=Module[{a},inputexpr//.{
Tanh[a_/(2 T)]:>Symbol["TanhFiniteT"][a,T],Tanh[a_/T]:>Symbol["TanhFiniteT"][a,2T],Tanh[a_/(2 T)]^n_:>Symbol["TanhFiniteT"][a,T]^n,Tanh[a_/T]^n_:>Symbol["TanhFiniteT"][a,2T]^n,Coth[a_/(2 T)]:>Symbol["CothFiniteT"][a,T],Coth[a_/T]:>Symbol["CothFiniteT"][a,2T],Coth[a_/(2 T)]^n_:>Symbol["CothFiniteT"][a,T]^n,Coth[a_/T]^n_:>Symbol["CothFiniteT"][a,2T]^n,Csch[a_/(2 T)]:>Symbol["CschFiniteT"][a,T],Csch[a_/T]:>Symbol["CschFiniteT"][a,2T],Csch[a_/(2 T)]^n_:>Symbol["CschFiniteT"][a,T]^n,Csch[a_/T]^n_:>Symbol["CschFiniteT"][a,2T]^n,Sech[a_/(2 T)]:>Symbol["SechFiniteT"][a,T],Sech[a_/T]:>Symbol["SechFiniteT"][a,2T],Sech[a_/(2 T)]^n_:>Symbol["SechFiniteT"][a,T]^n,Sech[a_/T]^n_:>Symbol["SechFiniteT"][a,2T]^n
}
];


(* ::Chapter:: *)
(*Flow code generation*)


(* ::Section::Closed:: *)
(*Kernel predefines*)


(* ::Input::Initialization:: *)
$PredefRegFunc={"RB","RF","RBdot","RFdot","dq2RB","dq2RF"};
$StandardKernelDefinitions=Map[
FunKit`MakeCppFunction["Name"->#,"Body"->"return Regulator::"<>#<>"(k2, p2);","Prefix"->"static KOKKOS_FORCEINLINE_FUNCTION","Suffix"->"","Parameters"->{"k2","p2"}]&,
$PredefRegFunc];

GetStandardKernelDefinitions[]:=$StandardKernelDefinitions


(* ::Input::Initialization:: *)
getRegulator[name_,{optName_,optDef_}]:=Module[{ret},
ret="";
If[optName=!="",
ret=ret<>optDef<>"\n";
];
ret=ret<>"using Regulator = "<>name<>"<";
If[optName=!="",
ret=ret<>optName;
];
ret=ret<>">;";
Return[FunKit`FormatCode[ret]];
];


(* ::Section:: *)
(*Safety checks*)


(* ::Input::Initialization:: *)
DiFfRG::MissingKey="The key \"`1`\" is missing.";

CheckKey[kernel_Association,name_String,test_]:=Module[{valid},
If[Not@KeyExistsQ[kernel,name],Message[DiFfRG::MissingKey,name];Return[False]];
If[Not@test[kernel[name]],Return[False]];
Return[True];
];

KernelSpecQ[spec_Association]:=Module[{validKeys,validKeyTypes},
validKeys=CheckKey[spec,"Name",StringQ]&&
CheckKey[spec,"Integrator",StringQ]&&
CheckKey[spec,"d",IntegerQ]&&
CheckKey[spec,"AD",BooleanQ]&&
CheckKey[spec,"Device",StringQ];
Return[validKeys];
];


(* ::Section:: *)
(*Code for CMake and flow class*)


(* ::Input::Initialization:: *)
ExportCode::WrongSyntax="Incorrect arguments for ExportCode: `1`";

ExportCode[b___]:=(Message[ExportCode::WrongSyntax,{b}];Abort[])
ExportCode[fileName_,content_]:=Module[{},
If[FileExistsQ[fileName],
If[Import[fileName,"Text"]===content,
Print[fileName<>" unchanged"];
Return[]
];
];
Export[fileName,content,"Text"];
Print["Exported to \""<>fileName<>"\""];
];


(* ::Input::Initialization:: *)
UpdateFlows[varName_:flowName]:=Module[{},
updateCMake[varName];
updateFlowClass[varName];
];


(* ::Input::Initialization:: *)
updateCMake[varName_:"Flows"]:=Module[
{folders,sources,cmake,fileName=flowDir<>"CMakeLists.txt",flowFolderName},

flowFolderName=StringSplit[flowDir,"/"][[-1]];
folders=Select[FileNames["*",flowDir,1],DirectoryQ];
sources=Flatten@Join[Map[Get[#<>"/sources.m"]&,folders]];
sources="    "<>StringRiffle[sources,"\n    "];

cmake="set("<>varName<>"_SOURCES 
"<>sources<>"
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library("<>varName<>" STATIC ${"<>varName<>"_SOURCES})
target_link_libraries("<>varName<>" DiFfRG::DiFfRG)
target_compile_options(
  "<>varName<>" PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wno-unused-parameter
                         -Wno-unused-variable >)
";
ExportCode[fileName,cmake];
];


(* ::Input::Initialization:: *)
updateFlowClass[varName_:"Flows"]:=Module[
{folders,flowFolderName,integrators,
flowHeader,flowCpp,integratorInitializations},

flowFolderName=StringSplit[flowDir,"/"][[-1]];
folders=Select[FileNames["*",flowDir,1],DirectoryQ];
integrators=Map[
{StringSplit[#,"/"][[-1]],StringSplit[#,"/"][[-1]]<>"_integrator","./"<>StringSplit[#,"/"][[-1]]<>"/"<>StringSplit[#,"/"][[-1]]<>".hh"}&
,folders];

flowHeader=FunKit`MakeCppHeader[
"Includes"->Join[{"DiFfRG/common/utils.hh","DiFfRG/physics/integration.hh"},integrators[[All,3]]],
"Body"->{
FunKit`MakeCppClass[
"Name"->varName,
"MembersPublic"->Join[{
FunKit`MakeCppFunction[
"Name"->varName,
"Parameters"->{<|"Type"->"DiFfRG::JSONValue","Reference"->True,"Const"->True,"Name"->"json"|>},
"Body"->None,
"Return"->""
],
FunKit`MakeCppFunction[
"Name"->"set_k",
"Parameters"->{<|"Type"->"double","Const"->True,"Name"->"k","Reference"->False|>},
"Body"->None,
"Return"->"void"
],
FunKit`MakeCppFunction[
"Name"->"set_T",
"Parameters"->{<|"Type"->"double","Const"->True,"Name"->"T","Reference"->False|>},
"Body"->None,
"Return"->"void"
],
FunKit`MakeCppFunction[
"Name"->"set_x_extent",
"Parameters"->{<|"Type"->"double","Const"->True,"Name"->"x_extent","Reference"->False|>},
"Body"->None,
"Return"->"void"
],
"DiFfRG::QuadratureProvider quadrature_provider;"
},
Map[#[[2]]<>" "<>#[[1]]<>";"&,integrators]
]
]}
];

integratorInitializations=If[Length[integrators]>0,": "<>StringRiffle[integrators[[All,1]],"(quadrature_provider, json), "]<>"(quadrature_provider, json)",""];

flowCpp=FunKit`MakeCppBlock[
"Includes"->{"./flows.hh"},
"Body"->{
FunKit`MakeCppFunction[
"Name"->varName,
"Class"->varName,
"Suffix"->integratorInitializations,
"Body"->"",
"Parameters"->{<|"Type"->"DiFfRG::JSONValue","Reference"->True,"Const"->True,"Name"->"json"|>},
"Return"->""
],
FunKit`MakeCppFunction[
"Name"->"set_k",
"Class"->varName,
"Parameters"->{<|"Type"->"double","Const"->True,"Name"->"k","Reference"->False|>},
"Body"->StringJoin[
Map["DiFfRG::all_set_k("<>#[[1]]<>", k);"&,integrators]
],
"Return"->"void"
],
FunKit`MakeCppFunction[
"Name"->"set_T",
"Class"->varName,
"Parameters"->{<|"Type"->"double","Const"->True,"Name"->"T","Reference"->False|>},
"Body"->StringJoin[
Map["DiFfRG::all_set_T("<>#[[1]]<>", T);"&,integrators]
],
"Return"->"void"
],
FunKit`MakeCppFunction[
"Name"->"set_x_extent",
"Class"->varName,
"Parameters"->{<|"Type"->"double","Const"->True,"Name"->"x_extent","Reference"->False|>},
"Body"->StringJoin[
Map["DiFfRG::all_set_x_extent("<>#[[1]]<>", x_extent);"&,integrators]
],
"Return"->"void"
]
}
];

ExportCode[flowDir<>"flows.hh",flowHeader];
ExportCode[flowDir<>"flows.cc",flowCpp];
];


(* ::Section:: *)
(*Code for Kernels*)


(* ::Input::Initialization:: *)
ClearAll[MakeKernel]

MakeKernel::Invalid="The given arguments are invalid. See MakeKernel::usage";
MakeKernel::InvalidSpec="The given kernel specification is invalid.";

Options[MakeKernel]={
"Coordinates"->{},
"CoordinateArguments"->{},
"IntegrationVariables"->{},
"KernelDefinitions"->$StandardKernelDefinitions,
"Regulator"->"DiFfRG::PolynomialExpRegulator",
"RegulatorOpts"->{"",""},
"KernelBody"->"",
"ConstantBody"->""
};

$ADReplacements={};
$ADReplacementsDirect={"double"->"autodiff::real"};

MakeKernel[__]:=(Message[MakeKernel::Invalid];Abort[]);
MakeKernel[kernelExpr_,spec_Association,parameters_List,OptionsPattern[]]:=MakeKernel@@(Join[{kernelExpr,0,spec,parameters},Thread[Rule@@{#,OptionValue[MakeKernel,#]}]&@Keys[Options[MakeKernel]]]);
MakeKernel[kernelExpr_,constExpr_,spec_Association,parameters_List,OptionsPattern[]]:=Module[
{expr,const,exec,
kernel,constant,kernelClass,kernelHeader,
integratorHeader,integratorCpp,integratorTemplateParams,integratorADTemplateParams,
tparams=<|"Name"->"...t","Type"->"auto&&","Reference"->False,"Const"->False|>,
kernelDefs=OptionValue["KernelDefinitions"],
coordinates=OptionValue["Coordinates"],
getArgs=OptionValue["CoordinateArguments"],
preArguments,
regulator,
params,paramsAD,explParamAD,
i,arguments,
outputPath,sources
},

If[Not@KernelSpecQ[spec],Message[MakeKernel::InvalidSpec];Abort[]];

expr=kernelExpr;
While[ListQ[expr],expr=Plus@@expr];
const=constExpr;
While[ListQ[const],const=Plus@@const];

(********************************************************************)
(* First, the kernel itself *)
(********************************************************************)

kernel=FunKit`MakeCppFunction[
expr,
"Name"->"kernel",
"Suffix"->"",
"Prefix"->"static KOKKOS_FORCEINLINE_FUNCTION",
"Parameters"->Join[OptionValue["IntegrationVariables"],getArgs,parameters],
"Body"->"using namespace DiFfRG;using namespace DiFfRG::compute;"<>OptionValue["KernelBody"]
];

constant=FunKit`MakeCppFunction[
constExpr,
"Name"->"constant",
"Suffix"->"",
"Prefix"->"static KOKKOS_FORCEINLINE_FUNCTION",
"Parameters"->Join[getArgs,parameters],
"Body"->"using namespace DiFfRG;using namespace DiFfRG::compute;"<>OptionValue["ConstantBody"]
];

kernelClass=FunKit`MakeCppClass[
"TemplateTypes"->{"_Regulator"},
"Name"->spec["Name"]<>"_kernel",
"MembersPublic"->{"using Regulator = _Regulator;",kernel,constant},
"MembersPrivate"->kernelDefs
];

kernelHeader=FunKit`MakeCppHeader[
"Includes"->{"DiFfRG/physics/utils.hh","DiFfRG/physics/interpolation.hh","DiFfRG/physics/physics.hh"},
"Body"->{"namespace DiFfRG {",kernelClass,"} using DiFfRG::"<>spec["Name"]<>"_kernel;"}];

(********************************************************************)
(* Next, the corresponding class holding the map and get functions *)
(********************************************************************)

(*We set up lists of parameters for the map/get functions, depending on their AD setting*)
params=FunKit`Private`prepParam/@parameters;
paramsAD=params;
For[i=1,i<=Length[params],i++,
If[KeyFreeQ[params[[i]],"Type"],
params[[i]]=Association[(Normal@(params[[i]]))\[Union]{"Type"->"double"}]
,
If[params[[i]]["Type"]==="auto",
params[[i]]=KeyDrop[params[[i]],{"Type"}];
params[[i]]=Association[Normal@(params[[i]])\[Union]{"Type"->"double"}]
];
If[KeyFreeQ[params[[i]],"Const"],
params[[i]]=Association[Normal@(params[[i]])\[Union]{"Const"->True}]
];
If[KeyFreeQ[params[[i]],"Reference"],
params[[i]]=Association[Normal@(params[[i]])\[Union]{"Reference"->True}]
];
];
If[KeyFreeQ[paramsAD[[i]],"Type"],
paramsAD[[i]]=Association[(Normal@(paramsAD[[i]]))\[Union]{"Type"->"double"}]
,
If[paramsAD[[i]]["Type"]==="auto",
paramsAD[[i]]=KeyDrop[paramsAD[[i]],{"Type"}];
paramsAD[[i]]=Association[Normal@(paramsAD[[i]])\[Union]{"Type"->"autodiff::real"}],
explParamAD=StringReplace[paramsAD[[i]]["Type"],$ADReplacements];
paramsAD[[i]]=KeyDrop[paramsAD[[i]],{"Type"}];
paramsAD[[i]]=Association[Normal@(paramsAD[[i]])\[Union]{"Type"->explParamAD}];
];
If[KeyFreeQ[paramsAD[[i]],"Const"],
paramsAD[[i]]=Association[Normal@(paramsAD[[i]])\[Union]{"Const"->True}]
];
If[KeyFreeQ[paramsAD[[i]],"Reference"],
paramsAD[[i]]=Association[Normal@(paramsAD[[i]])\[Union]{"Reference"->True}]
];
];
];
arguments=StringRiffle[Map[#["Name"]&,params],", "];

getArgs=FunKit`Private`prepParam/@getArgs;
For[i=1,i<=Length[getArgs],i++,
If[KeyFreeQ[getArgs[[i]],"Type"],
getArgs[[i]]=Association[(Normal@(getArgs[[i]]))\[Union]{"Type"->"double"}]
,
If[getArgs[[i]]["Type"]==="auto",
getArgs[[i]]=KeyDrop[getArgs[[i]],{"Type"}];
getArgs[[i]]=Association[Normal@(getArgs[[i]])\[Union]{"Type"->"double"}]
];
If[KeyFreeQ[getArgs[[i]],"Const"],
getArgs[[i]]=Association[Normal@(getArgs[[i]])\[Union]{"Const"->True}]
];
If[KeyFreeQ[getArgs[[i]],"Reference"],
getArgs[[i]]=Association[Normal@(getArgs[[i]])\[Union]{"Reference"->True}]
];
];
];
preArguments=StringRiffle[Map[#["Name"]&,getArgs],", "];
If[preArguments=!="",preArguments=preArguments<>", "];

(* Choose the execution space. Default is TBB, as only TBB is compatible with the FEM assemblers. *)
exec=If[KeyFreeQ[spec,"Device"]||FreeQ[{"GPU","Threads"},spec["Device"]],"DiFfRG::TBB_exec","DiFfRG::"<>spec["Device"]<>"_exec"];

integratorTemplateParams={};
If[KeyExistsQ[spec,"d"]&&spec["d"]=!=None,AppendTo[integratorTemplateParams,ToString[spec["d"]]]];
If[KeyExistsQ[spec,"Type"],AppendTo[integratorTemplateParams,ToString[spec["Type"]]],AppendTo[integratorTemplateParams,"double"]];
AppendTo[integratorTemplateParams,spec["Name"]<>"_kernel<Regulator>"];
AppendTo[integratorTemplateParams,exec];
integratorTemplateParams=StringRiffle[integratorTemplateParams,", "];

integratorADTemplateParams={};
If[KeyExistsQ[spec,"d"]&&spec["d"]=!=None,AppendTo[integratorADTemplateParams,ToString[spec["d"]]]];
If[KeyExistsQ[spec,"Type"],AppendTo[integratorADTemplateParams,StringReplace[ToString[spec["Type"]],$ADReplacementsDirect]],AppendTo[integratorADTemplateParams,"autodiff::real"]];
AppendTo[integratorADTemplateParams,spec["Name"]<>"_kernel<Regulator>"];
AppendTo[integratorADTemplateParams,If[KeyFreeQ[spec,"Device"]||FreeQ[{"GPU","Threads"},spec["Device"]],"DiFfRG::TBB_exec","DiFfRG::"<>spec["Device"]<>"_exec"]];
integratorADTemplateParams=StringRiffle[integratorADTemplateParams,", "];

(* Now, we create the header which holds the class with the integrators and the map/get methods *)
integratorHeader=FunKit`MakeCppHeader[
"Includes"->{"DiFfRG/physics/integration.hh","DiFfRG/physics/physics.hh","DiFfRG/physics/interpolation.hh"},
"Body"->{"namespace DiFfRG {
  template<typename> class "<>spec["Name"]<>"_kernel;\n",
FunKit`MakeCppClass[
"Name"->spec["Name"]<>"_integrator",
"MembersPublic"->
Join[
{
FunKit`MakeCppFunction[
"Name"->spec["Name"]<>"_integrator",
"Parameters"->{<|"Type"->"DiFfRG::QuadratureProvider","Reference"->True,"Const"->False,"Name"->"quadrature_provider"|>,<|"Type"->"DiFfRG::JSONValue","Reference"->True,"Const"->True,"Name"->"json"|>},
"Body"->None,
"Return"->""
],
getRegulator[OptionValue["Regulator"],OptionValue["RegulatorOpts"]],
spec["Integrator"]<>"<"<>integratorTemplateParams<>"> integrator;",
If[spec["AD"],spec["Integrator"]<>"<"<>integratorADTemplateParams<>"> integrator_AD;",""]
},
Map[
FunKit`MakeCppFunction["Name"->"map","Return"->exec,"Body"->None,"Parameters"->Join[{<|"Name"->"dest","Type"->"double*","Const"->False,"Reference"->False|>,<|"Name"->"coordinates","Reference"->True,"Type"->#,"Const"->True|>},params]]&,
coordinates],
If[Length[coordinates]>0,#,{}]&@{
FunKit`MakeCppFunction["Name"->"map","Return"->exec,"Body"->"return device::apply([&](const auto...t){return map(dest, coordinates, t...);}, args);","Parameters"->Join[
{<|"Name"->"dest","Type"->"double*","Reference"->False,"Const"->False|>,
<|"Name"->"coordinates","Reference"->True,"Type"->"C","Const"->True|>,
<|"Name"->"args","Type"->"device::tuple<T...>","Reference"->True,"Const"->True|>}],
"Templates"->{"C",  "...T"}]
},
If[spec["AD"],#,{}]&@Map[
FunKit`MakeCppFunction["Name"->"map","Return"->"void","Body"->None,"Parameters"->Join[{<|"Name"->"dest","Type"->"autodiff::real*","Const"->False,"Reference"->False|>,<|"Name"->"coordinates","Reference"->True,"Type"->#,"Const"->True|>},paramsAD]]&,
coordinates],
{
FunKit`MakeCppFunction["Name"->"get","Return"->"void","Body"->None,"Parameters"->Join[{<|"Name"->"dest","Type"->"double","Reference"->True,"Const"->False|>},getArgs,params]],
FunKit`MakeCppFunction["Name"->"get","Return"->"void","Body"->"device::apply([&](const auto...t){get(dest, "<>preArguments<>"t...);}, args);","Parameters"->Join[
{<|"Name"->"dest","Type"->"double","Reference"->True,"Const"->False|>},
getArgs,
{<|"Name"->"args","Type"->"device::tuple<T...>","Reference"->True,"Const"->True|>}],
"Templates"->{ "...T"}],
If[spec["AD"],#,""]&@FunKit`MakeCppFunction["Name"->"get","Return"->"void","Body"->None,"Parameters"->Join[{<|"Name"->"dest","Type"->"autodiff::real","Reference"->True,"Const"->False|>},getArgs,paramsAD]]
}]
,
"MembersPrivate"->{"DiFfRG::QuadratureProvider& quadrature_provider;"}

],"}\nusing DiFfRG::"<>spec["Name"]<>"_integrator;"}
];

(* Finally, the code fo rall methods of the class. we will save them to different files, so they can all be compiled in separate units.*)

integratorCpp["Constructor"]=FunKit`MakeCppBlock[
"Includes"->{"../kernel.hh"},
"Body"->{
"#include \"../"<>spec["Name"]<>".hh\"\n",
FunKit`MakeCppFunction[
"Name"->spec["Name"]<>"_integrator",
"Class"->spec["Name"]<>"_integrator",
"Suffix"->": integrator(quadrature_provider, json), "<>
If[spec["AD"],"integrator_AD(quadrature_provider, json), ",""]<>"quadrature_provider(quadrature_provider)",
"Body"->"",
"Parameters"->{<|"Type"->"DiFfRG::QuadratureProvider","Reference"->True,"Const"->False,"Name"->"quadrature_provider"|>,<|"Type"->"DiFfRG::JSONValue","Reference"->True,"Const"->True,"Name"->"json"|>},
"Return"->""
]
}];
integratorCpp["CT","get"]=FunKit`MakeCppBlock[
"Includes"->{"../kernel.hh"},
"Body"->{
"#include \"../"<>spec["Name"]<>".hh\"\n",
FunKit`MakeCppFunction[
"Name"->"get",
"Class"->spec["Name"]<>"_integrator",
"Body"->"integrator.get(dest, "<>preArguments<>arguments<>");",
"Parameters"->Join[{<|"Name"->"dest","Type"->"double","Reference"->True,"Const"->False|>},getArgs,params],
"Return"->"void"
]
}];
integratorCpp["AD","get"]=FunKit`MakeCppBlock[
"Includes"->{"../kernel.hh"},
"Body"->{
"#include \"../"<>spec["Name"]<>".hh\"\n",
FunKit`MakeCppFunction[
"Name"->"get",
"Class"->spec["Name"]<>"_integrator",
"Body"->"integrator_AD.get(dest, "<>preArguments<>arguments<>");",
"Parameters"->Join[{<|"Name"->"dest","Type"->"autodiff::real","Reference"->True,"Const"->False|>},getArgs,paramsAD],
"Return"->"void"
]
}];
integratorCpp["CT","map"]=Map[FunKit`MakeCppBlock[
"Includes"->{"../kernel.hh"},
"Body"->{
"#include \"../"<>spec["Name"]<>".hh\"\n",
FunKit`MakeCppFunction[
"Name"->"map",
"Return"->exec,
"Class"->spec["Name"]<>"_integrator",
"Body"->"return integrator.map(dest, coordinates, "<>arguments<>");",
"Parameters"->Join[{<|"Name"->"dest","Type"->"double*","Const"->False,"Reference"->False|>,<|"Name"->"coordinates","Reference"->True,"Type"->#,"Const"->True|>},params]
]
}]&,coordinates];
integratorCpp["AD","map"]=Map[FunKit`MakeCppBlock[
"Includes"->{"../kernel.hh"},
"Body"->{
"#include \"../"<>spec["Name"]<>".hh\"\n",
FunKit`MakeCppFunction[
"Name"->"map",
"Return"->exec,
"Class"->spec["Name"]<>"_integrator",
"Body"->"return integrator_AD.map(dest, coordinates, "<>arguments<>");",
"Parameters"->Join[{<|"Name"->"dest","Type"->"autodiff::real*","Const"->False,"Reference"->False|>,<|"Name"->"coordinates","Reference"->True,"Type"->#,"Const"->True|>},paramsAD]
]
}]&,coordinates];

outputPath=flowDir<>spec["Name"]<>"/";
ExportCode[outputPath<>spec["Name"]<>".hh",integratorHeader];
ExportCode[outputPath<>"kernel.hh",kernelHeader];

sources={outputPath<>"src/constructor.cc"};
ExportCode[sources[[-1]],integratorCpp["Constructor"]];

AppendTo[sources,outputPath<>"src/CT_get.cc"];
ExportCode[sources[[-1]],integratorCpp["CT","get"]];

Do[
AppendTo[sources,outputPath<>"src/CT_map_"<>ToString[i]<>".cc"];
ExportCode[sources[[-1]],integratorCpp["CT","map"][[i]]],
{i,1,Length[coordinates]}];

If[spec["AD"],
AppendTo[sources,outputPath<>"src/AD_get.cc"];
ExportCode[sources[[-1]],integratorCpp["AD","get"]];

Do[
AppendTo[sources,outputPath<>"src/AD_map_"<>ToString[i]<>".cc"];
ExportCode[sources[[-1]],integratorCpp["AD","map"][[i]]],
{i,1,Length[coordinates]}];
];

sources=Map[StringReplace[#,outputPath->"${CMAKE_CURRENT_SOURCE_DIR}/"<>spec["Name"]<>"/"]&,sources];
Export[outputPath<>"sources.m",sources];
Print["Please run UpdateFlows[] to export an up-to-date CMakeLists.txt"];
];


(* ::Chapter:: *)
(*Finishing package*)


Protect["DiFfRG`CodeTools`*"];


End[];


EndPackage[];
