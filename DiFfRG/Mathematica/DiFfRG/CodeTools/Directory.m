(* ::Package:: *)

BeginPackage["DiFfRG`CodeTools`Directory`"];

Unprotect["DiFfRG`CodeTools`Directory`*"];

ClearAll["DiFfRG`CodeTools`Directory`*"];

ClearAll["DiFfRG`CodeTools`Directory`Private`*"];

SetFlowName::usage = "SetFlowName[name_String] sets the name of the flow directory.";

ShowFlowDirectory::usage = "ShowFlowDirectory[] shows the current flow directory.";

SetFlowDirectory::usage="SetFlowDirectory[dir]
Set the current flow directory, i.e. where all generated files are saved. Default is ./flows/";

flowDir::usage = "flowDir is the current flow directory.";

Begin["`Private`"];

flowName = "flows";

flowDir := FileNameJoin[If[$Notebooks, NotebookDirectory[], Directory[]], flowName]

SetFlowName[name_String] :=
    flowName = name;

SetFlowDirectory[name_String]:=Module[{dir},
    dir=CreateDirectory[name<>flowName<>"/"];
    If[dir=!=$Failed,
        flowDir:=name<>flowName<>"/";
    ,Abort[]
    ];
];
ShowFlowDirectory[] :=
    TemplateApply[StringTemplate["DiFfRG Flow output directory is set to `1`\n\tThis can be modified by using SetFlowName[\"YourNewName\"]"
        ], {flowDir}];

End[];

EndPackage[];
