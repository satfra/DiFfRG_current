(* ::Package:: *)

BeginPackage["DiFfRG`CodeTools`Directory`"];

Unprotect["DiFfRG`CodeTools`Directory`*"];

ClearAll["DiFfRG`CodeTools`Directory`*"];

ClearAll["DiFfRG`CodeTools`Directory`Private`*"];

SetFlowName::usage = "SetFlowName[name_String] sets the name of the flow directory.";

ShowFlowDirectory::usage = "ShowFlowDirectory[] shows the current flow directory.";

flowDir::usage = "flowDir is the current flow directory.";

Begin["`Private`"];

flowName = "flows";

flowDir := FileNameJoin[If[$Notebooks, NotebookDirectory[], Directory[]], flowName]

SetFlowName[name_String] :=
    flowName = name;

ShowFlowDirectory[] :=
    TemplateApply[StringTemplate["DiFfRG Flow output directory is set to `1`\n\tThis can be modified by using SetFlowName[\"YourNewName\"]"
        ], {flowDir}];

End[];

EndPackage[];
