BeginPackage["DiFfRG`CodeTools`Regulator`"];

Unprotect["DiFfRG`CodeTools`Regulator`*"];

ClearAll["DiFfRG`CodeTools`Regulator`*"];

ClearAll["DiFfRG`CodeTools`Regulator`Private`*"];

getRegulator::usage = "getRegulator[name, {optName, optDef}] generates a string for a C++ type alias 'Regulator'. \
'name' is the name of the regulator class. 'optName' and 'optDef' specify the template parameter and its definition; use {\"\", \"\"} for non-templated regulators.";


Begin["`Private`"];

getRegulator[name_, {optName_, optDef_}] :=
  Module[{returnValue},
    returnValue = StringTemplate["`1`\nusing Regulator = `2`<`3`>;"][If[optName =!= "", optDef, ""], name, optName];

    returnValue
  ];

End[];

EndPackage[];
