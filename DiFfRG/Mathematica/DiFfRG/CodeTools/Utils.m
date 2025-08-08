(* Exported symbols added here with SymbolName::usage *)

BeginPackage["DiFfRG`CodeTools`Utils`"]

appendDefaultAssociation::usage = "appendDefaultAssociation[config] appends default association to the given config.";

processParameters::usage = "processParameters[params, adReplacements] processes parameters for kernel generation.";

Begin["`Private`"]

ClearAll[appendDefaultAssociation, processParameters]

appendDefaultAssociation[config_] :=
    Merge[{config, <|"Type" -> "double", "Reference" -> True, "Const"
         -> True|>}, First];

processParameters[params_List, adReplacements_List] :=
    Module[{processedParams = params, paramsAD = {}},
        For[i = 1, i <= Length[processedParams], i++,
            processedParams[[i]] = appendDefaultAssociation[processedParams
                [[i]]];
            paramsAD =
                Append[
                    paramsAD
                    ,
                    Module[{typeAD},
                        typeAD = processedParams[[i]]["Type"] /. adReplacements
                            ;
                        Merge[{processedParams[[i]], <|"Type" -> typeAD
                            |>}, Last]
                    ]
                ];
        ];
        {processedParams, paramsAD}
    ]

End[]

EndPackage[]
