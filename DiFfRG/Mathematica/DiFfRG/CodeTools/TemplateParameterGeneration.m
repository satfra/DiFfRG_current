BeginPackage["DiFfRG`CodeTools`TemplateParameterGeneration`"]

TemplateParameterGeneration::usage = "";

Begin["`Private`"]

allowedDevices = {"TBB", "GPU", "Threads"};

appendKeyDimension::missingKey = "The key `1` is missing in the template parameters.";

appendKeyDimension[templateParameter_List, params_Association] :=
    Module[{},
    (* maybe rename "d" to "spatialDimension" *)
        If[KeyExistsQ[params,"d"],
            Append[templateParameter, ToString[params["d"]]],
            Message[appendKeyDimension::missingKey, "d"]
        ]
    ]

appendKeyName::missingKey = "The key `1` is missing in the template parameters.";

appendKeyName[templateParameter_List, params_Association] :=
    Module[{},
        If[KeyExistsQ[params, "Name"],
            Append[templateParameter, ToString[params["Name"]] <> "_kernel<Regulator>"],
            Message[appendKeyName::missingKey, "Name"]
        ]
    ]

appendKeyType::missingKey = "The key `1` is missing in the template parameters.";

appendKeyType[templateParameter_List, params_Association, {}] :=
    Module[{},
        If[KeyExistsQ[params, "ctype"],
            Append[templateParameter, ToString[params["ctype"]]],
            Append[templateParameter, "double"]
        ]
    ]

appendKeyType[templateParameter_List, params_Association, ADReplacements_] :=
    Module[{},
        If[KeyExistsQ[params, "ctype"],
            Append[templateParameter, ToString[params["ctype"]] /. ADReplacements],
            Append[templateParameter, "autodiff::real"]
        ]
    ]

appendKeyDevice::wrongDevice = "The Device `1` is not part of known devices `2`.";

appendKeyDevice[templateParameter_List, params_Association] :=
    Module[{},
        If[KeyExistsQ[params, "Device"],
            If[MemberQ[allowedDevices, params["Device"]],
                Append[templateParameter, "DiFfRG::" <> ToString[params["Device"]] <> "_exec"],
                Message[appendKeyDevice::wrongDevice, params["Device"], allowedDevices]
            ],
            Append[templateParameter, "DiFfRG::TBB_exec"]
        ]
    ]

TemplateParameterGeneration[params_Association] := TemplateParameterGeneration[params, {}];
TemplateParameterGeneration[params_Association, ADReplacements_] :=
    Module[
        {integratorTemplateParams = {}}
        ,
        integratorTemplateParams = appendKeyDimension[integratorTemplateParams, params];
        integratorTemplateParams = appendKeyType[integratorTemplateParams, params, ADReplacements];
        integratorTemplateParams = appendKeyName[integratorTemplateParams, params];
        integratorTemplateParams = appendKeyDevice[integratorTemplateParams, params];

        integratorTemplateParams
    ];

End[]

EndPackage[]
