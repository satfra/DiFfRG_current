Needs["AUMP`"];
Needs["DiFfRG`CodeTools`Utils`"];

sortKeys[assoc_] := Association[SortBy[Normal[assoc], First]];

$ADReplacements = {
    "double" -> "autodiff::real",
    "complex" -> "complex<autodiff::real>"
};

AUMPTestCase["appendDefaultAssociation supplies all defaults", {"make-kernel", "parameters"},
    AUMPCHECKEqual[
        sortKeys[appendDefaultAssociation[<||>]],
        sortKeys[<|"Type" -> "double", "Reference" -> True, "Const" -> True|>]
    ];
];

AUMPTestCase["appendDefaultAssociation preserves a custom type", {"make-kernel", "parameters"},
    AUMPCHECKEqual[
        sortKeys[appendDefaultAssociation[<|"Type" -> "float"|>]],
        sortKeys[<|"Type" -> "float", "Reference" -> True, "Const" -> True|>]
    ];
];

AUMPTestCase["appendDefaultAssociation preserves explicit flags", {"make-kernel", "parameters"},
    AUMPCHECKEqual[
        sortKeys[appendDefaultAssociation[<|"Type" -> "float", "Reference" -> False, "Const" -> False|>]],
        sortKeys[<|"Type" -> "float", "Reference" -> False, "Const" -> False|>]
    ];
];

AUMPTestCase["appendDefaultAssociation preserves extra keys", {"make-kernel", "parameters"},
    AUMPCHECKEqual[
        sortKeys[appendDefaultAssociation[<|"Extra" -> "Value"|>]],
        sortKeys[<|"Type" -> "double", "Reference" -> True, "Const" -> True, "Extra" -> "Value"|>]
    ];
];

AUMPTestCase["appendDefaultAssociation combines custom types and extra keys", {"make-kernel", "parameters"},
    AUMPCHECKEqual[
        sortKeys[appendDefaultAssociation[<|"Type" -> "float", "Extra" -> "Value"|>]],
        sortKeys[<|"Type" -> "float", "Reference" -> True, "Const" -> True, "Extra" -> "Value"|>]
    ];
];

AUMPTestCase["appendDefaultAssociation rejects a non-Association", {"make-kernel", "parameters", "abort"},
    AUMPCHECKAbort[appendDefaultAssociation["bad"]];
];

AUMPTestCase["processParameters rejects a non-List first argument", {"make-kernel", "parameters", "abort"},
    AUMPCHECKAbort[processParameters["bad", {}]];
];

AUMPTestCase["processParameters applies AD replacements", {"make-kernel", "parameters", "ad"},
    AUMPCHECKEqual[
        Map[
            sortKeys,
            processParameters[
                {
                    <|"Name" -> "p1", "Type" -> "double", "AD" -> True|>,
                    <|"Name" -> "p2", "Type" -> "complex", "AD" -> True|>
                },
                $ADReplacements
            ],
            {2}
        ],
        Map[
            sortKeys,
            {
                {
                    <|"AD" -> True, "Name" -> "p1", "Type" -> "double", "Reference" -> True, "Const" -> True|>,
                    <|"AD" -> True, "Name" -> "p2", "Type" -> "complex", "Reference" -> True, "Const" -> True|>
                },
                {
                    <|"AD" -> True, "Name" -> "p1", "Type" -> "autodiff::real", "Reference" -> True, "Const" -> True|>,
                    <|"AD" -> True, "Name" -> "p2", "Type" -> "complex<autodiff::real>", "Reference" -> True, "Const" -> True|>
                }
            },
            {2}
        ]
    ];
];

AUMPTestCase["processParameters handles empty input", {"make-kernel", "parameters"},
    AUMPCHECKEqual[processParameters[{}, {}], {{}, {}}];
];

AUMPTestCase["GetStandardKernelDefinitions returns compiler-inlineable definitions", {"make-kernel", "funkit"},
    AUMPAssume[Length @ PacletFind["FunKit"] > 0, "FunKit is not installed"];
    Needs["DiFfRG`CodeTools`MakeKernel`"];
    AUMPCHECK[
        ListQ[GetStandardKernelDefinitions[]] &&
            Length[GetStandardKernelDefinitions[]] === 6 &&
            AllTrue[GetStandardKernelDefinitions[], Not @ FreeQ[#, "static KOKKOS_INLINE_FUNCTION"] &] &&
            AllTrue[GetStandardKernelDefinitions[], FreeQ[#, "static KOKKOS_FORCEINLINE_FUNCTION"] &]
    ];
];
