(* CodeToolsTests.m — Tests for CodeTools.m functions *)

Needs["DiFfRG`CodeTools`"];

tests = {};

(* ===== SafeFiniteTFunctions tests ===== *)

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[Tanh[x/(2 T)], T],
        TanhFiniteT[x, T],
        TestID -> "SafeFiniteTFunctions: Tanh[x/(2T)] -> TanhFiniteT[x, T]"
    ]
];

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[Tanh[x/T], T],
        TanhFiniteT[x, 2 T],
        TestID -> "SafeFiniteTFunctions: Tanh[x/T] -> TanhFiniteT[x, 2T]"
    ]
];

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[Coth[x/(2 T)], T],
        CothFiniteT[x, T],
        TestID -> "SafeFiniteTFunctions: Coth[x/(2T)] -> CothFiniteT[x, T]"
    ]
];

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[Coth[x/T], T],
        CothFiniteT[x, 2 T],
        TestID -> "SafeFiniteTFunctions: Coth[x/T] -> CothFiniteT[x, 2T]"
    ]
];

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[Sech[x/(2 T)], T],
        SechFiniteT[x, T],
        TestID -> "SafeFiniteTFunctions: Sech[x/(2T)] -> SechFiniteT[x, T]"
    ]
];

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[Csch[x/T], T],
        CschFiniteT[x, 2 T],
        TestID -> "SafeFiniteTFunctions: Csch[x/T] -> CschFiniteT[x, 2T]"
    ]
];

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[Tanh[x/(2 T)]^2, T],
        TanhFiniteT[x, T]^2,
        TestID -> "SafeFiniteTFunctions: Tanh[x/(2T)]^2 -> TanhFiniteT[x, T]^2"
    ]
];

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[x^2 + y, T],
        x^2 + y,
        TestID -> "SafeFiniteTFunctions: expression with no finite-T functions unchanged"
    ]
];

AppendTo[tests,
    TestCreate[
        SafeFiniteTFunctions[Tanh[a/(2 T)] + Coth[b/T], T],
        TanhFiniteT[a, T] + CothFiniteT[b, 2 T],
        TestID -> "SafeFiniteTFunctions: compound expression with multiple replacements"
    ]
];

(* ===== DeclareSymmetricPoints / DeclareAngles tests ===== *)
(* These require FunKit`CppForm — guard with a FunKit check *)

If[Length@PacletFind["FunKit"] > 0,

    (* Load FunKit so that FunKit`CppForm is available *)
    Block[{Print}, Get["FunKit`"]];

    (* -- DeclareSymmetricPoints4DP4 -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareSymmetricPoints4DP4[q, p, {p1, p2, p3, p4}]},
                StringQ[result] && StringContainsQ[result, "const double"]
            ],
            True,
            TestID -> "DeclareSymmetricPoints4DP4: returns String with const double"
        ]
    ];

    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareSymmetricPoints4DP4[q, p, {p1, p2, p3, p4}]},
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp2q"] &&
                StringContainsQ[result, "cosp3q"] && StringContainsQ[result, "cosp4q"]
            ],
            True,
            TestID -> "DeclareSymmetricPoints4DP4: contains expected variable names"
        ]
    ];

    (* -- DeclareSymmetricPoints4DP3 -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareSymmetricPoints4DP3[q, p, {p1, p2, p3}]},
                StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp2q"] &&
                StringContainsQ[result, "cosp3q"]
            ],
            True,
            TestID -> "DeclareSymmetricPoints4DP3: returns valid C++ declarations"
        ]
    ];

    (* -- DeclareSymmetricPoints3DP4 -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareSymmetricPoints3DP4[q, p, {p1, p2, p3, p4}]},
                StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp4q"]
            ],
            True,
            TestID -> "DeclareSymmetricPoints3DP4: returns valid C++ declarations"
        ]
    ];

    (* -- DeclareSymmetricPoints3DP3 -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareSymmetricPoints3DP3[q, p, {p1, p2, p3}]},
                StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp3q"]
            ],
            True,
            TestID -> "DeclareSymmetricPoints3DP3: returns valid C++ declarations"
        ]
    ];

    (* -- DeclareSymmetricPoints2DP4 -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareSymmetricPoints2DP4[q, p, {p1, p2, p3, p4}]},
                StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp4q"]
            ],
            True,
            TestID -> "DeclareSymmetricPoints2DP4: returns valid C++ declarations"
        ]
    ];

    (* -- DeclareSymmetricPoints2DP3 -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareSymmetricPoints2DP3[q, p, {p1, p2, p3}]},
                StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cosp1q"] && StringContainsQ[result, "cosp3q"]
            ],
            True,
            TestID -> "DeclareSymmetricPoints2DP3: returns valid C++ declarations"
        ]
    ];

    (* -- DeclareAnglesP34Dpqr -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareAnglesP34Dpqr[q, p, r]},
                StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cospq"] && StringContainsQ[result, "cosqr"]
            ],
            True,
            TestID -> "DeclareAnglesP34Dpqr: returns valid C++ angle declarations"
        ]
    ];

    (* -- DeclareAnglesP33Dpqr -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareAnglesP33Dpqr[q, p, r]},
                StringQ[result] && StringContainsQ[result, "const double"] &&
                StringContainsQ[result, "cospq"] && StringContainsQ[result, "cosqr"]
            ],
            True,
            TestID -> "DeclareAnglesP33Dpqr: returns valid C++ angle declarations"
        ]
    ];

    (* -- Custom computeType -- *)
    AppendTo[tests,
        TestCreate[
            Module[{result = DeclareSymmetricPoints2DP3[q, p, {p1, p2, p3}, Symbol@"cos1", "float"]},
                StringQ[result] && StringContainsQ[result, "const float"]
            ],
            True,
            TestID -> "DeclareSymmetricPoints2DP3: custom computeType float"
        ]
    ];

    ,
    (* FunKit not available — skip DeclareSymmetricPoints/DeclareAngles tests *)
    Print["  [SKIP] FunKit not available — skipping DeclareSymmetricPoints/DeclareAngles tests"];
];
