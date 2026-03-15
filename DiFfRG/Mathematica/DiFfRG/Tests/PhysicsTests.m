(* PhysicsTests.m — Tests for DiFfRG.m physics functions *)
(* These tests require the full DiFfRG` package (which needs FunKit). *)
(* Tests are skipped gracefully if dependencies are not available. *)

tests = {};

If[Length@PacletFind["FunKit"] === 0,
    Print["  [SKIP] FunKit not available — skipping all PhysicsTests"];
    ,
    (* Load the full DiFfRG package *)
    Block[{Print},
        $diffrGPackageDir = DirectoryName[DirectoryName[$InputFileName]];
        If[!MemberQ[$Path, $diffrGPackageDir],
            AppendTo[$Path, $diffrGPackageDir];
        ];
        Get["DiFfRG`"];
    ];

    (* -- GetDirectory -- *)
    AppendTo[tests,
        TestCreate[
            StringQ[GetDirectory[]],
            True,
            TestID -> "GetDirectory returns a String"
        ]
    ];

    (* -- QuickSimplify -- *)
    AppendTo[tests,
        TestCreate[
            QuickSimplify[x^2 - x^2],
            0,
            TestID -> "QuickSimplify: x^2 - x^2 simplifies to 0"
        ]
    ];

    AppendTo[tests,
        TestCreate[
            QuickSimplify[a + b - a],
            b,
            TestID -> "QuickSimplify: a + b - a simplifies to b"
        ]
    ];

    (* -- MatsubaraSum: bosonic sum 1/(p0^2 + w^2) -- *)
    (* The bosonic Matsubara sum of 1/(p0^2 + w^2) is Coth[w/(2T)]/(2w) *)
    AppendTo[tests,
        TestCreate[
            FullSimplify[MatsubaraSum[1/(p0^2 + w^2), p0, T] - Coth[w/(2 T)]/(2 w)] === 0,
            True,
            TestID -> "MatsubaraSum: bosonic 1/(p0^2+w^2) = Coth[w/(2T)]/(2w)"
        ]
    ];

    (* -- FermionMatsubaraSum: fermionic sum 1/(p0^2 + w^2) -- *)
    (* The fermionic Matsubara sum of 1/(p0^2 + w^2) is Tanh[w/(2T)]/(2w) *)
    AppendTo[tests,
        TestCreate[
            FullSimplify[FermionMatsubaraSum[1/(p0^2 + w^2), p0, T] - Tanh[w/(2 T)]/(2 w)] === 0,
            True,
            TestID -> "FermionMatsubaraSum: fermionic 1/(p0^2+w^2) = Tanh[w/(2T)]/(2w)"
        ]
    ];
];
