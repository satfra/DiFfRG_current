Needs["DiFfRG`CodeTools`Utils`"]

sortKeys[assoc_] :=
     Association[SortBy[Normal[assoc], First]];

tests = {TestCreate[sortKeys[appendDefaultAssociation[<||>]], sortKeys[
     <|"Type" -> "double", "Reference" -> True, "Const" -> True|>]], TestCreate[
     sortKeys[appendDefaultAssociation[<|"Type" -> "float"|>]], sortKeys[<|
     "Type" -> "float", "Reference" -> True, "Const" -> True|>]], TestCreate[
     sortKeys[appendDefaultAssociation[<|"Type" -> "float", "Reference" ->
      False, "Const" -> False|>]], sortKeys[<|"Type" -> "float", "Reference"
      -> False, "Const" -> False|>]], TestCreate[sortKeys[appendDefaultAssociation[
     <|"Extra" -> "Value"|>]], sortKeys[<|"Type" -> "double", "Reference" 
     -> True, "Const" -> True, "Extra" -> "Value"|>]], TestCreate[sortKeys[
     appendDefaultAssociation[<|"Type" -> "float", "Extra" -> "Value"|>]],
      sortKeys[<|"Type" -> "float", "Reference" -> True, "Const" -> True, 
     "Extra" -> "Value"|>]],
     TestCreate[
          Quiet[CheckAbort[appendDefaultAssociation["bad"]; "no-abort", "aborted"]],
          "aborted",
          TestID -> "appendDefaultAssociation with non-Association should abort"
     ],
     TestCreate[
          Quiet[CheckAbort[processParameters["bad", {}]; "no-abort", "aborted"]],
          "aborted",
          TestID -> "processParameters with non-List first arg should abort"
     ]
};

$ADReplacements = {"double" -> "autodiff::real", "complex" -> "complex<autodiff::real>"
     };

tests = Join[tests, {TestCreate[Map[sortKeys, processParameters[{<|"Name"
      -> "p1", "Type" -> "double", "AD" -> True|>, <|"Name" -> "p2", "Type" -> "complex",
      "AD" -> True|>}, $ADReplacements], {2}], Map[sortKeys, {{<|"AD" -> True, "Name" -> "p1", "Type"
      -> "double", "Reference" -> True, "Const" -> True|>, <|"AD" -> True, "Name" -> "p2",
      "Type" -> "complex", "Reference" -> True, "Const" -> True|>}, {<|"AD" -> True, "Name"
      -> "p1", "Type" -> "autodiff::real", "Reference" -> True, "Const" ->
      True|>, <|"AD" -> True, "Name" -> "p2", "Type" -> "complex<autodiff::real>", "Reference"
      -> True, "Const" -> True|>}}, {2}]]}];

(* processParameters with empty input *)
AppendTo[tests,
    TestCreate[
        processParameters[{}, {}],
        {{}, {}},
        TestID -> "processParameters with empty input returns empty lists"
    ]
];

(* GetStandardKernelDefinitions — requires FunKit *)
If[Length@PacletFind["FunKit"] > 0,
    Needs["DiFfRG`CodeTools`MakeKernel`"];
    AppendTo[tests,
        TestCreate[
            ListQ[GetStandardKernelDefinitions[]] &&
                Length[GetStandardKernelDefinitions[]] === 6 &&
                AllTrue[GetStandardKernelDefinitions[], Not @ FreeQ[#, "static KOKKOS_INLINE_FUNCTION"]&] &&
                AllTrue[GetStandardKernelDefinitions[], FreeQ[#, "static KOKKOS_FORCEINLINE_FUNCTION"]&],
            True,
            TestID -> "GetStandardKernelDefinitions returns 6 compiler-inlineable definitions"
        ]
    ];
    ,
    Print["  [SKIP] FunKit not available — skipping GetStandardKernelDefinitions test"];
];

(* KernelTraits — the trait emitter. Asserted rather than trusted: every trait detector on the
   C++ side is `requires { requires K::trait; }`, which reads FALSE for a trait it cannot see, so
   a name that never fired is silently a disabled trait and a wrong right-hand side. *)
If[Length@PacletFind["FunKit"] > 0,
    Needs["DiFfRG`CodeTools`MakeKernel`"];
    With[{emit = DiFfRG`CodeTools`MakeKernel`Private`kernelTraitMembers},
        AppendTo[tests,
            TestCreate[
                emit[{"matsubara_finite_extent"}],
                {"static constexpr bool matsubara_finite_extent = true;"},
                TestID -> "KernelTraits emits a bare name as a true trait"
            ]
        ];
        AppendTo[tests,
            TestCreate[
                emit[{"matsubara_split" -> True, "matsubara_even" -> False}],
                {"static constexpr bool matsubara_split = true;", "static constexpr bool matsubara_even = false;"},
                TestID -> "KernelTraits honours an explicit Boolean"
            ]
        ];
        AppendTo[tests,
            TestCreate[emit[{}], {}, TestID -> "KernelTraits emits nothing by default"]
        ];
        AppendTo[tests,
            TestCreate[
                emit[{"matsubara_split", "matsubara_split"}],
                {"static constexpr bool matsubara_split = true;"},
                TestID -> "KernelTraits deduplicates"
            ]
        ];
        AppendTo[tests,
            TestCreate[
                Quiet[CheckAbort[emit[{"not an identifier"}]; "no-abort", "aborted"]],
                "aborted",
                TestID -> "KernelTraits aborts on a name that is not a C++ identifier"
            ]
        ];
        AppendTo[tests,
            TestCreate[
                Quiet[CheckAbort[emit[{"matsubara_split" -> "yes"}]; "no-abort", "aborted"]],
                "aborted",
                TestID -> "KernelTraits aborts on a non-Boolean value"
            ]
        ];
    ];
    ,
    Print["  [SKIP] FunKit not available — skipping KernelTraits tests"];
];
