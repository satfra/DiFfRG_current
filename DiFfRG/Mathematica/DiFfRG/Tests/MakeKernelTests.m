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
            ListQ[GetStandardKernelDefinitions[]] && Length[GetStandardKernelDefinitions[]] === 6,
            True,
            TestID -> "GetStandardKernelDefinitions returns list of 6 definitions"
        ]
    ];
    ,
    Print["  [SKIP] FunKit not available — skipping GetStandardKernelDefinitions test"];
];
