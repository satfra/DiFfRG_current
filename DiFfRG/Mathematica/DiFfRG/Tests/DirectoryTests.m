Needs["DiFfRG`CodeTools`Directory`"];

tests = {};

AppendTo[tests,
    TestCreate[
        DiFfRG`CodeTools`Directory`flowDir,
        If[$Notebooks, NotebookDirectory[], Directory[]] <> "/flows",
        TestID -> "Test flowDir default value"
    ]
];

AppendTo[tests,
    TestCreate[
        (
            DiFfRG`CodeTools`Directory`SetFlowName["myNewFlows"];
            DiFfRG`CodeTools`Directory`flowDir
        ),
        If[$Notebooks, NotebookDirectory[], Directory[]] <> "/" <> "myNewFlows",
        TestID -> "Test SetFlowName"
    ]
];

AppendTo[tests,
    TestCreate[
        Quiet[CheckAbort[DiFfRG`CodeTools`Directory`SetFlowName[123]; "no-abort", "aborted"]],
        "aborted",
        TestID -> "SetFlowName with non-String argument should abort"
    ]
];

AppendTo[tests,
    TestCreate[
        Quiet[CheckAbort[DiFfRG`CodeTools`Directory`SetFlowDirectory[123]; "no-abort", "aborted"]],
        "aborted",
        TestID -> "SetFlowDirectory with non-String argument should abort"
    ]
];

(* flowName is global package state and every test file shares one kernel session, so the
   SetFlowName test above would otherwise leak "myNewFlows" into every file that sorts after
   this one -- MakeKernelSecondOrderADTests then writes its kernel to <tmp>/myNewFlows/ while
   looking for it under <tmp>/flows/. Restore the default. *)
AppendTo[tests,
    TestCreate[
        (
            DiFfRG`CodeTools`Directory`SetFlowName["flows"];
            DiFfRG`CodeTools`Directory`flowDir
        ),
        If[$Notebooks, NotebookDirectory[], Directory[]] <> "/flows",
        TestID -> "SetFlowName restores the default flow name for later test files"
    ]
];
