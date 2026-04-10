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
