Needs["DiFfRG`CodeTools`Directory`"];

tests = {};

AppendTo[tests,
    VerificationTest[
        DiFfRG`CodeTools`Directory`flowDir,
        If[$Notebooks, NotebookDirectory[], Directory[]] <> "/flows",
        TestID -> "Test flowDir default value"
    ]
];

AppendTo[tests,
    (
        DiFfRG`CodeTools`Directory`SetFlowName["myNewFlows"];
        VerificationTest[
            DiFfRG`CodeTools`Directory`flowDir,
            If[$Notebooks, NotebookDirectory[], Directory[]] <> "/" <> "myNewFlows",
            TestID -> "Test SetFlowName"
        ]
    )
];
