Needs["DiFfRG`CodeTools`Export`"];

tests = {};

AppendTo[
    tests
    ,
    TestCreate[
        Module[{fileName, content, exportedContent},
            fileName = "test.txt";
            content = "test content";
            DiFfRG`CodeTools`Export`ExportCode[fileName, content];
            exportedContent = Import[fileName, "Text"];
            DeleteFile[fileName];
            exportedContent === content
        ]
        ,
        True
        ,
        TestID -> "ExportCode should export a file"
    ]
];

AppendTo[tests,
    TestCreate[
        Quiet[CheckAbort[DiFfRG`CodeTools`Export`ExportCode[123, "x"]; "no-abort", "aborted"]],
        "aborted",
        TestID -> "ExportCode with non-String fileName should abort"
    ]
];

AppendTo[tests,
    TestCreate[
        Quiet[CheckAbort[DiFfRG`CodeTools`Export`ExportCode["x", 123]; "no-abort", "aborted"]],
        "aborted",
        TestID -> "ExportCode with non-String content should abort"
    ]
];

AppendTo[tests,
    TestCreate[
        Quiet[CheckAbort[DiFfRG`CodeTools`Export`ExportCode[]; "no-abort", "aborted"]],
        "aborted",
        TestID -> "ExportCode with no arguments should abort"
    ]
];
