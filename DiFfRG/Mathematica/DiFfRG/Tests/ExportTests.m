Needs["DiFfRG`CodeTools`Export`"];

tests = {};

AppendTo[
    tests
    ,
    VerificationTest[
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
