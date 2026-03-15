BeginPackage["DiFfRG`CodeTools`Export`"];

Unprotect["DiFfRG`CodeTools`Export`*"];

ClearAll["DiFfRG`CodeTools`Export`*"];

ClearAll["DiFfRG`CodeTools`Export`Private`*"];

ExportCode::usage = "ExportCode[fileName_String,expression_String]
Writes the given expression to disk and runs clang-format on it.
";

ExportCode::WrongSyntax = "Incorrect arguments for ExportCode: `1`";
ExportCode::exportFailed = "Export to file `1` failed.";

Begin["`Private`"];

ExportCode[fileName_String, content_String] :=
    Module[{result},
        If[FileExistsQ[fileName],
            If[Import[fileName, "Text"] === content,
                Print[fileName <> " unchanged"];
                Return[]
            ];
        ];
        result = Export[fileName, content, "Text"];
        If[result === $Failed,
            Message[ExportCode::exportFailed, fileName];
            Abort[]
        ];
        Print["Exported to \"" <> fileName <> "\""];
    ];

ExportCode[b___] :=
    (
        Message[ExportCode::WrongSyntax, {b}];
        Abort[]
    )

End[];

EndPackage[];
