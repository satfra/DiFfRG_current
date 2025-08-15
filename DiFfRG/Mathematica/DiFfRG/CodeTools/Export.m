BeginPackage["DiFfRG`CodeTools`Export`"];

Unprotect["DiFfRG`CodeTools`Export`*"];

ClearAll["DiFfRG`CodeTools`Export`*"];

ClearAll["DiFfRG`CodeTools`Export`Private`*"];

ExportCode::usage = "ExportCode[fileName_String,expression_String]
Writes the given expression to disk and runs clang-format on it.
";

Begin["`Private`"];

ExportCode::WrongSyntax = "Incorrect arguments for ExportCode: `1`";

ExportCode[b___] :=
    (
        Message[ExportCode::WrongSyntax, {b}];
        Abort[]
    )

ExportCode[fileName_, content_] :=
    Module[{},
        If[FileExistsQ[fileName],
            If[Import[fileName, "Text"] === content,
                Print[fileName <> " unchanged"];
                Return[]
            ];
        ];
        Export[fileName, content, "Text"];
        Print["Exported to \"" <> fileName <> "\""];
    ];

End[];

EndPackage[];
