Needs["DiFfRG`CodeTools`MakeKernel`"]
Needs["DiFfRG`CodeTools`Directory`"]

formAvailableQ[] :=
    AnyTrue[
        {
            "/usr/local/bin/form",
            "/opt/homebrew/bin/form",
            "/usr/bin/form",
            FileNameJoin[{$HomeDirectory, "Library", "Wolfram", "Applications", "FormTracer", "form-4.2.0-x86_64-osx", "form"}],
            FileNameJoin[{$UserBaseDirectory, "Applications", "FormTracer", "form"}],
            FileNameJoin[{$UserBaseDirectory, "Applications", "FormTracer", "form-4.2.0-x86_64-osx", "form"}]
        },
        FileExistsQ
    ];

formExecutablePath[] :=
    SelectFirst[
        {
            "/usr/local/bin/form",
            "/opt/homebrew/bin/form",
            "/usr/bin/form",
            FileNameJoin[{$HomeDirectory, "Library", "Wolfram", "Applications", "FormTracer", "form-4.2.0-x86_64-osx", "form"}],
            FileNameJoin[{$UserBaseDirectory, "Applications", "FormTracer", "form"}],
            FileNameJoin[{$UserBaseDirectory, "Applications", "FormTracer", "form-4.2.0-x86_64-osx", "form"}]
        },
        FileExistsQ
    ];

If[Length@PacletFind["FunKit"] > 0 && formAvailableQ[],
    Needs["FormTracer`"];
    FormTracer`DefineFormExecutable[formExecutablePath[]];
    Block[{Print}, Get["FunKit`"]];
];

containsAll[text_String, patterns_List] :=
    AllTrue[patterns, StringContainsQ[text, #]&];

generatesSecondOrderComplexAD[] :=
    Module[{tmp, header, adGet, constructor, result},
        tmp = CreateDirectory[];
        SetFlowDirectory[tmp <> "/"];
        CreateDirectory[FileNameJoin[{tmp, "flows", "pion", "src"}], CreateIntermediateDirectories -> True];

        MakeKernel[
            mpi2,
            "IntegrationVariables" -> {"l1"},
            "Name" -> "pion",
            "Integrator" -> "Integrator_p2",
            "d" -> 3,
            "ctype" -> "DiFfRG::complex<double>",
            "Parameters" -> {
                <|"Name" -> "k", "Type" -> "double", "Const" -> True, "AD" -> False|>,
                <|"Name" -> "T", "Type" -> "double", "Const" -> True, "AD" -> False|>,
                <|"Name" -> "mpi2", "Type" -> "DiFfRG::complex<double>", "Const" -> True, "AD" -> True|>
            },
            "AD" -> True,
            "ADOrders" -> {1, 2},
            "ConstantReturnType" -> "complex<double>"
        ];

        header = Import[FileNameJoin[{tmp, "flows", "pion", "pion.hh"}], "Text"];
        adGet = Import[FileNameJoin[{tmp, "flows", "pion", "src", "AD_get.cc"}], "Text"];
        constructor = Import[FileNameJoin[{tmp, "flows", "pion", "src", "constructor.cc"}], "Text"];

        result =
            containsAll[
                header,
                {
                    "Integrator_p2<3, cxReal<2, double>, pion_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD2;",
                    "void get(cxReal<2, double> &dest, const double &k, const double &T, const cxReal<2, double> &mpi2);"
                }
            ] &&
            containsAll[
                adGet,
                {
                    "void pion_integrator::get(cxReal<2, double> &dest, const double &k, const double &T, const cxReal<2, double> &mpi2)",
                    "integrator_AD2.get(dest, k, T, mpi2);"
                }
            ] &&
            StringContainsQ[constructor, "integrator_AD2(quadrature_provider, json)"];

        DeleteDirectory[tmp, DeleteContents -> True];
        result
    ];

tests = {};

If[Length@PacletFind["FunKit"] > 0 && formAvailableQ[],
    AppendTo[
        tests,
        TestCreate[
            generatesSecondOrderComplexAD[],
            True,
            TestID -> "MakeKernel emits second-order complex AD get wrapper"
        ]
    ];
    ,
    Print["  [SKIP] FunKit/FORM not available -- skipping second-order AD MakeKernel test"];
];
