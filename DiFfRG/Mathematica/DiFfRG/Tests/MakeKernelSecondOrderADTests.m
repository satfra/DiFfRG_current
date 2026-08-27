Needs["AUMP`"];
Needs["DiFfRG`CodeTools`MakeKernel`"];
Needs["DiFfRG`CodeTools`Directory`"];

formCandidates[] := {
    "/usr/local/bin/form",
    "/opt/homebrew/bin/form",
    "/usr/bin/form",
    FileNameJoin[{$HomeDirectory, "Library", "Wolfram", "Applications", "FormTracer", "form-4.2.0-x86_64-osx", "form"}],
    FileNameJoin[{$UserBaseDirectory, "Applications", "FormTracer", "form"}],
    FileNameJoin[{$UserBaseDirectory, "Applications", "FormTracer", "form-4.2.0-x86_64-osx", "form"}]
};

formAvailableQ[] := AnyTrue[formCandidates[], FileExistsQ];

formExecutablePath[] := SelectFirst[formCandidates[], FileExistsQ];

containsAll[text_String, patterns_List] :=
    AllTrue[patterns, StringContainsQ[text, #] &];

generatesSecondOrderComplexAD[] :=
    Module[{tmp, header, kernelHeader, adGet, constructor},
        tmp = FileNameJoin[{AUMPTestTempDirectory[], "generated"}];
        CreateDirectory[tmp];
        SetFlowDirectory[tmp <> "/"];
        CreateDirectory[
            FileNameJoin[{tmp, "flows", "pion", "src"}],
            CreateIntermediateDirectories -> True
        ];
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
            "ConstantReturnType" -> "complex<double>"
        ];
        header = Import[FileNameJoin[{tmp, "flows", "pion", "pion.hh"}], "Text"];
        kernelHeader = Import[FileNameJoin[{tmp, "flows", "pion", "kernel.hh"}], "Text"];
        adGet = Import[FileNameJoin[{tmp, "flows", "pion", "src", "AD_get.cc"}], "Text"];
        constructor = Import[FileNameJoin[{tmp, "flows", "pion", "src", "constructor.cc"}], "Text"];

        (* Expectations match ExportCode's raw, unformatted output. *)
        containsAll[
            header,
            {
                "Integrator_p2<3, cxreal, pion_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD;",
                "Integrator_p2<3, cxReal<2, double>, pion_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD2;",
                "void get(cxreal& dest, const double& k, const double& T, const cxreal& mpi2)",
                "void get(cxReal<2, double>& dest, const double& k, const double& T, const cxReal<2, double>& mpi2)"
            }
        ] &&
            StringCount[kernelHeader, "static KOKKOS_INLINE_FUNCTION"] === 8 &&
            StringFreeQ[kernelHeader, "KOKKOS_FORCEINLINE_FUNCTION"] &&
            containsAll[
                adGet,
                {
                    "void pion_integrator::get(cxreal& dest, const double& k, const double& T, const cxreal& mpi2)",
                    "integrator_AD.get(dest,  k, T, mpi2);",
                    "void pion_integrator::get(cxReal<2, double>& dest, const double& k, const double& T, const cxReal<2, double>& mpi2)",
                    "integrator_AD2.get(dest,  k, T, mpi2);"
                }
            ] &&
            StringContainsQ[constructor, "integrator_AD(quadrature_provider, config)"] &&
            StringContainsQ[constructor, "integrator_AD2(quadrature_provider, config)"]
    ];

AUMPTestCase["MakeKernel emits second-order complex AD get wrappers", {"make-kernel", "ad", "funkit", "form"},
    AUMPAssume[Length @ PacletFind["FunKit"] > 0, "FunKit is not installed"];
    AUMPAssume[formAvailableQ[], "FORM is not installed"];
    Needs["FormTracer`"];
    FormTracer`DefineFormExecutable[formExecutablePath[]];
    Block[{Print}, Get["FunKit`"]];
    AUMPCHECK[Quiet[generatesSecondOrderComplexAD[], OptionValue::nodef]];
];
