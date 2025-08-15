(* 
  TestRunner.m
  This script discovers and runs all Mathematica test files ending in "Tests.m" 
  in its directory.
*)

RunAndReportTests[tests_List, testFileName_String] :=
    Module[{result, successCount, failureCount},
        Print["Running tests from: ", testFileName];
        result = TestReport[tests];
        successCount = Length[result["TestsSucceededKeys"]];
        failureCount = Length[result["TestsFailedWrongResultsKeys"]];
            
        Print["  ✓ ", successCount, " passed", "    ", "x ", failureCount,
             " failed"];
        If[failureCount > 0,
            Print["\n", "  Failed Tests Details:"];
            Scan[
                (
                    Print["\n", "  Test:", " ", #["Input"]];
                    Print["    Expected: ", #["ExpectedOutput"]];
                    Print["    Actual:   ", #["ActualOutput"]];
                )&
                ,
                Values[KeyTake[result["TestResults"], result["TestsFailedWrongResultsKeys"
                    ]]]
            ]
        ];
        Return[{successCount, failureCount}];
    ];

(* Main script execution logic *)

Module[{testFiles, totalSuccesses = 0, totalFailures = 0},
    AppendTo[$Path, DirectoryName[$InputFileName]];
    AppendTo[$Path, FileNameJoin[{DirectoryName[$InputFileName], "..",
         "modules"}]];
    testFiles = FileNames["*Tests.m", DirectoryName[$InputFileName]];
        
    Print["Discovering and running tests..."];
    Print["---------------------------------"];
    Scan[
        (
            Get[#];
            If[ValueQ[tests],
                Module[{results},
                    results = RunAndReportTests[tests, FileNameTake[#
                        ]];
                    totalSuccesses += results[[1]];
                    totalFailures += results[[2]];
                    Print[""]; (* newline separator *)
                ]
                ,
                (
                    Print["  ERROR: Test file ", FileNameTake[#], " does not define a 'tests' variable."
                        ];
                    totalFailures++;
                )
            ]
        )&
        ,
        testFiles
    ];
    Print["---------------------------------"];
    Print["Overall Test Summary"];
    Print["---------------------------------"];
    Print["✓ ", totalSuccesses, " passed", "    ", "x ", totalFailures,
         " failed"];
    Print["---------------------------------"];
    Exit[totalFailures];
];
