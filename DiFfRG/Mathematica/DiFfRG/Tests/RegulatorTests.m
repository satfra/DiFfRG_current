Needs["DiFfRG`CodeTools`Regulator`"];
tests = {};

AppendTo[tests,
    TestCreate[
        DiFfRG`CodeTools`Regulator`getRegulator["MyRegulator", {"MyOptions", "default"}],
        "default\nusing Regulator = MyRegulator<MyOptions>;",
        TestID -> "Test getRegulator with standard names"
    ]
];

AppendTo[tests,
    TestCreate[
        DiFfRG`CodeTools`Regulator`getRegulator["AnotherReg", {"OtherOptions", "default"}],
        "default\nusing Regulator = AnotherReg<OtherOptions>;",
        TestID -> "Test getRegulator with different names"
    ]
];

AppendTo[tests,
    TestCreate[
        Quiet[CheckAbort[DiFfRG`CodeTools`Regulator`getRegulator[123, {"a", "b"}]; "no-abort", "aborted"]],
        "aborted",
        TestID -> "getRegulator with non-String name should abort"
    ]
];

AppendTo[tests,
    TestCreate[
        Quiet[CheckAbort[DiFfRG`CodeTools`Regulator`getRegulator["x", {1, "b"}]; "no-abort", "aborted"]],
        "aborted",
        TestID -> "getRegulator with non-String optName should abort"
    ]
];

AppendTo[tests,
    TestCreate[
        DiFfRG`CodeTools`Regulator`getRegulator["MyReg", {"", ""}],
        "\nusing Regulator = MyReg<>;",
        TestID -> "getRegulator with empty options produces empty template args"
    ]
];