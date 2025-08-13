Needs["DiFfRG`CodeTools`Regulator`"];
tests = {};

AppendTo[tests,
    VerificationTest[
        DiFfRG`CodeTools`Regulator`getRegulator["MyRegulator", {"MyOptions", "default"}],
        "default\nusing Regulator = MyRegulator<MyOptions>;",
        TestID -> "Test getRegulator with standard names"
    ]
];

AppendTo[tests,
    VerificationTest[
        DiFfRG`CodeTools`Regulator`getRegulator["AnotherReg", {"OtherOptions", "default"}],
        "default\nusing Regulator = AnotherReg<OtherOptions>;",
        TestID -> "Test getRegulator with different names"
    ]
];