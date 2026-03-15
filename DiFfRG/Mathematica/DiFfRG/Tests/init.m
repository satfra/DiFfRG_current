(* init.m — Test initialization for DiFfRG Mathematica tests *)

(* Find the DiFfRG package directory (parent of Tests/) and add it to $Path *)
$diffrGPackageDir = DirectoryName[DirectoryName[$InputFileName]];
If[!MemberQ[$Path, $diffrGPackageDir],
    AppendTo[$Path, $diffrGPackageDir];
];

(* Load the DiFfRG CodeTools package *)
Block[{Print},
    Needs["DiFfRG`CodeTools`"];
];

Print["  DiFfRG CodeTools loaded from: " <> $diffrGPackageDir];
