Off[Assert::"AssertFalse"];

AppendTo[$Path, FileNameJoin[{DirectoryName[$InputFileName], "..", "modules"
     }]]

Get["MakeKernel.m"]

tests = {TestCreate[MakeKernel`Private`appendDefaultAssociation[<||>],
      <|"Type" -> "double", "Reference" -> True, "Const" -> True|>], TestCreate[
     MakeKernel`Private`appendDefaultAssociation[<|"Type" -> "float"|>], <|
     "Type" -> "float", "Reference" -> True, "Const" -> True|>], TestCreate[
     MakeKernel`Private`appendDefaultAssociation[<|"Type" -> "float", "Reference"
      -> False, "Const" -> False|>], <|"Type" -> "float", "Reference" -> False,
      "Const" -> False|>], TestCreate[MakeKernel`Private`appendDefaultAssociation[
     <|"Extra" -> "Value"|>], <|"Type" -> "double", "Reference" -> True, "Const"
      -> True, "Extra" -> "Value"|>], TestCreate[MakeKernel`Private`appendDefaultAssociation[
     <|"Type" -> "float", "Extra" -> "Value"|>], <|"Type" -> "float", "Reference"
      -> True, "Const" -> True, "Extra" -> "Value"|>]};
