Needs["DiFfRG`CodeTools`Utils`"]

sortKeys[assoc_] :=
     Association[SortBy[Normal[assoc], First]];

tests = {VerificationTest[sortKeys[appendDefaultAssociation[<||>]], sortKeys[
     <|"Type" -> "double", "Reference" -> True, "Const" -> True|>]], VerificationTest[
     sortKeys[appendDefaultAssociation[<|"Type" -> "float"|>]], sortKeys[<|
     "Type" -> "float", "Reference" -> True, "Const" -> True|>]], VerificationTest[
     sortKeys[appendDefaultAssociation[<|"Type" -> "float", "Reference" ->
      False, "Const" -> False|>]], sortKeys[<|"Type" -> "float", "Reference"
      -> False, "Const" -> False|>]], VerificationTest[sortKeys[appendDefaultAssociation[
     <|"Extra" -> "Value"|>]], sortKeys[<|"Type" -> "double", "Reference" 
     -> True, "Const" -> True, "Extra" -> "Value"|>]], VerificationTest[sortKeys[
     appendDefaultAssociation[<|"Type" -> "float", "Extra" -> "Value"|>]],
      sortKeys[<|"Type" -> "float", "Reference" -> True, "Const" -> True, 
     "Extra" -> "Value"|>]]};

$ADReplacements = {"double" -> "autodiff::real", "complex" -> "complex<autodiff::real>"
     };

tests = Join[tests, {VerificationTest[Map[sortKeys, processParameters[{<|"Name"
      -> "p1", "Type" -> "double"|>, <|"Name" -> "p2", "Type" -> "complex"
     |>}, $ADReplacements], {2}], Map[sortKeys, {{<|"Name" -> "p1", "Type"
      -> "double", "Reference" -> True, "Const" -> True|>, <|"Name" -> "p2",
      "Type" -> "complex", "Reference" -> True, "Const" -> True|>}, {<|"Name"
      -> "p1", "Type" -> "autodiff::real", "Reference" -> True, "Const" ->
      True|>, <|"Name" -> "p2", "Type" -> "complex<autodiff::real>", "Reference"
      -> True, "Const" -> True|>}}, {2}]]}];
