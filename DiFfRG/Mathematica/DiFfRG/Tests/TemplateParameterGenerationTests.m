Needs["DiFfRG`CodeTools`TemplateParameterGeneration`"]

sortList[l_] := Sort[l];

tests = {
     TestCreate[
          TemplateParameterGeneration[<|"d" -> 2, "Name" -> "MyKernel", "ctype" -> "float", "Device" -> "GPU"|>],
          {"2", "float", "MyKernel_kernel<Regulator>", "DiFfRG::GPU_exec"}
     ],
     TestCreate[
          TemplateParameterGeneration[<|"d" -> 3, "Name" -> "Integrator", "ctype" -> "double", "Device" -> "TBB"|>],
          {"3", "double", "Integrator_kernel<Regulator>", "DiFfRG::TBB_exec"}
     ],
     TestCreate[
          TemplateParameterGeneration[<|"d" -> 3, "Name" -> "Integrator", "ctype" -> "double", "Device" -> "TBB"|>, {"double" -> "autodiff::real"}],
          {"3", "autodiff::real", "Integrator_kernel<Regulator>", "DiFfRG::TBB_exec"}
     ],
     TestCreate[
          TemplateParameterGeneration[<|"d" -> 3, "Name" -> "pion", "ctype" -> "DiFfRG::complex<double>", "Device" -> "TBB"|>, {"DiFfRG::complex<double>" -> "cxReal<2, double>"}],
          {"3", "cxReal<2, double>", "pion_kernel<Regulator>", "DiFfRG::TBB_exec"},
          TestID -> "Second-order complex AD template parameter generation"
     ],
     TestCreate[
          TemplateParameterGeneration[<|"d" -> 1, "Name" -> "Test", "ctype" -> "float", "Device" -> "Threads"|>],
          {"1", "float", "Test_kernel<Regulator>", "DiFfRG::Threads_exec"}
     ],
     TestCreate[
          TemplateParameterGeneration[<|"d" -> 4, "Name" -> "DefaultType", "Device" -> "TBB"|>],
          {"4", "double", "DefaultType_kernel<Regulator>", "DiFfRG::TBB_exec"}
     ],
     TestCreate[
          TemplateParameterGeneration[<|"d" -> 2, "Name" -> "DefaultDevice", "ctype" -> "double"|>],
          {"2", "double", "DefaultDevice_kernel<Regulator>", "DiFfRG::TBB_exec"}
     ],
     TestCreate[
          Quiet[CheckAbort[TemplateParameterGeneration[<|"Name" -> "X", "Device" -> "TBB"|>]; "no-abort", "aborted"]],
          "aborted",
          TestID -> "Missing d key should abort"
     ],
     TestCreate[
          Quiet[CheckAbort[TemplateParameterGeneration[<|"d" -> 2, "Device" -> "TBB"|>]; "no-abort", "aborted"]],
          "aborted",
          TestID -> "Missing Name key should abort"
     ],
     TestCreate[
          Quiet[CheckAbort[TemplateParameterGeneration[<|"d" -> 2, "Name" -> "X", "Device" -> "BadDevice"|>]; "no-abort", "aborted"]],
          "aborted",
          TestID -> "Invalid Device should abort"
     ],
     TestCreate[
          Quiet[CheckAbort[TemplateParameterGeneration["not an association"]; "no-abort", "aborted"]],
          "aborted",
          TestID -> "Wrong arg type should abort"
     ]
};
