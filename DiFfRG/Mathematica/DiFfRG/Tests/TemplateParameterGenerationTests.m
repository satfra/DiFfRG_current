Needs["DiFfRG`CodeTools`TemplateParameterGeneration`"]

sortList[l_] := Sort[l];

tests = {
     VerificationTest[
          TemplateParameterGeneration[<|"d" -> 2, "Name" -> "MyKernel", "Type" -> "float", "Device" -> "GPU"|>],
          {"2", "MyKernel_kernel<Regulator>", "float", "DiFfRG::GPU_exec"}
     ],
     VerificationTest[
          TemplateParameterGeneration[<|"d" -> 3, "Name" -> "Integrator", "Type" -> "double", "Device" -> "TBB"|>],
          {"3", "Integrator_kernel<Regulator>", "double", "DiFfRG::TBB_exec"}
     ],
     VerificationTest[
          TemplateParameterGeneration[<|"d" -> 3, "Name" -> "Integrator", "Type" -> "double", "Device" -> "TBB"|>, {"double" -> "autodiff::real"}],
          {"3", "Integrator_kernel<Regulator>", "autodiff::real", "DiFfRG::TBB_exec"}
     ],
     VerificationTest[
          TemplateParameterGeneration[<|"d" -> 1, "Name" -> "Test", "Type" -> "float", "Device" -> "Threads"|>],
          {"1", "Test_kernel<Regulator>", "float", "DiFfRG::Threads_exec"}
     ],
     VerificationTest[
          TemplateParameterGeneration[<|"d" -> 4, "Name" -> "DefaultType", "Device" -> "TBB"|>],
          {"4", "DefaultType_kernel<Regulator>", "double", "DiFfRG::TBB_exec"}
     ],
     VerificationTest[
          TemplateParameterGeneration[<|"d" -> 2, "Name" -> "DefaultDevice", "Type" -> "double"|>],
          {"2", "DefaultDevice_kernel<Regulator>", "double", "DiFfRG::TBB_exec"}
     ]
};