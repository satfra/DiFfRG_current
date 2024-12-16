# ##############################################################################
# Tests on CPU
# ##############################################################################

add_executable(test_integrator_finiteTx0_cpu integrator_finiteTx0_cpu.cc)
setup_test(test_integrator_finiteTx0_cpu)

add_executable(test_integrator_finiteTq0_cpu integrator_finiteTq0_cpu.cc)
setup_test(test_integrator_finiteTq0_cpu)

add_executable(test_integrator_angle_finiteTx0_cpu
               integrator_angle_finiteTx0_cpu.cc)
setup_test(test_integrator_angle_finiteTx0_cpu)

add_executable(test_integrator_angle_finiteTq0_cpu
               integrator_angle_finiteTq0_cpu.cc)
setup_test(test_integrator_angle_finiteTq0_cpu)

add_executable(test_integrator_4D_finiteTx0_cpu integrator_4D_finiteTx0_cpu.cc)
setup_test(test_integrator_4D_finiteTx0_cpu)

add_executable(test_integrator_4D_finiteTq0_cpu integrator_4D_finiteTq0_cpu.cc)
setup_test(test_integrator_4D_finiteTq0_cpu)

# ##############################################################################
# Tests on GPU
# ##############################################################################

if(USE_CUDA)
  add_executable(test_integrator_finiteTx0_gpu integrator_finiteTx0_gpu.cu)
  setup_test(test_integrator_finiteTx0_gpu)

  add_executable(test_integrator_finiteTq0_gpu integrator_finiteTq0_gpu.cu)
  setup_test(test_integrator_finiteTq0_gpu)

  add_executable(test_integrator_angle_finiteTx0_gpu
                 integrator_angle_finiteTx0_gpu.cu)
  setup_test(test_integrator_angle_finiteTx0_gpu)

  add_executable(test_integrator_angle_finiteTq0_gpu
                 integrator_angle_finiteTq0_gpu.cu)
  setup_test(test_integrator_angle_finiteTq0_gpu)

  add_executable(test_integrator_4D_finiteTx0_gpu
                 integrator_4D_finiteTx0_gpu.cu)
  setup_test(test_integrator_4D_finiteTx0_gpu)

  add_executable(test_integrator_4D_finiteTq0_gpu
                 integrator_4D_finiteTq0_gpu.cu)
  setup_test(test_integrator_4D_finiteTq0_gpu)
endif()

# ##############################################################################
# Benchmarks
# ##############################################################################

if(USE_CUDA)
  add_executable(benchmark_integrator_4D_finiteT
                 benchmark_integrator_4D_finiteT.cu)
  setup_benchmark(benchmark_integrator_4D_finiteT)
endif()
