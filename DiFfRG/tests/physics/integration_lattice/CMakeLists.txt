# ##############################################################################
# Tests on CPU
# ##############################################################################

add_executable(test_integrator_1D_lattice_cpu integrator_1D_lattice_cpu.cc)
setup_test(test_integrator_1D_lattice_cpu)

add_executable(test_integrator_2D_lattice_cpu integrator_2D_lattice_cpu.cc)
setup_test(test_integrator_2D_lattice_cpu)

add_executable(test_integrator_3D_lattice_cpu integrator_3D_lattice_cpu.cc)
setup_test(test_integrator_3D_lattice_cpu)

add_executable(test_integrator_4D_lattice_cpu integrator_4D_lattice_cpu.cc)
setup_test(test_integrator_4D_lattice_cpu)

# ##############################################################################
# Tests on GPU
# ##############################################################################

if(USE_CUDA)

endif()

# ##############################################################################
# Benchmarks
# ##############################################################################

if(USE_CUDA)

endif()
