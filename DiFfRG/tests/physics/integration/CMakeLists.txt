# ##############################################################################
# # Add subdirectories
# ##############################################################################

add_subdirectory(lattice)

# ##############################################################################
# Tests for basic integrators
# ##############################################################################

add_executable(test_integrator_1D integrator_1D.cc)
setup_test(test_integrator_1D)

add_executable(test_integrator_2D integrator_2D.cc)
setup_test(test_integrator_2D)

add_executable(test_integrator_3D integrator_3D.cc)
setup_test(test_integrator_3D)

add_executable(test_integrator_4D integrator_4D.cc)
setup_test(test_integrator_4D)

# ##############################################################################
# Tests for derived integrators
# ##############################################################################

add_executable(test_integrator_p2 integrator_p2.cc)
setup_test(test_integrator_p2)

add_executable(test_integrator_p2_1ang integrator_p2_1ang.cc)
setup_test(test_integrator_p2_1ang)

add_executable(test_integrator_p2_4D_2ang integrator_p2_4D_2ang.cc)
setup_test(test_integrator_p2_4D_2ang)

add_executable(test_integrator_p2_4D_3ang integrator_p2_4D_3ang.cc)
setup_test(test_integrator_p2_4D_3ang)

# ##############################################################################
# Benchmarks
# ##############################################################################

add_executable(benchmark_integrator_1D benchmark_integrator_1D.cc)
setup_benchmark(benchmark_integrator_1D)
