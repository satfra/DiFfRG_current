# ##############################################################################
# Tests for basic integrators
# ##############################################################################

add_executable(test_integrator_lat_1D integrator_lat_1D.cc)
setup_test(test_integrator_lat_1D)

add_executable(test_integrator_lat_2D integrator_lat_2D.cc)
setup_test(test_integrator_lat_2D)

add_executable(test_integrator_lat_3D integrator_lat_3D.cc)
setup_test(test_integrator_lat_3D)

add_executable(test_integrator_lat_4D integrator_lat_4D.cc)
setup_test(test_integrator_lat_4D)

# ##############################################################################
# Benchmarks
# ##############################################################################

add_executable(benchmark_integrator_lat_4D benchmark_integrator_lat_4D.cc)
setup_benchmark(benchmark_integrator_lat_4D)
