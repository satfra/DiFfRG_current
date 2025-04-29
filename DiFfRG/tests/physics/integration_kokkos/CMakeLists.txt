# ##############################################################################
# Tests
# ##############################################################################

add_executable(test_integrator_1D integrator_1D.cc)
setup_test(test_integrator_1D)

# ##############################################################################
# Benchmarks
# ##############################################################################

add_executable(benchmark_integrator_1D benchmark_integrator_1D.cc)
setup_benchmark(benchmark_integrator_1D)
