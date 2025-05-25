# ##############################################################################
# Tests for basic integrators
# ##############################################################################

add_executable(test_quadrature_integrator quadrature_integrator.cc)
setup_test(test_quadrature_integrator)

add_executable(test_distributed_quadrature_integrator distributed_quadrature_integrator.cc)
setup_test(test_distributed_quadrature_integrator)

# ##############################################################################
# # Add subdirectories
# ##############################################################################

add_subdirectory(vacuum)
add_subdirectory(finiteT)
add_subdirectory(lattice)

# ##############################################################################
# Benchmarks
# ##############################################################################

add_executable(benchmark_integrator_1D benchmark_integrator_1D.cc)
setup_benchmark(benchmark_integrator_1D)
