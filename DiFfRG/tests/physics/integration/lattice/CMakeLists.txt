# ##############################################################################
# Tests for basic integrators
# ##############################################################################

setup_test(integrator_lat_1D.cc)
setup_test(integrator_lat_2D.cc)
setup_test(integrator_lat_3D.cc)
setup_test(integrator_lat_4D.cc)

# ##############################################################################
# Benchmarks
# ##############################################################################

setup_benchmark(benchmark_integrator_lat_4D.cc)
