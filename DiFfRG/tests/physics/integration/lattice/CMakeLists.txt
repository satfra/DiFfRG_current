# ##############################################################################
# Tests for basic integrators
# ##############################################################################

setup_test(integrator_lat_1d.cc)
setup_test(integrator_lat_2d.cc)
setup_test(integrator_lat_3d.cc)
setup_test(integrator_lat_4d.cc)

# ##############################################################################
# Benchmarks
# ##############################################################################

setup_benchmark(benchmark_integrator_lat_4d.cc)
