# ##############################################################################
# Tests
# ##############################################################################

setup_test(cg.cc)
setup_test(dg.cc)
setup_test(ddg.cc)
setup_test(ldg.cc)

setup_benchmark(benchmark_cg.cc)
setup_benchmark(benchmark_dg.cc)
setup_benchmark(benchmark_ddg.cc)
setup_benchmark(benchmark_ldg.cc)
