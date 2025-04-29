# ##############################################################################
# Test
# ##############################################################################

add_subdirectory(quadrature)

add_executable(test_configuration_helper test_configuration_helper.cc)
setup_test(test_configuration_helper)

add_executable(test_autodiff test_autodiff.cc)
setup_test(test_autodiff)

add_executable(test_complex_autodiff test_complex_autodiff.cc)
setup_test(test_complex_autodiff)
