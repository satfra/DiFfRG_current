# ##############################################################################
# Tests
# ##############################################################################

add_executable(test_linear_interpolator_1D linear_interpolator_1D.cc)
setup_test(test_linear_interpolator_1D)

add_executable(test_linear_interpolator_2D linear_interpolator_2D.cc)
setup_test(test_linear_interpolator_2D)

add_executable(test_linear_interpolator_3D linear_interpolator_3D.cc)
setup_test(test_linear_interpolator_3D)
