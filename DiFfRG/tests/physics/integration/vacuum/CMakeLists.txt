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
