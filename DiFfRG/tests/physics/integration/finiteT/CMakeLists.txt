# ##############################################################################
# Tests for basic integrators
# ##############################################################################

add_executable(test_integrator_fT_1D integrator_fT_1D.cc)
setup_test(test_integrator_fT_1D)

add_executable(test_integrator_fT_2D integrator_fT_2D.cc)
setup_test(test_integrator_fT_2D)

add_executable(test_integrator_fT_3D integrator_fT_3D.cc)
setup_test(test_integrator_fT_3D)

add_executable(test_integrator_fT_4D integrator_fT_4D.cc)
setup_test(test_integrator_fT_4D)

# ##############################################################################
# Tests for derived integrators
# ##############################################################################

add_executable(test_integrator_fT_p2 integrator_fT_p2.cc)
setup_test(test_integrator_fT_p2)

add_executable(test_integrator_fT_p2_1ang integrator_fT_p2_1ang.cc)
setup_test(test_integrator_fT_p2_1ang)

add_executable(test_integrator_fT_p2_4D_2ang integrator_fT_p2_4D_2ang.cc)
setup_test(test_integrator_fT_p2_4D_2ang)
