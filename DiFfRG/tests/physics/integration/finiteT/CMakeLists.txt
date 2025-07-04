# ##############################################################################
# Tests for basic integrators
# ##############################################################################

add_executable(test_quadrature_integrator_fT quadrature_integrator_fT.cc)
setup_test(test_quadrature_integrator_fT)

# ##############################################################################
# Tests for derived integrators
# ##############################################################################

add_executable(test_integrator_fT_p2 integrator_fT_p2.cc)
setup_test(test_integrator_fT_p2)

add_executable(test_integrator_fT_p2_1ang integrator_fT_p2_1ang.cc)
setup_test(test_integrator_fT_p2_1ang)

add_executable(test_integrator_fT_p2_4D_2ang integrator_fT_p2_4D_2ang.cc)
setup_test(test_integrator_fT_p2_4D_2ang)
