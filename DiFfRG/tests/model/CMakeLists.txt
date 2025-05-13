# ##############################################################################
# Test automatic differentiation
# ##############################################################################

add_executable(test_ad test_ad.cc)
setup_test(test_ad)

add_executable(test_component_descriptor component_descriptor.cc)
setup_test(test_component_descriptor)
