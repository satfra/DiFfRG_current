# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(solver)

# ##############################################################################
# Test timesteppers
# ##############################################################################

add_executable(test_sundials_ida test_sundials_ida.cc)
setup_test(test_sundials_ida)

add_executable(test_implicit_euler test_implicit_euler.cc)
setup_test(test_implicit_euler)

add_executable(test_explicit_euler test_explicit_euler.cc)
setup_test(test_explicit_euler)

add_executable(test_trbdf2 test_trbdf2.cc)
setup_test(test_trbdf2)