# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(FEM)
add_subdirectory(FV)
add_subdirectory(grid)
add_subdirectory(data)
add_subdirectory(coordinates)

# ##############################################################################
# Tests
# ##############################################################################

setup_test(mesh.cc)
setup_test(mesh_adaptor.cc)
setup_test(mesh_configuration_test.cpp)
setup_test(eom_finding.cc)
