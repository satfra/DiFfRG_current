# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(FEM)
add_subdirectory(grid)
add_subdirectory(data)
add_subdirectory(coordinates)

# ##############################################################################
# Tests
# ##############################################################################

setup_test(mesh.cc)
setup_test(mesh_adaptor.cc)
setup_test(EoM_finding.cc)
setup_test(mesh_configuration_test.cpp)
