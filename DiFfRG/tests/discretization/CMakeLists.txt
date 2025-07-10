# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(DG)
add_subdirectory(CG)
add_subdirectory(grid)
add_subdirectory(data)

# ##############################################################################
# Tests
# ##############################################################################

setup_test(mesh.cc)
setup_test(mesh_adaptor.cc)
setup_test(EoM_finding.cc)
