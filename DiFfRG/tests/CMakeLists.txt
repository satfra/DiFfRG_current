######################################################################
# Setup additional include directories and Catch2
######################################################################

include_directories(./)

find_package(Catch2 REQUIRED PATHS ../../external/Catch2_install)
include(CTest)
include(Catch)

######################################################################
# Add subdirectories
######################################################################

add_subdirectory(common)
add_subdirectory(discretization)
add_subdirectory(model)
add_subdirectory(physics)
add_subdirectory(timestepping)
