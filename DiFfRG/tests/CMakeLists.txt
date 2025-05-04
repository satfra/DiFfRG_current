# ##############################################################################
# Setup additional include directories and Catch2
# ##############################################################################

include_directories(./)

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH
    SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# Find Catch2 Testing Framework
cpmaddpackage(
  NAME
  Catch2
  GITHUB_REPOSITORY
  catchorg/Catch2
  VERSION
  3.8.1
  OPTIONS
  "CATCH_INSTALL_DOCS OFF"
  "CATCH_INSTALL_HELPERS OFF")
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

function(setup_test TARGET)
  setup_target(${TARGET})
  target_link_libraries(${TARGET} PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
  catch_discover_tests(${TARGET})
endfunction()

function(setup_benchmark TARGET)
  setup_target(${TARGET})
  target_link_libraries(${TARGET} PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
endfunction()

include(CTest)
include(Catch)

# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(common)
add_subdirectory(discretization)
add_subdirectory(model)
add_subdirectory(physics)
add_subdirectory(timestepping)
