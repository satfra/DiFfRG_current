# ##############################################################################
# Setup additional include directories and Catch2
# ##############################################################################

include_directories(./)
# include(../cmake/setup_build_system.cmake)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.1)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

find_package(DiFfRG HINTS ${CMAKE_PREFIX_PATH})

function(setup_test SOURCE)
  # Create a test executable from the source file, removing the extension
  get_filename_component(TARGET ${SOURCE} NAME_WE)
  add_executable(${TARGET} ${SOURCE})
  target_link_libraries(${TARGET} PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
  catch_discover_tests(${TARGET})
endfunction()

function(setup_benchmark SOURCE)
  # Create a test executable from the source file, removing the extension
  get_filename_component(TARGET ${SOURCE} NAME_WE)
  add_executable(${TARGET} ${SOURCE})
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
