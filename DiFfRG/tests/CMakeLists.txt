# ##############################################################################
# Setup additional include directories and Catch2
# ##############################################################################

include_directories(./)

Include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

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
