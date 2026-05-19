# ##############################################################################
# Tests for the header-only hdf5lib wrapper.
# ##############################################################################

# These tests only need the wrapper headers (under DiFfRG/include/hdf5lib) and
# the HDF5 C library. They deliberately do not depend on the rest of DiFfRG, so
# they can validate the wrapper in isolation.

add_executable(hdf5lib_wrapper wrapper.cc)
target_include_directories(hdf5lib_wrapper PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(hdf5lib_wrapper PRIVATE
  Catch2::Catch2WithMain
  hdf5-static)
target_include_directories(hdf5lib_wrapper PRIVATE ${HDF5_INCLUDE_DIRS})
catch_discover_tests(hdf5lib_wrapper)
