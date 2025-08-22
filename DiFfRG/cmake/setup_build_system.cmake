# ##############################################################################
# Setup directories
# ##############################################################################

# We need to find the bundle directory, which contains several external
# dependencies
if(${CMAKE_PROJECT_NAME} STREQUAL "DiFfRG")
  if(NOT DEFINED ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_FOUND)
      set(ENABLE_MPI
          ON
          CACHE BOOL "Enable MPI support")
    else()
      set(ENABLE_MPI
          OFF
          CACHE BOOL "Enable MPI support")
    endif()
  endif()

  # If we are building DiFfRG as a standalone project, we need to set the base
  # directory
  set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(DiFfRG_MPI ${ENABLE_MPI})
else()
  # If we are building a DiFfRG-based project, we need to set the bundle
  # directory relative to the DiFfRG base directory
  set(BASE_DIR ${DiFfRG_BASE_DIR})
endif()

set(CMAKE_PREFIX_PATH "${BUNDLED_DIR};${BUNDLED_DIR}/lib;${CMAKE_PREFIX_PATH}")

link_directories(${BUNDLED_DIR}/lib/)
include_directories(SYSTEM ${BUNDLED_DIR}/include)

message(STATUS "DiFfRG include directory: ${BASE_DIR}/include")
message(STATUS "DiFfRG bundle directory: ${BUNDLED_DIR}")
message(STATUS "MPI support has been set to ${DiFfRG_MPI}")

# ##############################################################################
# Set standard and language
# ##############################################################################

set(CMAKE_CXX_STANDARD_REQUIRED On)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
else()
  if(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "C++ standard must be at least 20")
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_language(CXX)

# By default, we build in Release mode, i.e. if the user does not make any other
# choice. After all, even if the user is unaware of cmake build types, we want
# to provide optimal performance.
if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, defaulting to Release")
endif()

# ##############################################################################
# Direct external dependencies
# ##############################################################################

# Find deal.II
find_package(deal.II 9.4.2 QUIET REQUIRED HINTS ${BUNDLED_DIR})
deal_ii_initialize_cached_variables()
message(STATUS "Found deal.II in  ${deal.II_DIR}")
# deal.II link dir

# Find TBB
find_package(TBB 2022.0.0 REQUIRED HINTS ${BUNDLED_DIR})
message(STATUS "Found TBB in ${TBB_DIR}")

# Find Kokkos
find_package(Kokkos REQUIRED HINTS ${BUNDLED_DIR})
message(STATUS "Found Kokkos in ${Kokkos_DIR}")

# Find Boost
find_package(
  Boost 1.81 REQUIRED HINTS ${BUNDLED_DIR}/ ${BUNDLED_DIR}/boost_install/lib/
  COMPONENTS thread iostreams serialization system)
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# Other dependencies
find_package(Eigen3 3.4.0 REQUIRED HINTS ${BUNDLED_DIR})
find_package(GSL REQUIRED)
find_package(autodiff 1.1.0 REQUIRED HINTS ${BUNDLED_DIR})
find_package(spdlog 1.14.1 REQUIRED HINTS ${BUNDLED_DIR})
find_package(h5cpp 0.7.1 QUIET HINTS ${BUNDLED_DIR})
if(h5cpp_FOUND)
  message(STATUS "Found h5cpp in ${h5cpp_DIR}")
  add_compile_definitions(H5CPP)
endif()

if(${DiFfRG_MPI})
  find_package(MPI REQUIRED)
endif()

# ##############################################################################
# Convenience functions
# ##############################################################################

# We redefine the deal_ii_setup_target function here such that we can choose
# precisely how to propagate flags and other details
function(setup_dealii TARGET)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(${TARGET} PUBLIC deal_II.g)
    target_link_libraries(${TARGET} INTERFACE deal_II.g)
    set(_build "DEBUG")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_libraries(${TARGET} PUBLIC deal_II)
    target_link_libraries(${TARGET} INTERFACE deal_II)
    set(_build "RELEASE")
  endif()
  target_include_directories(${TARGET} SYSTEM PUBLIC ${DEAL_II_INCLUDE_DIRS})

  set(_cflags "${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_${_build}}")
  # remove c++20 flag and O2 flag - CMake adds them automatically and we thus
  # avoid the nvcc_wrapper warnings
  string(REPLACE "-std=c++20" "" _cflags ${_cflags})
  string(REPLACE "-O2" "" _cflags ${_cflags})
  separate_arguments(_cflags)
  target_compile_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${_cflags}>)

  set(_lflags "${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_${_build}}")
  separate_arguments(_lflags)
  target_link_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${_lflags}>)
endfunction()

function(setup_target TARGET)
  setup_dealii(${TARGET})

  # Check if the target is DiFfRG
  if(${TARGET} STREQUAL "DiFfRG")
    target_include_directories(${TARGET} PRIVATE ${autodiff_SOURCE_DIR})
  else()
    target_link_libraries(${TARGET} PUBLIC autodiff::autodiff)
  endif()

  # Do not warn about missing braces
  target_compile_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
                                          -Wno-missing-braces>)

  target_link_libraries(${TARGET} PUBLIC GSL::gsl)
  target_link_libraries(${TARGET} PUBLIC Eigen3)
  target_link_libraries(${TARGET} PUBLIC spdlog::spdlog)
  target_link_libraries(${TARGET} PUBLIC ${Boost_LIBRARIES})
  target_link_libraries(${TARGET} PUBLIC TBB::tbb)
  target_link_libraries(${TARGET} PUBLIC Kokkos::kokkos)
  target_link_libraries(${TARGET} PUBLIC petsc)

  if(${DiFfRG_MPI})
    target_link_libraries(${TARGET} PUBLIC MPI::MPI_CXX)
    target_compile_definitions(${TARGET} PUBLIC HAVE_MPI)
  endif()

  if(NOT ${CMAKE_BUILD_TYPE} STREQUAL Debug)
    target_compile_options(
      ${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-march=native -ffast-math
                       -ffp-contract=fast -fno-finite-math-only >)
    target_compile_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                                            --use_fast_math>)
  endif()

  target_compile_definitions(${TARGET} PUBLIC _HAS_AUTO_PTR_ETC=0)
endfunction()
