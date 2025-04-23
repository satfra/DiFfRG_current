# ##############################################################################
# Setup directories
# ##############################################################################

# We need to find the bundle directory, which contains several external
# dependencies
if(${CMAKE_PROJECT_NAME} STREQUAL "DiFfRG")
  # If we are building DiFfRG as a standalone project, we need to set the base
  # directory
  set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT DEFINED BUNDLED_DIR)
    set(BUNDLED_DIR ${BASE_DIR}/../external)
  endif()
else()
  # If we are building a DiFfRG-based project, we need to set the bundle
  # directory relative to the DiFfRG base directory
  set(BASE_DIR ${DiFfRG_BASE_DIR})
  set(BUNDLED_DIR ${BASE_DIR}/bundled)
endif()
message(STATUS "DiFfRG include directory: ${BASE_DIR}/include")
message(STATUS "DiFfRG bundle directory: ${BUNDLED_DIR}")

# ##############################################################################
# Set standard and language
# ##############################################################################

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
else()
  if(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "C++ standard must be at least 20")
  endif()
endif()

set(CMAKE_CXX_STANDARD_REQUIRED On)
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
find_package(
  deal.II
  9.4.2
  QUIET
  REQUIRED
  HINTS
  ${DEAL_II_DIR}
  ${BUNDLED_DIR}/dealii_install)
deal_ii_initialize_cached_variables()
message(STATUS "Found deal.II in  ${deal.II_DIR}")

# Find TBB
find_package(TBB 2022.0.0 REQUIRED HINTS ${BUNDLED_DIR}/oneTBB_install)
message(STATUS "Found TBB in ${TBB_DIR}")

# Find TBB
find_package(Kokkos REQUIRED HINTS ${BUNDLED_DIR}/kokkos_install)
message(STATUS "Found Kokkos in ${Kokkos_DIR}")

# Find Boost
find_package(Boost 1.80 REQUIRED HINTS ${BUNDLED_DIR}/boost_install
             COMPONENTS thread random iostreams math serialization system)
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# ##############################################################################
# CPM dependencies
# ##############################################################################

# Get CPM for package management
list(APPEND CMAKE_MODULE_PATH "${BASE_DIR}/cmake")
include(${BASE_DIR}/cmake/CPM.cmake)
# If CPM_SOURCE_CACHE is set to OFF, we set it to the default cache directory
if("${CPM_SOURCE_CACHE}" STREQUAL "OFF" OR NOT DEFINED CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{HOME}/.cache/CPM)
endif()
message(STATUS "CPM source cache directory: ${CPM_SOURCE_CACHE}")

# Get Eigen3
cpmaddpackage(
  NAME
  Eigen3
  VERSION
  3.4.0
  URL
  https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  # Eigen's CMakelists are not intended for library use
  DOWNLOAD_ONLY
  YES)
if(Eigen3_ADDED)
  add_library(Eigen3 INTERFACE IMPORTED)
  target_include_directories(Eigen3 INTERFACE ${Eigen3_SOURCE_DIR})
  add_library(Eigen3::Eigen ALIAS Eigen3)
endif()

# Get GSL
find_package(GSL)
if(NOT GSL_FOUND)
  cpmfindpackage(
    NAME
    gsl
    GITHUB_REPOSITORY
    ampl/gsl
    VERSION
    2.5.0
    OPTIONS
    "GSL_DISABLE_TESTS 1"
    "DOCUMENTATION OFF")
  add_library(GSL::gsl ALIAS gsl)
else()
  message(STATUS "GSL found: ${GSL_INCLUDE_DIR}")
endif()

# Get rapidcsv
cpmaddpackage(
  NAME
  rapidcsv
  GITHUB_REPOSITORY
  d99kris/rapidcsv
  VERSION
  8.84
  DOWNLOAD_ONLY
  YES)
include_directories(SYSTEM ${rapidcsv_SOURCE_DIR}/src)

# Get autodiff
cpmaddpackage(
  NAME
  autodiff
  GITHUB_REPOSITORY
  autodiff/autodiff
  GIT_TAG
  v1.1.0
  PATCHES
  "autodiff.patch"
  OPTIONS
  "CMAKE_CUDA_ARCHITECTURES native"
  "AUTODIFF_BUILD_TESTS OFF"
  "AUTODIFF_BUILD_EXAMPLES OFF"
  "AUTODIFF_BUILD_DOCS OFF"
  "AUTODIFF_BUILD_PYTHON OFF"
  "Eigen3_DIR ${Eigen3_BINARY_DIR}")

# Get spdlog
cpmaddpackage(
  NAME
  spdlog
  GITHUB_REPOSITORY
  gabime/spdlog
  VERSION
  1.14.1
  OPTIONS
  "CMAKE_BUILD_TYPE Release"
  "SPDLOG_INSTALL ON")

# ##############################################################################
# Convenience functions
# ##############################################################################

set(USE_CUDA OFF)

function(setup_target TARGET)
  # target_link_libraries(${TARGET} PUBLIC dealii::dealii)
  deal_ii_setup_target(${TARGET})

  # Check if the target is DiFfRG
  if(${TARGET} STREQUAL "DiFfRG")
    target_include_directories(${TARGET} PRIVATE ${autodiff_SOURCE_DIR})
  else()
    target_link_libraries(${TARGET} autodiff::autodiff)
  endif()

  target_link_libraries(${TARGET} GSL::gsl)
  target_link_libraries(${TARGET} Eigen3)
  target_link_libraries(${TARGET} spdlog::spdlog)
  target_link_libraries(${TARGET} ${Boost_LIBRARIES})
  target_link_libraries(${TARGET} TBB::tbb)
  target_link_libraries(${TARGET} Kokkos::kokkos)
endfunction()
