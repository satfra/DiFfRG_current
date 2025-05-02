cmake_minimum_required(VERSION 3.27)

project(DiFfRG)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# get the user's home directory
set(HOME_DIR $ENV{HOME})

# if the install prefix is not set, set DEPEND_INSTALL_PREFIX to the build
# directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message(
    WARNING
      "CMAKE_INSTALL_PREFIX not set. Setting to ${HOME_DIR}/.local/share/DiFfRG"
  )
  set(CMAKE_INSTALL_PREFIX
      ${HOME_DIR}/.local/share/DiFfRG
      CACHE PATH "Install prefix" FORCE)
endif()
set(DEPEND_INSTALL_PREFIX
    ${CMAKE_INSTALL_PREFIX}/bundled/
    CACHE PATH "Install prefix for dependencies")

message(STATUS "Dependencies will be installed to ${DEPEND_INSTALL_PREFIX}")

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Set up external dependencies
include(ExternalProject)

# ##############################################################################
# Various basics
# ##############################################################################

# build general deps
ExternalProject_Add(
  general_dep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/general
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/general
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  LOG_DIR ${CMAKE_CURRENT_BINARY_DIR}/logs/general/
  BUILD_COMMAND make all
  INSTALL_COMMAND make install
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# build autodiff
ExternalProject_Add(
  autodiff_dep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/autodiff
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/autodiff
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  LOG_DIR ${CMAKE_CURRENT_BINARY_DIR}/logs/autodiff/
  DEPENDS general_dep
  BUILD_COMMAND make all
  INSTALL_COMMAND make install
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# build boost
ExternalProject_Add(
  boost_dep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/boost
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/boost
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  LOG_DIR ${CMAKE_CURRENT_BINARY_DIR}/logs/boost/
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# ##############################################################################
# Kokkos
# ##############################################################################

# Find out which backends are available
include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(HAVE_CUDA ON)
  set(HAVE_HIP OFF)
else()
  set(HAVE_CUDA OFF)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    set(HAVE_HIP ON)
  else()
    set(HAVE_HIP OFF)
  endif()
endif()
message(STATUS "HIP: ${HAVE_HIP}")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()
check_language(OpenMP)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(HAVE_OPENMP ON)
  set(HAVE_THREADS OFF)
else()
  set(HAVE_OPENMP OFF)
  set(HAVE_THREADS ON)
endif()

message(STATUS "OpenMP: ${HAVE_OPENMP}")
message(STATUS "CUDA: ${HAVE_CUDA}")
message(STATUS "HIP: ${HAVE_HIP}")
message(STATUS "Threads: ${HAVE_THREADS}")

# add the Kokkos dependency to the build
ExternalProject_Add(
  kokkos_dep
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 4.6.01
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/kokkos
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_CXX_STANDARD:STRING=20
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DKokkos_ARCH_NATIVE:BOOL=ON
    -DKokkos_ENABLE_CUDA:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_CUDA_CONSTEXPR:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_HIP:BOOL=${HAVE_HIP}
    -DKokkos_ENABLE_OPENMP:BOOL=${HAVE_OPENMP}
    -DKokkos_ENABLE_THREADS:BOOL=${HAVE_THREADS}
    -DKokkos_ENABLE_SERIAL:BOOL=ON
    -DKokkos_ENABLE_TESTS:BOOL=OFF
    -DCMAKE_CXX_FLAGS:STRING=-fPIC)

# ##############################################################################
# deal.II
# ##############################################################################

# build deal.II
ExternalProject_Add(
  deal.II_dep
  GIT_REPOSITORY https://github.com/dealii/dealii.git
  GIT_TAG v9.4.2
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/deal.II
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep boost_dep
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  LOG_DIR ${CMAKE_CURRENT_BINARY_DIR}/logs/deal.II
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=DebugRelease
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
    -DENABLE_MPI:BOOL=OFF
    -DDEAL_II_COMPONENT_EXAMPLES:BOOL=OFF
    -DDEAL_II_COMPONENT_DOCUMENTATION:BOOL=OFF
    -DDEAL_II_WITH_MPI:BOOL=OFF
    -DDEAL_II_WITH_KOKKOS:BOOL=OFF
    -DDEAL_II_WITH_UMFPACK:BOOL=ON
    -DDEAL_II_WITH_TASKFLOW:BOOL=OFF
    -DCMAKE_CXX_STANDARD:STRING=17
    -DTBB_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DSUNDIALS_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DBOOST_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# ##############################################################################
# DiFfRG
# ##############################################################################

option(DiFfRG_TEST "Build DiFfRG tests" OFF)
option(DiFfRG_DOCUMENTATION "Build DiFfRG documentation" ON)

# build the actual DiFfRG library
ExternalProject_Add(
  DiFfRG
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/DiFfRG/
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  DEPENDS general_dep boost_dep deal.II_dep kokkos_dep autodiff_dep
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  CMAKE_CACHE_ARGS
    -DENABLE_MPI:BOOL=OFF
    -DBUNDLED_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DDiFfRG_TEST:BOOL=${DiFfRG_TEST}
    -DDiFfRG_DOCUMENTATION:BOOL=${DiFfRG_DOCUMENTATION}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})

# ##############################################################################
# Finish
# ##############################################################################

add_custom_target(
  finish ALL
  DEPENDS DiFfRG
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/cmake/finish.sh
  VERBATIM)
