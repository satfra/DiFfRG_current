cmake_minimum_required(VERSION 3.20)

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

# ##############################################################################
# Find out which backends are available
# ##############################################################################

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

check_language(OpenMP)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(HAVE_OPENMP ON)
  set(HAVE_THREADS OFF)
else()
  set(HAVE_OPENMP OFF)
  set(HAVE_THREADS ON)
endif()
# if ENABLE_MPI is not set, set it to OFF
if(NOT DEFINED ENABLE_MPI)
  find_package(MPI)
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

message(STATUS "MPI: ${ENABLE_MPI}")
message(STATUS "OpenMP: ${HAVE_OPENMP}")
message(STATUS "CUDA: ${HAVE_CUDA}")
message(STATUS "HIP: ${HAVE_HIP}")
message(STATUS "Threads: ${HAVE_THREADS}")

# ##############################################################################
# Dependencies
# ##############################################################################

include(ExternalProject)

# build general deps
ExternalProject_Add(
  general_dep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/general
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/general
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
                   -DENABLE_MPI:BOOL=${ENABLE_MPI})

# build autodiff
ExternalProject_Add(
  autodiff_dep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/autodiff
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/autodiff
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# build boost
ExternalProject_Add(
  boost_dep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/boost
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/boost
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# ##############################################################################
# Kokkos
# ##############################################################################

if(NOT DEFINED Kokkos_ARCH)
  set(Kokkos_ARCH_NATIVE
      ON
      CACHE BOOL "Enable Kokkos native architecture")
else()
  set(Kokkos_ARCH_NATIVE
      OFF
      CACHE BOOL "Enable Kokkos native architecture")
endif()

if(NOT DEFINED Kokkos_ARCH_LIST)
  message(
    WARNING
      "Kokkos_ARCH_LIST not set. GPU architecture must be detectable when building Kokkos."
  )
  set(Kokkos_ARCH_LIST "")
else()
  # prepend every element with a -D and postpend with a :BOOL=ON
  string(REPLACE ";" ";-D" Kokkos_ARCH_LIST ";${Kokkos_ARCH_LIST}")
  # remove the first element
  string(SUBSTRING "${Kokkos_ARCH_LIST}" 1 -1 Kokkos_ARCH_LIST)
  string(REPLACE ";" ":BOOL=ON " Kokkos_ARCH_LIST "${Kokkos_ARCH_LIST};")
  # remove the last element
  string(SUBSTRING "${Kokkos_ARCH_LIST}" 0 -1 Kokkos_ARCH_LIST)
endif()

# add the Kokkos dependency to the build
ExternalProject_Add(
  kokkos_dep
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG develop
  # GIT_TAG 4.6.01
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/kokkos
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_CXX_FLAGS=-fPIC
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD:STRING=20
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DKokkos_ARCH_NATIVE:BOOL=${Kokkos_ARCH_NATIVE}
    ${Kokkos_ARCH_LIST}
    -DKokkos_ENABLE_CUDA:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_CUDA_CONSTEXPR:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_HIP:BOOL=${HAVE_HIP}
    -DKokkos_ENABLE_OPENMP:BOOL=${HAVE_OPENMP}
    -DKokkos_ENABLE_THREADS:BOOL=${HAVE_THREADS}
    -DKokkos_ENABLE_SERIAL:BOOL=ON
    -DKokkos_ENABLE_TESTS:BOOL=OFF
    )

# ##############################################################################
# PETSc
# ##############################################################################

# build PETSc
ExternalProject_Add(
  petsc_dep
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/petsc
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/petsc
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  BUILD_COMMAND cmake -E echo "Skipping build step for PETSc."
  INSTALL_COMMAND cmake -E echo "Skipping install step for PETSc."
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DHAVE_OPENMP:BOOL=${HAVE_OPENMP} -DHAVE_MPI:BOOL=${ENABLE_MPI})

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
  DEPENDS general_dep boost_dep petsc_dep
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=DebugRelease
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.31
    -DCMAKE_POLICY_DEFAULT_CMP0167:STRING=NEW
    -DCMAKE_WARN_DEPRECATED:BOOL=OFF
    -DDEAL_II_COMPONENT_EXAMPLES:BOOL=OFF
    -DDEAL_II_COMPONENT_DOCUMENTATION:BOOL=OFF
    -DDEAL_II_WITH_MPI:BOOL=${ENABLE_MPI}
    -DDEAL_II_WITH_KOKKOS:BOOL=OFF
    -DDEAL_II_WITH_UMFPACK:BOOL=ON
    -DDEAL_II_WITH_TASKFLOW:BOOL=OFF
    -DDEAL_II_WITH_LAPACK:BOOL=ON
    -DPETSC_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    -DPETSC_ARCH:STRING=arch-linux-c-opt
    -DCMAKE_CXX_STANDARD:STRING=17
    -DTBB_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DSUNDIALS_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DBOOST_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# create a directory necessary for using deal.II
file(MAKE_DIRECTORY ${DEPEND_INSTALL_PREFIX}/include/deal.II/bundled)

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
  CMAKE_CACHE_ARGS
    -DENABLE_MPI:BOOL=${ENABLE_MPI}
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
