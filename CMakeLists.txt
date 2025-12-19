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

include(${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/cmake/common.cmake)

message(
  "${BoldWhite}
  __| |________________________________________________| |__
  __   ________________________________________________   __
    | |                                                | |
    | |  ██████╗ ██╗███████╗███████╗██████╗  ██████╗   | |
    | |  ██╔══██╗██║██╔════╝██╔════╝██╔══██╗██╔════╝   | |
    | |  ██║  ██║██║█████╗  █████╗  ██████╔╝██║  ███╗  | |
    | |  ██║  ██║██║██╔══╝  ██╔══╝  ██╔══██╗██║   ██║  | |
    | |  ██████╔╝██║██║     ██║     ██║  ██║╚██████╔╝  | |
    | |  ╚═════╝ ╚═╝╚═╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝   | |
  __| |________________________________________________| |__
  __   ________________________________________________   __
    | |                                                | |
${ColourReset}

This is the main CMake file for DiFfRG.
It will download and build all dependencies, including DiFfRG itself.
")

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

message(
  "${BoldWhite}Dependencies will be installed to ${DEPEND_INSTALL_PREFIX}${ColourReset}
  ")

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# ##############################################################################
# Find out which backends are available
# ##############################################################################

include(CheckLanguage)

check_language(Fortran)
check_language(CXX)
check_language(C)

check_language(CUDA)
# if the compiler id is NVHPC, then we have CUDA
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

set(HAVE_THREADS ON)
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
message("")

message("Available parallel backends:")
# make the on and off values green/yellow
if(ENABLE_MPI)
  message("  ${BoldGreen}MPI: ${ENABLE_MPI}${ColourReset}")
else()
  message("  ${BoldYellow}MPI: ${ENABLE_MPI}${ColourReset}")
endif()
if(HAVE_CUDA)
  message("  ${BoldGreen}CUDA: ${HAVE_CUDA}${ColourReset}")
else()
  message("  ${BoldYellow}CUDA: ${HAVE_CUDA}${ColourReset}")
endif()
if(HAVE_HIP)
  message("  ${BoldGreen}HIP: ${HAVE_HIP}${ColourReset}")
else()
  message("  ${BoldYellow}HIP: ${HAVE_HIP}${ColourReset}")
endif()
if(HAVE_THREADS)
  message("  ${BoldGreen}Threads: ${HAVE_THREADS}${ColourReset}")
else()
  message("  ${BoldYellow}Threads: ${HAVE_THREADS}${ColourReset}")
endif()
message("")

# ##############################################################################
# Dependencies
# ##############################################################################

include(ExternalProject)

option(BUILD_OpenBLAS "Build OpenBLAS" OFF)

# build general deps
ExternalProject_Add(
  general_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/general
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/general
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=-march=native
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DBUILD_OpenBLAS:BOOL=${BUILD_OpenBLAS}
    -DENABLE_MPI:BOOL=${ENABLE_MPI})

if(BUILD_OpenBLAS)
  if(APPLE)
    set(BLAS_LIBRARY_PATH
        "${DEPEND_INSTALL_PREFIX}/lib/libopenblasDiFfRG.dylib")
  else()
    set(BLAS_LIBRARY_PATH "${DEPEND_INSTALL_PREFIX}/lib/libopenblasDiFfRG.so")
  endif()
endif()

# build sundials
ExternalProject_Add(
  sundials_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/sundials
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/sundials
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=-march=native
    -DCMAKE_C_FLAGS:STRING=-march=native
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DENABLE_MPI:BOOL=${ENABLE_MPI})

# build autodiff
ExternalProject_Add(
  autodiff_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/autodiff
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/autodiff
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep sundials_dep
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
                   -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# build boost
ExternalProject_Add(
  boost_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/boost
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/boost
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
             -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

# check if HDF5 is available
find_package(HDF5 QUIET)
if(HDF5_FOUND)
  message("${BoldGreen}HDF5 found:${ColourReset} ${HDF5_LIBRARIES}")
else()
  message(
    "${BoldYellow}HDF5 not found.${ColourReset} DiFfRG will not be able to use HDF5 for I/O."
  )
endif()
if(HDF5_IS_PARALLEL)
  message("${Green}HDF5 has parallel support${ColourReset}")
else()
  message(
    STATUS
      "${Yellow}The provided HDF5 installation does not support parallel I/O.${ColourReset}"
  )
endif()
message("")

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
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 5.0.0
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/kokkos
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep sundials_dep autodiff_dep
  CMAKE_ARGS -DCMAKE_CXX_FLAGS=-fPIC
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD:STRING=20
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DKokkos_ARCH_NATIVE:BOOL=${Kokkos_ARCH_NATIVE}
    ${Kokkos_ARCH_LIST}
    -DCMAKE_CUDA_COMPILER:STRING=${CMAKE_CUDA_COMPILER}
    -DKokkos_ENABLE_CUDA:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_CUDA_CONSTEXPR:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_HIP:BOOL=${HAVE_HIP}
    -DKokkos_ENABLE_OPENMP:BOOL=OFF
    -DKokkos_ENABLE_THREADS:BOOL=${HAVE_THREADS}
    -DKokkos_ENABLE_SERIAL:BOOL=ON
    -DKokkos_ENABLE_TESTS:BOOL=OFF)

# ##############################################################################
# PETSc
# ##############################################################################

option(BUILD_PETSC "Build PETSc" OFF)

if(BUILD_PETSC)
  # build PETSc
  ExternalProject_Add(
    petsc_dep
    LOG_CONFIGURE ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
    DEPENDS general_dep sundials_dep autodiff_dep kokkos_dep
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/petsc
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/petsc
    INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    BUILD_COMMAND cmake -E echo "Skipping build step for PETSc."
    INSTALL_COMMAND cmake -E echo "Skipping install step for PETSc."
    # Propagate DEPEND_INSTALL_PREFIX to the dependencies
    CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
      -DOpenBLAS_LIBRARY:STRING=${BLAS_LIBRARY_PATH}
      -DHAVE_MPI:BOOL=${ENABLE_MPI})
  set(is_petsc_dep petsc_dep)
endif()

# ##############################################################################
# deal.II
# ##############################################################################

if(BUILD_OpenBLAS)
  set(dealii_lapack "-DLAPACK_LIBRARIES:STRING=${BLAS_LIBRARY_PATH}")
endif()

# build deal.II
ExternalProject_Add(
  deal.II_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  GIT_REPOSITORY https://github.com/dealii/dealii.git
  GIT_TAG v9.4.2
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/deal.II
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep boost_dep ${is_petsc_dep} sundials_dep
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_Fortran_COMPILER:STRING=${CMAKE_Fortran_COMPILER}
    -DCMAKE_BUILD_TYPE:STRING=DebugRelease
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
    -DCMAKE_POLICY_DEFAULT_CMP0167:STRING=NEW
    -DCMAKE_WARN_DEPRECATED:BOOL=OFF
    -DDEAL_II_CXX_FLAGS:STRING=-march=native
    -DDEAL_II_COMPONENT_EXAMPLES:BOOL=OFF
    -DDEAL_II_COMPONENT_DOCUMENTATION:BOOL=OFF
    -DDEAL_II_WITH_MPI:BOOL=${ENABLE_MPI}
    -DDEAL_II_WITH_KOKKOS:BOOL=OFF
    -DDEAL_II_WITH_UMFPACK:BOOL=ON
    -DDEAL_II_WITH_TASKFLOW:BOOL=OFF
    ${dealii_lapack}
    -DDEAL_II_WITH_LAPACK:BOOL=ON
    -DDEAL_II_WITH_HDF5:BOOL=${HDF5_FOUND}
    -DHDF5_INCLUDE_DIRS:PATH=${HDF5_INCLUDE_DIRS}
    -DPETSC_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    -DPETSC_ARCH:STRING=arch-linux-c-opt
    -DCMAKE_CXX_STANDARD:STRING=17
    -DTBB_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DSUNDIALS_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DBOOST_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DCMAKE_PREFIX_PATH:STRING=${DEPEND_INSTALL_PREFIX}
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
    -DENABLE_HDF5:BOOL=${HDF5_FOUND}
    -DBUNDLED_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DDiFfRG_TEST:BOOL=${DiFfRG_TEST}
    -DDiFfRG_DOCUMENTATION:BOOL=${DiFfRG_DOCUMENTATION}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})

# ##############################################################################
# Forwarding DiFfRG tests for CTest to discover them
# ##############################################################################

option(DiFfRG_TEST "Build tests" OFF)
if(DiFfRG_TEST)
  enable_testing()
  add_subdirectory(DiFfRG/tests)
endif()

# ##############################################################################
# Finish
# ##############################################################################

add_custom_target(
  finish ALL
  DEPENDS DiFfRG
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/cmake/finish.sh
  VERBATIM)
