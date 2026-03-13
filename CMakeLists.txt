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
    "${BoldYellow}CMAKE_INSTALL_PREFIX not set. Setting to ${HOME_DIR}/.local/share/DiFfRG${ColourReset}"
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

# Pre-flight check: git is required for ExternalProject git clones
find_program(GIT_EXECUTABLE git)
if(NOT GIT_EXECUTABLE)
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  'git' is required but was not found.\n"
      "======================================================================\n"
      "  DiFfRG uses git to fetch external dependencies during the build.\n"
      "  Install it using your package manager:\n"
      "    Ubuntu/Debian:  sudo apt install git\n"
      "    Arch Linux:     sudo pacman -S git\n"
      "    Rocky/RHEL:     sudo dnf install git\n"
      "    macOS:          brew install git\n"
      "======================================================================\n"
  )
endif()

check_language(Fortran)
check_language(CXX)
check_language(C)

# ##############################################################################
# Pre-flight checks: detect missing system dependencies early
# ##############################################################################

# Fortran compiler — required by deal.II for LAPACK
if(NOT CMAKE_Fortran_COMPILER)
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  Fortran compiler not found.\n"
      "======================================================================\n"
      "  deal.II requires a Fortran compiler (gfortran or flang) for LAPACK.\n"
      "\n"
      "  Install it using your package manager:\n"
      "    Ubuntu/Debian:  sudo apt install gfortran\n"
      "    Arch Linux:     sudo pacman -S gcc-fortran\n"
      "    Rocky/RHEL:     sudo dnf install gcc-gfortran\n"
      "    macOS:          brew install gcc\n"
      "  Or, if using Clang/LLVM:  install flang\n"
      "======================================================================\n"
  )
endif()

# GSL — required by DiFfRG
find_package(GSL QUIET)
if(NOT GSL_FOUND)
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  Required system dependency not found: GSL\n"
      "======================================================================\n"
      "  The GNU Scientific Library (GSL) must be installed on your system.\n"
      "\n"
      "  Install it using your package manager:\n"
      "    Ubuntu/Debian:  sudo apt install libgsl-dev\n"
      "    Arch Linux:     sudo pacman -S gsl\n"
      "    Rocky/RHEL:     sudo dnf install gsl-devel\n"
      "    macOS:          brew install gsl\n"
      "======================================================================\n"
  )
endif()

# LAPACK/BLAS — required by deal.II (skip if building OpenBLAS)
option(BUILD_OpenBLAS "Build OpenBLAS" OFF)
if(NOT BUILD_OpenBLAS)
  find_package(LAPACK QUIET)
  if(NOT LAPACK_FOUND)
    message(
      FATAL_ERROR
        "\n"
        "======================================================================\n"
        "  Required system dependency not found: LAPACK/BLAS\n"
        "======================================================================\n"
        "  deal.II requires a LAPACK/BLAS implementation.\n"
        "\n"
        "  Either install a system LAPACK:\n"
        "    Ubuntu/Debian:  sudo apt install libopenblas-dev\n"
        "    Arch Linux:     sudo pacman -S blas-openblas\n"
        "    Rocky/RHEL:     sudo dnf install openblas-devel\n"
        "    macOS:          brew install openblas\n"
        "\n"
        "  Or let DiFfRG build OpenBLAS for you:\n"
        "    cmake .. -DBUILD_OpenBLAS=ON\n"
        "======================================================================\n"
    )
  else()
    # Verify that LAPACK actually links correctly
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
    check_c_source_compiles(
      "
      extern void dgemm_(void);
      extern void dgetrf_(void);
      int main() { dgemm_(); dgetrf_(); return 0; }
    "
      LAPACK_LINKS_OK)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(NOT LAPACK_LINKS_OK)
      message(
        WARNING
          "\n" "  LAPACK was found but key symbols could not be linked.\n"
          "  The LAPACK installation may be incomplete or broken.\n"
          "  If the deal.II build fails, try: cmake .. -DBUILD_OpenBLAS=ON\n")
    endif()
  endif()
endif()

# Python — required by Boost build
find_package(Python3 QUIET)
if(NOT Python3_FOUND)
  find_package(Python QUIET)
endif()
if(NOT Python3_FOUND AND NOT Python_FOUND)
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  Required dependency not found: Python\n"
      "======================================================================\n"
      "  Python is required by the Boost build system.\n"
      "\n"
      "  Install it using your package manager:\n"
      "    Ubuntu/Debian:  sudo apt install python3\n"
      "    Arch Linux:     sudo pacman -S python\n"
      "    Rocky/RHEL:     sudo dnf install python3\n"
      "    macOS:          brew install python3\n"
      "======================================================================\n"
  )
endif()

message("${BoldGreen}All pre-flight checks passed.${ColourReset}")

option(GPU "Enable/Disable GPU support" ON)

if(GPU)
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
else()
  set(HAVE_CUDA OFF)
  set(HAVE_HIP OFF)
endif()

if(GPU
   AND NOT HAVE_CUDA
   AND NOT HAVE_HIP)
  message(
    WARNING "\n" "  GPU=ON but neither CUDA nor HIP compiler was found.\n"
            "  The build will continue with CPU-only backends.\n"
            "  To silence this warning, configure with -DGPU=OFF\n")
endif()

set(HAVE_THREADS ON)

option(MPI "Enable MPI support" OFF)
if(MPI)
  find_package(MPI)
  if(MPI_FOUND)

  else()
    message(
      FATAL_ERROR
        "\n"
        "======================================================================\n"
        "  MPI enabled but no MPI library found.\n"
        "======================================================================\n"
        "  Install an MPI implementation:\n"
        "    Ubuntu/Debian:  sudo apt install libopenmpi-dev\n"
        "    Arch Linux:     sudo pacman -S openmpi\n"
        "    Rocky/RHEL:     sudo dnf install openmpi-devel\n"
        "    macOS:          brew install open-mpi\n"
        "\n"
        "  Or disable MPI: cmake .. -DMPI=OFF\n"
        "======================================================================\n"
    )
  endif()
endif()
message("")

message("Available parallel backends:")
# make the on and off values green/yellow
if(MPI)
  message("  ${BoldGreen}MPI: ${MPI}${ColourReset}")
else()
  message("  ${BoldYellow}MPI: ${MPI}${ColourReset}")
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

message("${BoldCyan}[1/8] Configuring HDF5...${ColourReset}")
ExternalProject_Add(
  hdf5_dep
  LOG_CONFIGURE ON
  LOG_OUTPUT_ON_FAILURE ON
  BUILD_COMMAND cmake -E echo "Skipping build step for hdf5."
  INSTALL_COMMAND cmake -E echo "Skipping install step for hdf5."
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/hdf5
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/hdf5
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=-march=native
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DMPI:BOOL=${MPI})

message(
  "${BoldCyan}[2/8] Configuring general dependencies (TBB, Eigen3, spdlog, ...)...${ColourReset}"
)
ExternalProject_Add(
  general_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/general
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/general
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS hdf5_dep
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=-march=native
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DBUILD_OpenBLAS:BOOL=${BUILD_OpenBLAS}
    -DCMAKE_PREFIX_PATH:PATH=${DEPEND_INSTALL_PREFIX}
    -DMPI:BOOL=${MPI})

if(BUILD_OpenBLAS)
  if(APPLE)
    set(BLAS_LIBRARY_PATH
        "${DEPEND_INSTALL_PREFIX}/lib/libopenblasDiFfRG.dylib")
  else()
    set(BLAS_LIBRARY_PATH "${DEPEND_INSTALL_PREFIX}/lib/libopenblasDiFfRG.so")
  endif()
endif()

message("${BoldCyan}[3/8] Configuring SUNDIALS...${ColourReset}")
ExternalProject_Add(
  sundials_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/sundials
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/sundials
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=-march=native
    -DCMAKE_C_FLAGS:STRING=-march=native
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DMPI:BOOL=${MPI})

message("${BoldCyan}[4/8] Configuring autodiff...${ColourReset}")
ExternalProject_Add(
  autodiff_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/autodiff
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/autodiff
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
                   -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX})

message("${BoldCyan}[5/8] Configuring Boost...${ColourReset}")
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
    "  ${BoldYellow}Kokkos_ARCH_LIST not set. GPU architecture must be detectable when building Kokkos.${ColourReset}"
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

message("${BoldCyan}[6/8] Configuring Kokkos...${ColourReset}")
ExternalProject_Add(
  kokkos_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 5.0.1
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/kokkos
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  # DEPENDS general_dep sundials_dep autodiff_dep
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
    DEPENDS general_dep # sundials_dep autodiff_dep kokkos_dep
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/petsc
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/petsc
    INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    BUILD_COMMAND cmake -E echo "Skipping build step for PETSc."
    INSTALL_COMMAND cmake -E echo "Skipping install step for PETSc."
    # Propagate DEPEND_INSTALL_PREFIX to the dependencies
    CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
      -DOpenBLAS_LIBRARY:STRING=${BLAS_LIBRARY_PATH} -DHAVE_MPI:BOOL=${MPI})
  set(is_petsc_dep petsc_dep)
endif()

# ##############################################################################
# deal.II
# ##############################################################################

if(BUILD_OpenBLAS)
  set(dealii_lapack "-DLAPACK_LIBRARIES:STRING=${BLAS_LIBRARY_PATH}")
endif()

# RAM-aware thread cap for deal.II (linking can OOM)
cmake_host_system_information(RESULT _total_ram_mb QUERY TOTAL_PHYSICAL_MEMORY)
math(EXPR _max_threads_by_ram "${_total_ram_mb} / 2000")
if(_max_threads_by_ram LESS 1)
  set(_max_threads_by_ram 1)
endif()
include(ProcessorCount)
ProcessorCount(_nproc)
if(_nproc LESS 1)
  set(_nproc 1)
endif()
math(EXPR _half_cores "${_nproc} / 2")
if(_half_cores LESS 1)
  set(_half_cores 1)
endif()
# Use the lesser of half cores, RAM-based limit, and 8
set(_dealii_jobs ${_half_cores})
if(_max_threads_by_ram LESS _dealii_jobs)
  set(_dealii_jobs ${_max_threads_by_ram})
endif()
if(_dealii_jobs GREATER 8)
  set(_dealii_jobs 8)
endif()
message(
  "  deal.II build parallelism: ${_dealii_jobs} threads (${_total_ram_mb} MB RAM, ${_nproc} cores)"
)

option(NODEBUG "Disable the debug build in deal.ii" OFF)

set(dealiiConfig "DebugRelease")
if(NODEBUG)
  set(dealiiConfig "Release")
endif()

message("${BoldCyan}[7/8] Configuring deal.II...${ColourReset}")
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
  BUILD_COMMAND cmake --build . -- -j${_dealii_jobs}
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_Fortran_COMPILER:STRING=${CMAKE_Fortran_COMPILER}
    -DCMAKE_BUILD_TYPE:STRING=${dealiiConfig}
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
    -DCMAKE_POLICY_DEFAULT_CMP0167:STRING=NEW
    -DCMAKE_WARN_DEPRECATED:BOOL=OFF
    -DDEAL_II_CXX_FLAGS:STRING=-march=native
    -DDEAL_II_COMPONENT_EXAMPLES:BOOL=OFF
    -DDEAL_II_COMPONENT_DOCUMENTATION:BOOL=OFF
    -DDEAL_II_WITH_MPI:BOOL=${MPI}
    -DDEAL_II_WITH_KOKKOS:BOOL=OFF
    -DDEAL_II_WITH_UMFPACK:BOOL=ON
    -DDEAL_II_WITH_TASKFLOW:BOOL=OFF
    ${dealii_lapack}
    -DDEAL_II_WITH_LAPACK:BOOL=ON
    -DDEAL_II_WITH_HDF5:BOOL=OFF
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

message("${BoldCyan}[8/8] Configuring DiFfRG...${ColourReset}")
ExternalProject_Add(
  DiFfRG
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/DiFfRG/
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  DEPENDS general_dep boost_dep deal.II_dep kokkos_dep autodiff_dep
  CMAKE_CACHE_ARGS
    -DMPI:BOOL=${MPI}
    -DHDF5:BOOL=ON
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
