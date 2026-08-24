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
It will configure and build all dependencies, including DiFfRG itself.
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

    # The whole stack assumes 32-bit BLAS/LAPACK indices: deal.II's types::blas_int
    # is a plain int and PETSc is configured --with-64-bit-indices=0. A cluster
    # module that provides an ILP64 BLAS (MKL's ilp64 layer, an OpenBLAS built with
    # INTERFACE64=1) links fine and then reads every dimension argument as garbage,
    # so establish the index size here instead of letting it through.
    #
    # Knowing the answer is also what keeps PETSc's own detection honest: its probe
    # has to *run* a test binary, and where it cannot -- batch nodes, restricted
    # login nodes -- it assumes 64-bit and then refuses ScaLAPACK, i.e. MUMPS, with
    #   Cannot use scalapack with 64-bit BLAS/LAPACK indices
    # We forward the verdict as --known-64-bit-blas-indices only when the probe
    # below actually produced one.
    set(_blas_probe_src ${CMAKE_CURRENT_BINARY_DIR}/blas_index_probe.c)
    file(
      WRITE ${_blas_probe_src}
      "extern double ddot_(const int *, const double *, const int *, const double *, const int *);\n"
      "int main(void)\n"
      "{\n"
      "  double x[4] = {3.0, 5.0, 7.0, 9.0};\n"
      "  int n = 2, one = 1;\n"
      "  /* 32-bit indices: 3*3 + 5*5 = 34. With 64-bit indices n and one are read as garbage. */\n"
      "  return ddot_(&n, x, &one, x, &one) == 34.0 ? 0 : 1;\n"
      "}\n")
    try_run(
      _blas_probe_run _blas_probe_compiled ${CMAKE_CURRENT_BINARY_DIR}/blas_index_probe
      ${_blas_probe_src} LINK_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

    if(_blas_probe_compiled AND _blas_probe_run STREQUAL "0")
      set(BLAS_INDEX_SIZE 32)
    elseif(_blas_probe_compiled AND _blas_probe_run STREQUAL "1")
      set(BLAS_INDEX_SIZE 64)
    else()
      # Did not compile, or did not run (cross-compilation, a crash inside the
      # BLAS). No verdict; PETSc falls back to its own detection.
      set(BLAS_INDEX_SIZE unknown)
    endif()

    if(BLAS_INDEX_SIZE STREQUAL "64")
      message(
        FATAL_ERROR
          "\n"
          "======================================================================\n"
          "  The BLAS/LAPACK found here uses 64-bit integer indices (ILP64)\n"
          "======================================================================\n"
          "  Found: ${LAPACK_LIBRARIES}\n"
          "\n"
          "  deal.II and PETSc are both built with 32-bit indices here, and an\n"
          "  ILP64 BLAS links against them without complaint while silently\n"
          "  misreading every matrix dimension. PETSc additionally refuses to\n"
          "  build ScaLAPACK/MUMPS against it.\n"
          "\n"
          "  Either point the build at an LP64 BLAS -- on a cluster that usually\n"
          "  means loading the module's lp64 variant rather than the ilp64 one --\n"
          "  or let DiFfRG build its own:\n"
          "    cmake .. -DBUILD_OpenBLAS=ON\n"
          "======================================================================\n")
    endif()

    if(BLAS_INDEX_SIZE STREQUAL "unknown")
      message(
        STATUS
          "Could not determine the BLAS/LAPACK index size; leaving the detection to PETSc.")
    endif()

    # Pin PETSc to the same BLAS/LAPACK deal.II gets. Left to itself PETSc searches
    # independently and can land on a different library entirely (MKL via MKLROOT,
    # say), which is how an ILP64 BLAS gets into the build on a machine whose
    # default BLAS is LP64. PETSc takes the list as --with-blaslapack-lib=[a,b,c],
    # so anything CMake could put in it that a comma-separated list cannot carry --
    # generator expressions, embedded separators -- means falling back to PETSc's
    # own search rather than handing it something malformed.
    set(_system_blaslapack ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    list(REMOVE_DUPLICATES _system_blaslapack)
    if(NOT "${_system_blaslapack}" MATCHES "[$,]")
      list(JOIN _system_blaslapack "," PETSC_BLASLAPACK_LIB)
    endif()
  endif()
else()
  # The bundled OpenBLAS is built without INTERFACE64, so its indices are 32-bit.
  set(BLAS_INDEX_SIZE 32)
endif()

message("${BoldGreen}All pre-flight checks passed.${ColourReset}")

# Optional ccache support for dependency builds
option(USE_CCACHE "Use ccache for dependency builds if available" ON)
if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message("${BoldGreen}Found ccache: ${CCACHE_PROGRAM}${ColourReset}")
    set(CCACHE_CMAKE_ARGS
        -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CCACHE_PROGRAM}
        -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CCACHE_PROGRAM})
  endif()
endif()

# ##############################################################################
# Native architecture optimization
# ##############################################################################
# Whether to optimize for the CPU of the build machine (-march=native). ON gives
# the fastest binaries but they only run on CPUs with the same instruction set;
# turn OFF for portable builds (e.g. CI images that must run on other hardware).
option(NATIVE "Optimize for the build machine's CPU (-march=native). Disable for portable binaries." ON)
diffrg_report_native(${NATIVE} "${CMAKE_CXX_COMPILER}")

# Per-language flag fragments forwarded to the dependency ExternalProjects. Empty
# when NATIVE=OFF, so the corresponding CMAKE_CACHE_ARGS entry simply vanishes.
if(NATIVE)
  set(_native_cxx_arg "-DCMAKE_CXX_FLAGS:STRING=-march=native")
  set(_native_c_arg "-DCMAKE_C_FLAGS:STRING=-march=native")
  set(_native_dealii_arg "-DDEAL_II_CXX_FLAGS:STRING=-march=native")
  set(_native_petsc_flag "-march=native")
else()
  set(_native_cxx_arg "")
  set(_native_c_arg "")
  set(_native_dealii_arg "")
  set(_native_petsc_flag "")
endif()

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

# DiFfRG's CPU parallelism is TBB: deal.II's MeshWorker drives a tbb::parallel_pipeline, and every
# production flow instantiates its integrators with TBB_exec. A Kokkos std::threads host backend
# would be a *second* pool of spinning workers on the same cores -- and not an idle one, since
# deal.II routes its own vector kernels through Kokkos::DefaultHostExecutionSpace
# (lac/vector.templates.h, lac/la_parallel_vector.templates.h). Off by default so the host space is
# Kokkos::Serial and there is exactly one CPU thread pool.
option(KOKKOS_THREADS "Build Kokkos' std::threads host backend (contends with TBB; see comment)" OFF)
set(HAVE_THREADS ${KOKKOS_THREADS})

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
# CPU work always runs on TBB; this line says which Kokkos host backend sits beside it.
if(HAVE_THREADS)
  message("  ${BoldYellow}CPU: TBB + Kokkos::Threads (two pools -- they contend)${ColourReset}")
else()
  message("  ${BoldGreen}CPU: TBB (Kokkos host backend is Serial)${ColourReset}")
endif()
message("")

# ##############################################################################
# Dependencies
# ##############################################################################

include(ExternalProject)

# ##############################################################################
# Dependency selection: system vs bundled
#
# Boost, TBB, HDF5 and SUNDIALS can each be taken from the system instead of
# being rebuilt. For every one of them: <LIB>_DIR set     -> use the install at
# that prefix (FATAL if unusable) BUILD_<LIB>=ON    -> always build the bundled,
# pinned copy otherwise (auto)  -> use a viable system install if found, else
# build it The blocks below resolve, for each library, whether the bundled
# ExternalProject is needed (_build_<lib>) and which location to hand to deal.II
# / the DiFfRG library build.
#
# Note for Boost specifically: the bundled build unpacks and compiles the source
# archive tracked at dependencies/boost/boost_1_81_0-diffrg-slim.tar.xz and never
# downloads anything, so the fallback path stays usable on offline / cluster
# machines.
# ##############################################################################

set(BOOST_DIR
    ""
    CACHE PATH "Use the Boost install at this prefix instead of building one")
option(BUILD_BOOST
       "Always build the bundled, pinned Boost (ignore any system Boost)" OFF)
set(TBB_DIR
    ""
    CACHE PATH "Use the TBB install at this prefix instead of building one")
option(BUILD_TBB "Always build the bundled oneTBB (ignore any system TBB)" OFF)
set(HDF5_DIR
    ""
    CACHE PATH "Use the HDF5 install at this prefix instead of building one")
option(BUILD_HDF5 "Always build the bundled HDF5 (ignore any system HDF5)" OFF)
set(SUNDIALS_DIR
    ""
    CACHE PATH
          "Use the SUNDIALS install at this prefix instead of building one")
option(BUILD_SUNDIALS
       "Always build the bundled SUNDIALS (ignore any system SUNDIALS)" OFF)

set(DiFfRG_BOOST_MIN_VERSION 1.81)
set(DiFfRG_TBB_MIN_VERSION 2021)
set(DiFfRG_HDF5_MIN_VERSION 1.12.0)
set(DiFfRG_SUNDIALS_MIN_VERSION 5.4.0) # deal.II's own SUNDIALS floor
set(DiFfRG_SUNDIALS_KNOWN_GOOD_MAX 7.7) # newest version validated with bundled
                                        # deal.II

# TBB_DIR/HDF5_DIR double as our "use this prefix" knobs and as CMake's own
# config-mode result cache entries: a failed find_package(<Pkg> CONFIG)
# overwrites them with "<Pkg>_DIR-NOTFOUND". On the next configure of the same
# build directory that leftover would look like a user-supplied prefix and turn
# auto-detection into a hard error, so strip it before reading the knob.
macro(_diffrg_user_prefix out var)
  set(${out} "${${var}}")
  if("${${out}}" MATCHES "-NOTFOUND$")
    set(${out} "")
  endif()
endmacro()

# ----------------------------------------------------------------------------
# Boost — needed by both deal.II and the DiFfRG library. Use Boost's own
# BoostConfig.cmake (config mode); the legacy FindBoost module is removed in
# CMake >= 3.30 and deal.II 9.7 consumes a modern config-mode Boost fine.
#
# When the bundled build is selected, it unpacks and compiles the tracked source
# archive (dependencies/boost/boost_1_81_0-diffrg-slim.tar.xz) — no network
# access is involved.
# ----------------------------------------------------------------------------
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
set(_boost_components thread iostreams serialization system)
set(_build_boost OFF)
set(_boost_user_dir "${BOOST_DIR}")
if(NOT "${_boost_user_dir}" STREQUAL "")
  set(BOOST_ROOT "${_boost_user_dir}")
  # BOOST_ROOT is only a *hint*: in config mode find_package would still fall
  # back to CMAKE_SYSTEM_PREFIX_PATH, so an unusable BOOST_DIR would silently
  # resolve to some other system Boost instead of failing. Restrict the search
  # to the requested prefix so BOOST_DIR is authoritative, as documented.
  find_package(Boost ${DiFfRG_BOOST_MIN_VERSION} QUIET
               COMPONENTS ${_boost_components}
               PATHS "${_boost_user_dir}" NO_DEFAULT_PATH)
  if(NOT Boost_FOUND)
    message(
      FATAL_ERROR
        "BOOST_DIR=${_boost_user_dir} does not provide a usable Boost >= ${DiFfRG_BOOST_MIN_VERSION}.\n"
        "  Point it at a prefix containing BoostConfig.cmake, or use -DBUILD_BOOST=ON."
    )
  endif()
  set(_boost_status
      "  ${BoldGreen}[Boost] Using Boost ${Boost_VERSION} from BOOST_DIR=${_boost_user_dir}.${ColourReset}"
  )
elseif(BUILD_BOOST)
  set(_build_boost ON)
else()
  find_package(Boost ${DiFfRG_BOOST_MIN_VERSION} QUIET
               COMPONENTS ${_boost_components})
  if(NOT Boost_FOUND)
    set(_build_boost ON)
  endif()
endif()

if(_build_boost)
  # The bundled build needs the tracked Boost source archive. Its contents are
  # hash-checked in dependencies/boost/CMakeLists.txt when boost_dep configures;
  # here we only fail fast, at superbuild configure time, if it is absent.
  set(_boost_archive
      "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/boost/boost_1_81_0-diffrg-slim.tar.xz"
  )
  if(NOT EXISTS "${_boost_archive}")
    message(
      FATAL_ERROR
        "No usable system Boost >= ${DiFfRG_BOOST_MIN_VERSION} was found and the "
        "bundled Boost source archive is missing. Expected ${_boost_archive}.\n"
        "  Install Boost and point -DBOOST_DIR=<prefix> at it, or restore the "
        "archive with\n"
        "    git checkout -- dependencies/boost/boost_1_81_0-diffrg-slim.tar.xz")
  endif()
  set(_boost_root "${DEPEND_INSTALL_PREFIX}")
  set(_boost_incdir "${DEPEND_INSTALL_PREFIX}/include")
  set(_boost_libdir "${DEPEND_INSTALL_PREFIX}/lib")
  set(_boost_extra_args "")
  set(_boost_dep boost_dep)
else()
  # Config mode populates Boost_INCLUDE_DIRS and Boost_DIR but not the legacy
  # Boost_LIBRARY_DIRS, so derive the install prefix and library dir ourselves.
  list(GET Boost_INCLUDE_DIRS 0 _boost_incdir)
  get_filename_component(_boost_root "${_boost_incdir}" DIRECTORY)
  if(DEFINED Boost_LIBRARY_DIRS AND NOT "${Boost_LIBRARY_DIRS}" STREQUAL "")
    list(GET Boost_LIBRARY_DIRS 0 _boost_libdir)
  elseif(EXISTS "${_boost_root}/lib")
    set(_boost_libdir "${_boost_root}/lib")
  else()
    set(_boost_libdir "${_boost_root}/lib64")
  endif()
  set(_boost_extra_args -DBoost_DIR:PATH=${Boost_DIR})
  set(_boost_dep "")
  # Do not clobber the more specific message set by the BOOST_DIR branch above.
  if(NOT DEFINED _boost_status)
    set(_boost_status
        "  ${BoldGreen}[Boost] Using system Boost ${Boost_VERSION} (${_boost_incdir}); skipping bundled build.${ColourReset}"
    )
  endif()
endif()

# ----------------------------------------------------------------------------
# TBB — needed by both deal.II (path-based) and the DiFfRG library
# (find_package(TBB) -> TBB::tbb). The bundled oneTBB is built inside
# general_dep (see GENERAL_BUILD_TBB), not its own ExternalProject.
# ----------------------------------------------------------------------------
set(_build_tbb OFF)
# find_package(TBB) overwrites TBB_DIR with the discovered config dir (or with
# TBB_DIR-NOTFOUND); capture the user-given prefix first.
_diffrg_user_prefix(_tbb_user_dir TBB_DIR)
if(NOT "${_tbb_user_dir}" STREQUAL "")
  find_package(
    TBB
    ${DiFfRG_TBB_MIN_VERSION}
    QUIET
    CONFIG
    PATHS
    "${_tbb_user_dir}"
    PATH_SUFFIXES
    lib/cmake/TBB
    lib64/cmake/TBB
    cmake/TBB
    NO_DEFAULT_PATH)
  if(NOT TBB_FOUND)
    message(
      FATAL_ERROR
        "TBB_DIR=${_tbb_user_dir} does not provide TBB >= ${DiFfRG_TBB_MIN_VERSION}.\n"
        "  Point it at a prefix containing TBBConfig.cmake, or use -DBUILD_TBB=ON."
    )
  endif()
  set(_tbb_status
      "  ${BoldGreen}[TBB] Using TBB ${TBB_VERSION} from TBB_DIR=${_tbb_user_dir}.${ColourReset}"
  )
elseif(BUILD_TBB)
  set(_build_tbb ON)
else()
  find_package(TBB ${DiFfRG_TBB_MIN_VERSION} QUIET CONFIG)
  if(NOT TBB_FOUND)
    set(_build_tbb ON)
  endif()
endif()

if(_build_tbb)
  set(_tbb_dir "${DEPEND_INSTALL_PREFIX}")
else()
  set(_tbb_dir "${TBB_DIR}") # the discovered TBBConfig.cmake directory
  set(_tbb_status
      "  ${BoldGreen}[TBB] Using system TBB ${TBB_VERSION} (${TBB_DIR}); skipping bundled build.${ColourReset}"
  )
endif()

# ----------------------------------------------------------------------------
# HDF5 — needed only by the DiFfRG library (deal.II is built with
# DEAL_II_WITH_HDF5=OFF). DiFfRG only uses the HDF5 C API, so 1.12 is the floor.
# ----------------------------------------------------------------------------
set(_build_hdf5 OFF)
_diffrg_user_prefix(_hdf5_user_dir HDF5_DIR)
# Do not pass the version to find_package: HDF5's config-version file uses a
# same-major-version compatibility policy, so requesting 1.12 would reject a
# newer 2.x install. Gate the version manually instead. Many distros (Fedora,
# Debian, Ubuntu, ...) ship HDF5 without a CMake config package (only headers +
# libs + the FindHDF5 module / pkg-config). So try config mode first (gives
# imported targets; matches the bundled static build and Arch), then fall back
# to module mode (FindHDF5) for those config-less installs.
macro(_diffrg_find_hdf5)
  find_package(HDF5 QUIET CONFIG COMPONENTS C ${ARGN})
  if(NOT HDF5_FOUND OR HDF5_VERSION VERSION_LESS ${DiFfRG_HDF5_MIN_VERSION})
    find_package(HDF5 QUIET MODULE COMPONENTS C)
  endif()
endmacro()

if(NOT "${_hdf5_user_dir}" STREQUAL "")
  set(HDF5_ROOT "${_hdf5_user_dir}")
  _diffrg_find_hdf5(
    PATHS
    "${_hdf5_user_dir}"
    PATH_SUFFIXES
    lib/cmake/hdf5
    lib64/cmake/hdf5
    cmake/hdf5
    share/cmake/hdf5)
  if(NOT HDF5_FOUND OR HDF5_VERSION VERSION_LESS ${DiFfRG_HDF5_MIN_VERSION})
    message(
      FATAL_ERROR
        "HDF5_DIR=${_hdf5_user_dir} does not provide a usable HDF5 >= ${DiFfRG_HDF5_MIN_VERSION} "
        "(found '${HDF5_VERSION}').\n"
        "  Point it at an HDF5 install prefix, or use -DBUILD_HDF5=ON.")
  endif()
  set(_hdf5_status
      "  ${BoldGreen}[HDF5] Using HDF5 ${HDF5_VERSION} from HDF5_DIR=${_hdf5_user_dir}.${ColourReset}"
  )
elseif(BUILD_HDF5)
  set(_build_hdf5 ON)
else()
  _diffrg_find_hdf5()
  if(NOT HDF5_FOUND OR HDF5_VERSION VERSION_LESS ${DiFfRG_HDF5_MIN_VERSION})
    set(_build_hdf5 ON)
  endif()
endif()

if(_build_hdf5)
  set(_hdf5_extra_args "")
  set(_hdf5_dep hdf5_dep)
else()
  # Pass the config-package dir on if config mode found it; otherwise (module
  # mode) pass the install prefix so the library build resolves the same HDF5.
  if(HDF5_DIR AND NOT "${HDF5_DIR}" STREQUAL "HDF5_DIR-NOTFOUND")
    set(_hdf5_extra_args -DHDF5_DIR:PATH=${HDF5_DIR})
    set(_hdf5_loc "${HDF5_DIR}")
  else()
    list(GET HDF5_INCLUDE_DIRS 0 _hdf5_inc0)
    get_filename_component(_hdf5_loc "${_hdf5_inc0}" DIRECTORY)
    set(_hdf5_extra_args -DHDF5_ROOT:PATH=${_hdf5_loc})
  endif()
  set(_hdf5_dep "")
  set(_hdf5_status
      "  ${BoldGreen}[HDF5] Using system HDF5 ${HDF5_VERSION} (${_hdf5_loc}); skipping bundled build.${ColourReset}"
  )
endif()

# ----------------------------------------------------------------------------
# SUNDIALS — needed only by deal.II (DiFfRG links it transitively via
# dealii::dealii). Replicate deal.II's own discovery: the
# arkode/kinsol/nvecserial libraries plus version >= 5.4.0; otherwise build the
# bundled copy.
# ----------------------------------------------------------------------------
set(_build_sundials OFF)
set(_sundials_prefix "${DEPEND_INSTALL_PREFIX}")
set(_sundials_dep "")
set(_sun_hint "")
set(_sun_explicit OFF)
set(_sun_mpi_mismatch OFF)
if(NOT "${SUNDIALS_DIR}" STREQUAL "")
  set(_sun_hint "${SUNDIALS_DIR}")
  set(_sun_explicit ON)
elseif(BUILD_SUNDIALS)
  set(_build_sundials ON)
endif()

if(NOT _build_sundials)
  find_library(
    _SUN_ARKODE
    NAMES sundials_arkode
    HINTS ${_sun_hint}
    PATH_SUFFIXES lib lib64)
  find_library(
    _SUN_KINSOL
    NAMES sundials_kinsol
    HINTS ${_sun_hint}
    PATH_SUFFIXES lib lib64)
  find_library(
    _SUN_NVEC
    NAMES sundials_nvecserial
    HINTS ${_sun_hint}
    PATH_SUFFIXES lib lib64)
  # The version macros live in sundials_config.h on most installs
  # (sundials_version.h only declares the runtime accessors and #includes the
  # config -- e.g. on Fedora it carries no macros), so locate the config header
  # and read whichever of the two actually defines SUNDIALS_VERSION_MAJOR.
  find_path(
    _SUN_INC sundials/sundials_config.h
    HINTS ${_sun_hint}
    PATH_SUFFIXES include)
  set(_sun_ok FALSE)
  set(_sun_ver "")
  set(_sun_reject_reason "")
  if(_SUN_ARKODE
     AND _SUN_KINSOL
     AND _SUN_NVEC
     AND _SUN_INC)
    set(_sun_verhdr "")
    foreach(_h sundials/sundials_config.h sundials/sundials_version.h)
      if(EXISTS "${_SUN_INC}/${_h}")
        file(STRINGS "${_SUN_INC}/${_h}" _sun_probe
             REGEX "#define[ \t]+SUNDIALS_VERSION_MAJOR")
        if(_sun_probe)
          set(_sun_verhdr "${_SUN_INC}/${_h}")
          break()
        endif()
      endif()
    endforeach()
    if(_sun_verhdr)
      file(STRINGS "${_sun_verhdr}" _sun_mj
           REGEX "#define[ \t]+SUNDIALS_VERSION_MAJOR")
      file(STRINGS "${_sun_verhdr}" _sun_mn
           REGEX "#define[ \t]+SUNDIALS_VERSION_MINOR")
      file(STRINGS "${_sun_verhdr}" _sun_pt
           REGEX "#define[ \t]+SUNDIALS_VERSION_PATCH")
      string(REGEX REPLACE "^.*MAJOR[ \t]+([0-9]+).*" "\\1" _sun_mj
                           "${_sun_mj}")
      string(REGEX REPLACE "^.*MINOR[ \t]+([0-9]+).*" "\\1" _sun_mn
                           "${_sun_mn}")
      string(REGEX REPLACE "^.*PATCH[ \t]+([0-9]+).*" "\\1" _sun_pt
                           "${_sun_pt}")
      set(_sun_ver "${_sun_mj}.${_sun_mn}.${_sun_pt}")
      # deal.II requires deal.II and SUNDIALS to agree on MPI support; otherwise
      # it silently sets DEAL_II_WITH_SUNDIALS=OFF, which leaves the
      # dealii::SUNDIALS namespace undefined and breaks our KINSOL/IDA wrappers
      # at compile time. Probe the system SUNDIALS MPI setting
      # (sundials_config.h) and require it to match the MPI option this build
      # passes to deal.II.
      set(_sun_mpi OFF)
      if(EXISTS "${_SUN_INC}/sundials/sundials_config.h")
        file(STRINGS "${_SUN_INC}/sundials/sundials_config.h" _sun_mpi_probe
             REGEX "#define[ \t]+SUNDIALS_MPI_ENABLED[ \t]+1")
        if(_sun_mpi_probe)
          set(_sun_mpi ON)
        endif()
      endif()
      if(MPI)
        set(_sun_want_mpi ON)
      else()
        set(_sun_want_mpi OFF)
      endif()
      if(_sun_ver MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$"
         AND NOT _sun_ver VERSION_LESS ${DiFfRG_SUNDIALS_MIN_VERSION})
        if(_sun_mpi STREQUAL _sun_want_mpi)
          set(_sun_ok TRUE)
        else()
          set(_sun_mpi_mismatch ON)
        endif()
      endif()
    endif()
  endif()
  if(_sun_ok)
    if(_sun_explicit)
      set(_sundials_prefix "${SUNDIALS_DIR}")
    else()
      get_filename_component(_sundials_prefix "${_SUN_INC}" DIRECTORY)
    endif()
    if(_sun_ver VERSION_GREATER ${DiFfRG_SUNDIALS_KNOWN_GOOD_MAX})
      message(
        WARNING
          "Using system SUNDIALS ${_sun_ver}, newer than the last version "
          "(${DiFfRG_SUNDIALS_KNOWN_GOOD_MAX}) validated with the bundled deal.II. "
          "If the deal.II build fails, configure with -DBUILD_SUNDIALS=ON.")
    endif()
    set(_sundials_status
        "  ${BoldGreen}[SUNDIALS] Using system SUNDIALS ${_sun_ver} (${_SUN_INC}); skipping bundled build.${ColourReset}"
    )
  elseif(_sun_mpi_mismatch)
    # Found a usable SUNDIALS, but its MPI support does not match MPI=${MPI}.
    if(_sun_explicit)
      message(
        FATAL_ERROR
          "SUNDIALS_DIR=${SUNDIALS_DIR} provides SUNDIALS ${_sun_ver} built with "
          "SUNDIALS_MPI_ENABLED=${_sun_mpi}, but DiFfRG is configured with MPI=${MPI}. "
          "deal.II requires deal.II and SUNDIALS to agree on MPI support.\n"
          "  Use -DBUILD_SUNDIALS=ON to build a matching bundled SUNDIALS, or point "
          "SUNDIALS_DIR at a SUNDIALS whose MPI setting matches MPI=${MPI}.")
    else()
      set(_sundials_status
          "  ${BoldYellow}[SUNDIALS] System SUNDIALS ${_sun_ver} (${_SUN_INC}) MPI setting (SUNDIALS_MPI_ENABLED=${_sun_mpi}) does not match MPI=${MPI}; building the bundled SUNDIALS instead. deal.II requires both to agree on MPI.${ColourReset}"
      )
      set(_build_sundials ON)
    endif()
  elseif(_sun_explicit)
    if(NOT _sun_reject_reason)
      set(_sun_reject_reason
          "the required libraries (arkode, kinsol, nvecserial) or version header were not found"
      )
    endif()
    message(
      FATAL_ERROR
        "SUNDIALS_DIR=${SUNDIALS_DIR} is not usable: ${_sun_reject_reason}.\n"
        "  Use -DBUILD_SUNDIALS=ON to build the bundled SUNDIALS instead.")
  else()
    if(_sun_reject_reason)
      set(_sundials_status
          "  ${BoldYellow}[SUNDIALS] Ignoring system SUNDIALS (${_sun_reject_reason}); building the bundled copy.${ColourReset}"
      )
    endif()
    set(_build_sundials ON)
  endif()
endif()

if(_build_sundials)
  set(_sundials_prefix "${DEPEND_INSTALL_PREFIX}")
  set(_sundials_dep sundials_dep)
endif()

# deal.II's FindSUNDIALS, when handed an explicit SUNDIALS_DIR, searches only
# <DIR>/lib and <DIR>/lib64 -- it misses Debian/Ubuntu multiarch paths such as
# /usr/lib/x86_64-linux-gnu, so it finds the headers but not the libraries and
# then *silently* disables SUNDIALS (it is auto/optional). DiFfRG, however,
# requires it (SUNDIALS::KINSOL/IDA), so that failure only surfaces later when
# kinsol.cc fails to compile. Therefore: only hand deal.II the hint for a
# bundled build or an explicit -DSUNDIALS_DIR (both have a plain lib/); for an
# auto-detected system install, omit the hint and let deal.II's standard,
# multiarch-aware search locate the libraries.
if(_build_sundials OR _sun_explicit)
  set(_dealii_sundials_arg "-DSUNDIALS_DIR:STRING=${_sundials_prefix}")
else()
  set(_dealii_sundials_arg "")
endif()

message("${BoldCyan}[1/8] Configuring HDF5...${ColourReset}")
if(_build_hdf5)
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
      ${_native_cxx_arg}
      -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
      ${CCACHE_CMAKE_ARGS}
      -DMPI:BOOL=${MPI})
else()
  message(${_hdf5_status})
endif()

message(
  "${BoldCyan}[2/8] Configuring general dependencies (TBB, Eigen3, spdlog, ...)...${ColourReset}"
)
if(NOT _build_tbb)
  message(${_tbb_status})
endif()
ExternalProject_Add(
  general_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/general
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/general
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS ${_hdf5_dep}
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    ${_native_cxx_arg}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    -DCPM_SOURCE_CACHE:PATH=${CMAKE_CURRENT_BINARY_DIR}/cpm-cache
    ${CCACHE_CMAKE_ARGS}
    -DBUILD_OpenBLAS:BOOL=${BUILD_OpenBLAS}
    -DGENERAL_BUILD_TBB:BOOL=${_build_tbb}
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
if(_sundials_status)
  message(${_sundials_status})
endif()
if(_build_sundials)
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
      ${_native_cxx_arg}
      ${_native_c_arg}
      -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
      ${CCACHE_CMAKE_ARGS}
      -DMPI:BOOL=${MPI})
endif()

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
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX} ${CCACHE_CMAKE_ARGS})

message("${BoldCyan}[5/8] Configuring Boost...${ColourReset}")
# Boost selection was resolved above; only register the bundled build if needed.
# The bundled build unpacks and compiles the source archive in dependencies/boost.
if(_build_boost)
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
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
                     -DCCACHE_PROGRAM:STRING=${CCACHE_PROGRAM})
else()
  message(${_boost_status})
endif()

# ##############################################################################
# Kokkos
# ##############################################################################

message("${BoldCyan}[6/8] Configuring Kokkos...${ColourReset}")

# Kokkos_ARCH_NATIVE is Kokkos' *host* -march=native switch (cmake/kokkos_arch.cmake:
# it adds -march=native -mtune=native to the host compiler), so it must follow the
# NATIVE option -- otherwise the bundled Kokkos objects bake in the build machine's
# ISA (e.g. AVX-512) and crash on lesser hardware. GPU archs come from Kokkos_ARCH_LIST.
if(NATIVE AND NOT DEFINED Kokkos_ARCH)
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
    "  ${BoldYellow}[Kokkos] Kokkos_ARCH_LIST not set. GPU architecture must be detectable when building Kokkos.${ColourReset}"
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

ExternalProject_Add(
  kokkos_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 5.1.1
  GIT_SHALLOW ON
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/kokkos
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  # DEPENDS general_dep sundials_dep autodiff_dep
  CMAKE_ARGS -DCMAKE_CXX_FLAGS=-fPIC
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD:STRING=20
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    # Build Kokkos as a SHARED library. As a static archive it gets linked into BOTH the
    # consuming executable AND libdeal_II.so, so Kokkos' global state (cuda_devices, the
    # per-device constantMem* caches, ...) is constructed and destroyed TWICE at process exit
    # -> "corrupted size vs. prev_size" / "double free" in __cxa_finalize. A single shared
    # libkokkoscore.so gives one copy of that state across all consumers.
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DKokkos_ARCH_NATIVE:BOOL=${Kokkos_ARCH_NATIVE}
    ${Kokkos_ARCH_LIST}
    -DCMAKE_CUDA_COMPILER:STRING=${CMAKE_CUDA_COMPILER}
    -DKokkos_ENABLE_CUDA:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_CUDA_CONSTEXPR:BOOL=${HAVE_CUDA}
    -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE:BOOL=OFF
    # deal.II 9.7.1 uses the deprecated Kokkos::View<>::HostMirror alias, which
    # Kokkos 5.x gates behind DEPRECATED_CODE_4. Re-enable it so the bundled
    # deal.II compiles against Kokkos 5.1.1.
    -DKokkos_ENABLE_DEPRECATED_CODE_4:BOOL=ON
    -DKokkos_ENABLE_HIP:BOOL=${HAVE_HIP}
    -DKokkos_ENABLE_OPENMP:BOOL=OFF
    -DKokkos_ENABLE_THREADS:BOOL=${HAVE_THREADS}
    -DKokkos_ENABLE_SERIAL:BOOL=ON
    -DKokkos_ENABLE_TESTS:BOOL=OFF)

# Machine-capability probe, shared by every heavy dependency build (deal.II and
# PETSc). Linking deal.II can OOM and PETSc's own make is unbounded, so both are
# capped against RAM as well as core count -- never a bare -j/-j$(nproc).
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

# ##############################################################################
# PETSc
# ##############################################################################

# PETSc supplies the distributed linear algebra (PETScWrappers::MPI::{Vector,
# SparseMatrix}) and, via deal.II, the only MPI-capable SUNDIALS::IDA
# instantiation that does not drag in Trilinos: deal.II instantiates IDA for
# Vector, BlockVector, TrilinosWrappers::MPI::* and PETScWrappers::MPI::* only --
# notably *not* for LinearAlgebra::distributed::Vector (that one gets KINSOL but
# not IDA). PETSc is therefore required for a distributed FE/FV run, and useless
# without MPI, hence the default.
option(BUILD_PETSC "Build PETSc (distributed linear algebra and solvers)" ${MPI})

# Solver capability. Both cost build time -- MUMPS a great deal of it -- so each is
# its own switch. deal.II detects what PETSc was actually built with and exports
# DEAL_II_PETSC_WITH_HYPRE / DEAL_II_PETSC_WITH_MUMPS (see its
# cmake/configure/configure_20_petsc.cmake); DiFfRG's solver wrappers gate on
# those macros rather than on these options, so a mismatch degrades rather than
# miscompiles.
option(PETSC_GMRES
       "PETSc with hypre -> PreconditionBoomerAMG for Krylov solves (no Fortran required)"
       ON)
# MUMPS defaults to ON under MPI so that a distributed run keeps the *direct* solve a serial
# run has: DefaultLinearSolver picks PETScDirect where deal.II reports MUMPS, and falls back to
# the iterative PETScKrylov where it does not. That fallback changes the character of the solve,
# which a fine-tuned flow can notice, so the default is worth the build time -- and MUMPS is the
# bulk of it (it pulls the ScaLAPACK chain and needs Fortran). Turning it off is supported and
# only costs the direct solver.
#
# It is a hard error only when asked for explicitly. Defaulted on and unbuildable, it degrades to
# PETSC_GMRES with a warning, so a machine without MPI Fortran can still configure.
if(NOT DEFINED PETSC_MUMPS_WAS_DEFAULTED)
  if(DEFINED PETSC_MUMPS)
    set(PETSC_MUMPS_WAS_DEFAULTED FALSE CACHE INTERNAL "PETSC_MUMPS was set explicitly")
  else()
    set(PETSC_MUMPS_WAS_DEFAULTED TRUE CACHE INTERNAL "PETSC_MUMPS came from the MPI default")
  endif()
endif()
option(PETSC_MUMPS
       "PETSc with MUMPS -> SparseDirectMUMPS distributed direct solves (requires Fortran; long build)"
       ${MPI})

if(BUILD_PETSC)
  if(NOT MPI)
    message(
      WARNING
        "BUILD_PETSC=ON with MPI=OFF: PETSc will be built serial, which buys nothing "
        "over the existing UMFPACK path. Set -DMPI=ON for distributed linear algebra.")
  endif()

  # MUMPS' PETSc recipe raises a hard error without an F90 compiler, so resolve one up front
  # rather than failing 20 minutes into the build. Both probes live here, in the superbuild:
  # dependencies/petsc/CMakeLists.txt needs an MPI Fortran *wrapper* as well, and it runs inside
  # the ExternalProject configure step with LOG_CONFIGURE ON, so a failure there lands in a log
  # file rather than on the terminal.
  set(_petsc_fc "${CMAKE_Fortran_COMPILER}")
  if(NOT _petsc_fc)
    find_program(_petsc_fc NAMES gfortran flang f95)
  endif()

  set(_mumps_viable TRUE)
  set(_mumps_why "")
  if(NOT _petsc_fc)
    set(_mumps_viable FALSE)
    set(_mumps_why "no Fortran compiler (gfortran/flang/f95) was found")
  endif()
  if(MPI)
    find_program(_petsc_mpifc NAMES mpifort mpif90)
    if(NOT _petsc_mpifc)
      set(_mumps_viable FALSE)
      set(_mumps_why "no MPI Fortran wrapper (mpifort/mpif90) was found on PATH")
    endif()
  endif()

  if(PETSC_MUMPS AND NOT _mumps_viable)
    if(PETSC_MUMPS_WAS_DEFAULTED)
      message(
        WARNING
          "MUMPS was enabled by default because MPI=ON, but ${_mumps_why}. Falling back to the "
          "hypre/BoomerAMG Krylov path. Distributed runs will use PETScKrylov instead of "
          "PETScDirect, which is iterative rather than direct. Install your MPI's Fortran "
          "support and reconfigure in a fresh build directory to get MUMPS, or pass "
          "-DPETSC_MUMPS=ON explicitly to make this a hard error instead.")
      set(PETSC_MUMPS OFF CACHE BOOL "" FORCE)
      set(PETSC_GMRES ON CACHE BOOL "" FORCE)
    else()
      message(
        FATAL_ERROR
          "PETSC_MUMPS=ON was requested explicitly, but ${_mumps_why}. Install gfortran/flang "
          "and your MPI's Fortran wrapper, or use -DPETSC_MUMPS=OFF and rely on PETSC_GMRES "
          "(hypre/BoomerAMG), which needs no Fortran.")
    endif()
  endif()

  # Bound PETSc's own make the same way deal.II's build is bounded: never a bare -j.
  set(PETSC_MAX_JOBS
      8
      CACHE STRING "Maximum parallel jobs for the PETSc build")
  set(_petsc_jobs ${_half_cores})
  if(_max_threads_by_ram LESS _petsc_jobs)
    set(_petsc_jobs ${_max_threads_by_ram})
  endif()
  if(_petsc_jobs GREATER ${PETSC_MAX_JOBS})
    set(_petsc_jobs ${PETSC_MAX_JOBS})
  endif()
  message(
    "  ${BoldGreen}[PETSc] build parallelism: ${_petsc_jobs} jobs (hypre: ${PETSC_GMRES}, MUMPS: ${PETSC_MUMPS}).${ColourReset}"
  )

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
      -DNATIVE:BOOL=${NATIVE}
      -DOpenBLAS_LIBRARY:STRING=${BLAS_LIBRARY_PATH}
      -DPETSC_BLASLAPACK_LIB:STRING=${PETSC_BLASLAPACK_LIB}
      -DBLAS_INDEX_SIZE:STRING=${BLAS_INDEX_SIZE}
      -DHAVE_MPI:BOOL=${MPI}
      -DPETSC_GMRES:BOOL=${PETSC_GMRES}
      -DPETSC_MUMPS:BOOL=${PETSC_MUMPS}
      -DPETSC_JOBS:STRING=${_petsc_jobs}
      -DMPI_C_COMPILER:STRING=${MPI_C_COMPILER}
      -DMPI_CXX_COMPILER:STRING=${MPI_CXX_COMPILER}
      -DPETSC_Fortran_COMPILER:STRING=${_petsc_fc}
      # PETSc must be built with the *same* compilers as deal.II and DiFfRG. The
      # MPI wrappers alone are not enough: mpicc/mpicxx wrap whichever compiler is
      # the system default, which need not be the one this superbuild selected.
      -DPETSC_C_COMPILER:STRING=${CMAKE_C_COMPILER}
      -DPETSC_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER})
  set(is_petsc_dep petsc_dep)
endif()

# ##############################################################################
# deal.II
# ##############################################################################

message("${BoldCyan}[7/8] Configuring deal.II...${ColourReset}")

if(BUILD_OpenBLAS)
  set(dealii_lapack "-DLAPACK_LIBRARIES:STRING=${BLAS_LIBRARY_PATH}")
endif()

# Only hand deal.II the PETSc hints when we actually built PETSc. PETSC_ARCH is
# deliberately empty: `configure --prefix=` installs a flat tree with no arch
# subdirectory, and deal.II's FindPETSC joins PETSC_DIR/PETSC_ARCH when the latter
# is non-empty.
if(BUILD_PETSC)
  set(_dealii_petsc_args "-DPETSC_DIR:PATH=${DEPEND_INSTALL_PREFIX}"
                         "-DPETSC_ARCH:STRING=")
else()
  set(_dealii_petsc_args "")
endif()

# Use the lesser of half cores, RAM-based limit, and hard cap
set(DEALII_MAX_JOBS
    16
    CACHE STRING "Maximum parallel jobs for deal.II build")
set(_dealii_jobs ${_half_cores})
if(_max_threads_by_ram LESS _dealii_jobs)
  set(_dealii_jobs ${_max_threads_by_ram})
endif()
if(_dealii_jobs GREATER ${DEALII_MAX_JOBS})
  set(_dealii_jobs ${DEALII_MAX_JOBS})
endif()
message(
  "  ${BoldGreen}[deal.II] build parallelism: ${_dealii_jobs} threads (${_total_ram_mb} MB RAM, ${_nproc} cores).${ColourReset}"
)

option(DEVEL "Enable development mode (deal.II DebugRelease build)" OFF)
if(DEVEL)
  set(dealiiConfig "DebugRelease")
else()
  set(dealiiConfig "Release")
endif()

# ############################################################################
# deal.II <-> Kokkos compiler reconciliation
# ############################################################################
# deal.II consumes the bundled Kokkos via find_package() and inherits its
# INTERFACE_COMPILE_OPTIONS. When Kokkos' CUDA backend runs on the *nvcc*
# toolchain (i.e. the host compiler is GCC, which is not CUDA-aware), Kokkos
# bakes nvcc-only flags (-arch=sm_89, -extended-lambda, -expt-relaxed-constexpr,
# ...) into those options. Plain g++ rejects them and deal.II's compiler-flag
# sanity check (DEAL_II_HAVE_USABLE_FLAGS_*) aborts configuration.
#
# Kokkos ships nvcc_wrapper for exactly this case: it forwards the nvcc flags to
# nvcc while delegating host compilation to a chosen host compiler. So when GCC
# is paired with a CUDA-enabled Kokkos, route deal.II through nvcc_wrapper.
#
# nvcc_wrapper must be told to use *our* GCC; left to itself it falls back to the
# bare `g++` on PATH (typically the newest system GCC), whose libstdc++ headers
# nvcc cannot parse. Its only selector that survives CMake's internal
# try_compile probes -- the working-compiler check and deal.II's feature tests
# all run before any *_CXX_FLAGS we set take effect -- is the
# NVCC_WRAPPER_DEFAULT_COMPILER environment variable (this is also how Kokkos
# drives nvcc_wrapper internally). Bake it into a tiny shim used as the compiler
# so every invocation, probes and build alike, delegates to the configured GCC.
# The clang-CUDA path needs none of this (clang is itself CUDA-aware), hence the
# GNU guard.
set(_dealii_cxx_compiler "${CMAKE_CXX_COMPILER}")
if(HAVE_CUDA AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(_nvcc_wrapper_shim "${CMAKE_CURRENT_BINARY_DIR}/dealii_nvcc_wrapper.sh")
  file(
    WRITE "${_nvcc_wrapper_shim}"
    "#!/bin/sh\n"
    "# Generated by DiFfRG: pin nvcc_wrapper's host compiler to the configured GCC.\n"
    "export NVCC_WRAPPER_DEFAULT_COMPILER=\"${CMAKE_CXX_COMPILER}\"\n"
    "exec \"${DEPEND_INSTALL_PREFIX}/bin/nvcc_wrapper\" \"$@\"\n")
  file(CHMOD "${_nvcc_wrapper_shim}"
       PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
                   GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  set(_dealii_cxx_compiler "${_nvcc_wrapper_shim}")
  message(
    STATUS
    "${BoldYellow}[deal.II] GCC host compiler with a CUDA-enabled Kokkos: routing deal.II through nvcc_wrapper (host compiler ${CMAKE_CXX_COMPILER}).${ColourReset}"
  )
endif()

ExternalProject_Add(
  deal.II_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  GIT_REPOSITORY https://github.com/dealii/dealii.git
  GIT_TAG v9.7.1
  GIT_SHALLOW ON
  # Patch deal.II 9.7.1 for DiFfRG's IDA diagnostics/max-step handling and for
  # clang-CUDA + Kokkos 5.x + Boost 1.9x (see deal.II issue #18797). Idempotent:
  # check-reverse first (silently) so an already-patched tree is detected without
  # the forward apply spewing "patch does not apply" errors on every re-run/
  # re-clone. If neither reverse-check nor forward-apply succeeds the tree is in a
  # *partially* applied state -- an existing build/ patched by an older revision of
  # the patch file -- so restore the touched files to pristine and apply cleanly.
  # sh -c keeps it working on Linux and macOS.
  PATCH_COMMAND
    sh -c
    "git apply --reverse --check --whitespace=nowarn '${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dealii_ida_max_num_steps.patch' 2>/dev/null && echo 'deal.II IDA patch already applied' || git apply --whitespace=nowarn '${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dealii_ida_max_num_steps.patch' 2>/dev/null || ( echo 'deal.II IDA patch stale -- restoring and re-applying' && git checkout -- include/deal.II/sundials/ida.h source/sundials/ida.cc && git apply --whitespace=nowarn '${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dealii_ida_max_num_steps.patch' )"
    COMMAND
    sh -c
    "git apply --reverse --check --whitespace=nowarn '${CMAKE_CURRENT_SOURCE_DIR}/patches/dealii-9.7.1-clang-cuda.patch' 2>/dev/null && echo 'deal.II patch already applied' || git apply --whitespace=nowarn '${CMAKE_CURRENT_SOURCE_DIR}/patches/dealii-9.7.1-clang-cuda.patch'"
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/deal.II
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep kokkos_dep ${_boost_dep} ${is_petsc_dep} ${_sundials_dep}
  BUILD_COMMAND cmake --build . -- -j${_dealii_jobs}
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:STRING=${_dealii_cxx_compiler}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_Fortran_COMPILER:STRING=${CMAKE_Fortran_COMPILER}
    -DCMAKE_BUILD_TYPE:STRING=${dealiiConfig}
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
    -DCMAKE_POLICY_DEFAULT_CMP0167:STRING=NEW
    -DDEAL_II_UNITY_BUILD:BOOL=ON
    -DCMAKE_WARN_DEPRECATED:BOOL=OFF
    ${_native_dealii_arg}
    -DDEAL_II_COMPONENT_EXAMPLES:BOOL=OFF
    -DDEAL_II_COMPONENT_DOCUMENTATION:BOOL=OFF
    -DDEAL_II_WITH_MPI:BOOL=${MPI}
    -DDEAL_II_WITH_KOKKOS:BOOL=ON
    -DDEAL_II_WITH_UMFPACK:BOOL=ON
    -DDEAL_II_WITH_TASKFLOW:BOOL=OFF
    ${dealii_lapack}
    -DDEAL_II_WITH_LAPACK:BOOL=ON
    -DDEAL_II_WITH_HDF5:BOOL=OFF
    -DDEAL_II_WITH_VTK:BOOL=OFF
    # PETSc: enable it explicitly -- passing only PETSC_DIR/PETSC_ARCH (as this did
    # before) leaves DEAL_II_WITH_PETSC off, so even a successful PETSc build had no
    # effect at all. PETSC_ARCH must be *empty* for a --prefix= install: the
    # installed tree has no arch subdirectory, and a stale "arch-linux-c-opt" sends
    # deal.II's FindPETSC looking in a path that does not exist.
    -DDEAL_II_WITH_PETSC:BOOL=${BUILD_PETSC}
    ${_dealii_petsc_args}
    # DEAL_II_ALLOW_AUTODETECTION defaults to ON, so a system p4est or Trilinos would
    # otherwise be picked up silently and change what the bundled deal.II supports.
    # Pin them off; P4EST flips on when the mesh itself becomes distributed.
    -DDEAL_II_WITH_P4EST:BOOL=OFF
    -DDEAL_II_WITH_TRILINOS:BOOL=OFF
    -DCMAKE_CXX_STANDARD:STRING=20
    -DCMAKE_CXX_EXTENSIONS:BOOL=OFF
    -DKOKKOS_DIR:STRING=${DEPEND_INSTALL_PREFIX}
    -DTBB_DIR:STRING=${_tbb_dir}
    # Required by DiFfRG -- force ON so an unusable SUNDIALS fails here at
    # configure rather than silently disabling and breaking kinsol.cc later.
    -DDEAL_II_WITH_SUNDIALS:BOOL=ON
    ${_dealii_sundials_arg}
    -DBOOST_ROOT:PATH=${_boost_root}
    -DBOOST_DIR:STRING=${_boost_root}
    -DBOOST_INCLUDEDIR:PATH=${_boost_incdir}
    -DBOOST_LIBRARYDIR:PATH=${_boost_libdir}
    ${_boost_extra_args}
    -DCMAKE_PREFIX_PATH:STRING=${DEPEND_INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
    ${CCACHE_CMAKE_ARGS})

# create a directory necessary for using deal.II
file(MAKE_DIRECTORY ${DEPEND_INSTALL_PREFIX}/include/deal.II/bundled)

# ##############################################################################
# Pin resolved dependencies for standalone library rebuilds
# ##############################################################################
#
# Record exactly which Boost/TBB/HDF5 this superbuild resolved (system vs
# bundled) into a small CMake snippet inside the bundle dir. A later standalone
# library build (cd DiFfRG && cmake .. -DBUNDLED_DIR=...) and downstream
# find_package(DiFfRG) consumers include it (see
# DiFfRG/cmake/setup_build_system.cmake) so they reuse exactly these
# dependencies instead of re-running find_package and possibly latching onto a
# different system install or a stray copy in the bundle dir.
#
# Written at configure time (file(WRITE) creates missing parents) and
# overwritten on every configure, so switching system<->bundled regenerates a
# correct pin. Pinned hint variables are guarded with if(NOT DEFINED ...) so an
# explicit -DBOOST_ROOT=/-DTBB_DIR=/-DHDF5_DIR= on the command line still wins.

# A bundled Boost build compiles Boost itself, so its version is not yet known
# via find_package; read it from the bundled-Boost wrapper CMakeLists. For a
# system Boost we already have Boost_VERSION.
set(_pinned_boost_version "${Boost_VERSION}")
if(_build_boost)
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/boost/CMakeLists.txt"
       _boost_ver_line REGEX "set\\(TRY_BOOST_VERSION")
  if(_boost_ver_line)
    string(REGEX REPLACE ".*\"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1"
                         _pinned_boost_version "${_boost_ver_line}")
  else()
    message(
      FATAL_ERROR
        "Could not read TRY_BOOST_VERSION from dependencies/boost/CMakeLists.txt")
  endif()
endif()

set(_pin_content
    "# Auto-generated by the DiFfRG superbuild -- do not edit.
# Pins the Boost/TBB/HDF5 the superbuild resolved so a standalone library
# rebuild given this bundle dir reuses exactly the same dependencies.
# if(NOT DEFINED) guards let an explicit -DX= on the command line override.

")

# Boost: always pin BOOST_ROOT. For a system Boost also pin the exact
# BoostConfig.cmake dir (Boost_DIR) -- with CMP0167 NEW, config-mode
# find_package then resolves straight to it and ignores HINTS/CMAKE_PREFIX_PATH,
# so a stray Boost in the bundle dir is never picked up.
string(APPEND _pin_content
       "if(NOT DEFINED BOOST_ROOT)\n  set(BOOST_ROOT \"${_boost_root}\")\nendif()\n")
if(NOT _build_boost)
  string(APPEND _pin_content
         "if(NOT DEFINED Boost_DIR)\n  set(Boost_DIR \"${Boost_DIR}\")\nendif()\n")
endif()
string(APPEND _pin_content
       "set(DiFfRG_PINNED_BOOST_VERSION \"${_pinned_boost_version}\")\n\n")

# TBB: pin the config dir (bundled prefix or discovered TBBConfig.cmake dir).
string(APPEND _pin_content
       "if(NOT DEFINED TBB_DIR)\n  set(TBB_DIR \"${_tbb_dir}\")\nendif()\n")
if(NOT _build_tbb)
  string(APPEND _pin_content
         "set(DiFfRG_PINNED_TBB_VERSION \"${TBB_VERSION}\")\n")
endif()
string(APPEND _pin_content "\n")

# HDF5: only a system HDF5 needs pinning (bundled lives in BUNDLED_DIR and is
# found by HINTS). Mirror the config-mode (HDF5_DIR) vs module-mode (HDF5_ROOT)
# branch used above for the DiFfRG ExternalProject.
if(NOT _build_hdf5)
  if(HDF5_DIR AND NOT "${HDF5_DIR}" STREQUAL "HDF5_DIR-NOTFOUND")
    string(APPEND _pin_content
           "if(NOT DEFINED HDF5_DIR)\n  set(HDF5_DIR \"${HDF5_DIR}\")\nendif()\n")
  else()
    string(APPEND _pin_content
           "if(NOT DEFINED HDF5_ROOT)\n  set(HDF5_ROOT \"${_hdf5_loc}\")\nendif()\n")
  endif()
  string(APPEND _pin_content
         "set(DiFfRG_PINNED_HDF5_VERSION \"${HDF5_VERSION}\")\n")
endif()

file(WRITE "${DEPEND_INSTALL_PREFIX}/DiFfRG_bundled_config.cmake" "${_pin_content}")
message(
  "  ${BoldGreen}[pin] Recorded resolved dependencies in ${DEPEND_INSTALL_PREFIX}/DiFfRG_bundled_config.cmake${ColourReset}"
)

# ##############################################################################
# DiFfRG
# ##############################################################################

option(DiFfRG_TEST "Build DiFfRG tests" OFF)
option(DiFfRG_DOCUMENTATION "Build DiFfRG documentation" ON)

# Only forward a CUDA compiler if one was resolved; passing an empty value would pin the
# cache entry to nothing and break the sub-build's own detection.
if(CMAKE_CUDA_COMPILER)
  set(_diffrg_cuda_compiler_arg
      "-DCMAKE_CUDA_COMPILER:STRING=${CMAKE_CUDA_COMPILER}")
else()
  set(_diffrg_cuda_compiler_arg "")
endif()

message("${BoldCyan}[8/8] Configuring DiFfRG...${ColourReset}")
ExternalProject_Add(
  DiFfRG
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/DiFfRG/
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  DEPENDS general_dep ${_boost_dep} deal.II_dep kokkos_dep autodiff_dep
  CMAKE_CACHE_ARGS
    # Forward the toolchain. Without this the sub-build silently falls back to CMake's
    # default compiler (/usr/bin/cc, /usr/bin/c++), which need not be -- and on a machine
    # with a newer system GCC usually is not -- the compiler deal.II and Kokkos were built
    # with. Every other dependency here already gets the compiler passed explicitly; DiFfRG
    # itself did not, so its toolchain depended on whether CC/CXX happened to be exported
    # in the caller's environment.
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    ${_diffrg_cuda_compiler_arg}
    -DMPI:BOOL=${MPI}
    -DHDF5:BOOL=ON
    -DBUNDLED_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    -DBOOST_ROOT:PATH=${_boost_root}
    ${_boost_extra_args}
    -DTBB_DIR:STRING=${_tbb_dir}
    ${_hdf5_extra_args}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DNATIVE:BOOL=${NATIVE}
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
