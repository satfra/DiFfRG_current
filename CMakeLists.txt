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

# ##############################################################################
# Dependency selection: system vs bundled
#
# Boost, TBB, HDF5 and SUNDIALS can each be taken from the system instead of
# being rebuilt. For every one of them:
#   <LIB>_DIR set     -> use the install at that prefix (FATAL if unusable)
#   BUILD_<LIB>=ON    -> always build the bundled, pinned copy
#   otherwise (auto)  -> use a viable system install if found, else build it
# The blocks below resolve, for each library, whether the bundled ExternalProject
# is needed (_build_<lib>) and which location to hand to deal.II / the DiFfRG
# library build.
# ##############################################################################

set(BOOST_DIR "" CACHE PATH
    "Use the Boost install at this prefix instead of building one")
option(BUILD_BOOST "Always build the bundled, pinned Boost (ignore any system Boost)" OFF)
set(TBB_DIR "" CACHE PATH
    "Use the TBB install at this prefix instead of building one")
option(BUILD_TBB "Always build the bundled oneTBB (ignore any system TBB)" OFF)
set(HDF5_DIR "" CACHE PATH
    "Use the HDF5 install at this prefix instead of building one")
option(BUILD_HDF5 "Always build the bundled HDF5 (ignore any system HDF5)" OFF)
set(SUNDIALS_DIR "" CACHE PATH
    "Use the SUNDIALS install at this prefix instead of building one")
option(BUILD_SUNDIALS "Always build the bundled SUNDIALS (ignore any system SUNDIALS)" OFF)

set(DiFfRG_BOOST_MIN_VERSION 1.81)
set(DiFfRG_TBB_MIN_VERSION 2021)
set(DiFfRG_HDF5_MIN_VERSION 1.12.0)
set(DiFfRG_SUNDIALS_MIN_VERSION 5.4.0)   # deal.II's own SUNDIALS floor
set(DiFfRG_SUNDIALS_KNOWN_GOOD_MAX 7.7)  # newest version validated with bundled deal.II

# ----------------------------------------------------------------------------
# Boost — needed by both deal.II and the DiFfRG library. Use Boost's own
# BoostConfig.cmake (config mode); the legacy FindBoost module is removed in
# CMake >= 3.30 and deal.II 9.7 consumes a modern config-mode Boost fine.
# ----------------------------------------------------------------------------
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
set(_boost_components thread iostreams serialization system)
set(_build_boost OFF)
set(_boost_user_dir "${BOOST_DIR}")
if(NOT "${_boost_user_dir}" STREQUAL "")
  set(BOOST_ROOT "${_boost_user_dir}")
  find_package(Boost ${DiFfRG_BOOST_MIN_VERSION} QUIET COMPONENTS ${_boost_components})
  if(NOT Boost_FOUND)
    message(FATAL_ERROR
      "BOOST_DIR=${_boost_user_dir} does not provide a usable Boost >= ${DiFfRG_BOOST_MIN_VERSION}.\n"
      "  Point it at a prefix containing BoostConfig.cmake, or use -DBUILD_BOOST=ON.")
  endif()
  message(STATUS "[Boost] Using Boost ${Boost_VERSION} from BOOST_DIR=${_boost_user_dir}")
elseif(BUILD_BOOST)
  set(_build_boost ON)
else()
  find_package(Boost ${DiFfRG_BOOST_MIN_VERSION} QUIET COMPONENTS ${_boost_components})
  if(NOT Boost_FOUND)
    set(_build_boost ON)
  endif()
endif()

if(_build_boost)
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
  message(
    "  ${BoldGreen}[Boost] Using system Boost ${Boost_VERSION} (${_boost_incdir}); skipping bundled build.${ColourReset}"
  )
endif()

# ----------------------------------------------------------------------------
# TBB — needed by both deal.II (path-based) and the DiFfRG library
# (find_package(TBB) -> TBB::tbb). The bundled oneTBB is built inside general_dep
# (see GENERAL_BUILD_TBB), not its own ExternalProject.
# ----------------------------------------------------------------------------
set(_build_tbb OFF)
# find_package(TBB) overwrites TBB_DIR with the discovered config dir; capture
# the user-given prefix first.
set(_tbb_user_dir "${TBB_DIR}")
if(NOT "${_tbb_user_dir}" STREQUAL "")
  find_package(TBB ${DiFfRG_TBB_MIN_VERSION} QUIET CONFIG
    PATHS "${_tbb_user_dir}"
    PATH_SUFFIXES lib/cmake/TBB lib64/cmake/TBB cmake/TBB
    NO_DEFAULT_PATH)
  if(NOT TBB_FOUND)
    message(FATAL_ERROR
      "TBB_DIR=${_tbb_user_dir} does not provide TBB >= ${DiFfRG_TBB_MIN_VERSION}.\n"
      "  Point it at a prefix containing TBBConfig.cmake, or use -DBUILD_TBB=ON.")
  endif()
  message(STATUS "[TBB] Using TBB ${TBB_VERSION} from TBB_DIR=${_tbb_user_dir}")
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
  message(
    "  ${BoldGreen}[TBB] Using system TBB ${TBB_VERSION} (${TBB_DIR}); skipping bundled build.${ColourReset}"
  )
endif()

# ----------------------------------------------------------------------------
# HDF5 — needed only by the DiFfRG library (deal.II is built with
# DEAL_II_WITH_HDF5=OFF). DiFfRG only uses the HDF5 C API, so 1.12 is the floor.
# ----------------------------------------------------------------------------
set(_build_hdf5 OFF)
set(_hdf5_user_dir "${HDF5_DIR}")
# Do not pass the version to find_package: HDF5's config-version file uses a
# same-major-version compatibility policy, so requesting 1.12 would reject a
# newer 2.x install. Gate the version manually instead.
# Many distros (Fedora, Debian, Ubuntu, ...) ship HDF5 without a CMake config
# package (only headers + libs + the FindHDF5 module / pkg-config). So try config
# mode first (gives imported targets; matches the bundled static build and Arch),
# then fall back to module mode (FindHDF5) for those config-less installs.
macro(_diffrg_find_hdf5)
  find_package(HDF5 QUIET CONFIG COMPONENTS C ${ARGN})
  if(NOT HDF5_FOUND OR HDF5_VERSION VERSION_LESS ${DiFfRG_HDF5_MIN_VERSION})
    find_package(HDF5 QUIET MODULE COMPONENTS C)
  endif()
endmacro()

if(NOT "${_hdf5_user_dir}" STREQUAL "")
  set(HDF5_ROOT "${_hdf5_user_dir}")
  _diffrg_find_hdf5(PATHS "${_hdf5_user_dir}"
    PATH_SUFFIXES lib/cmake/hdf5 lib64/cmake/hdf5 cmake/hdf5 share/cmake/hdf5)
  if(NOT HDF5_FOUND OR HDF5_VERSION VERSION_LESS ${DiFfRG_HDF5_MIN_VERSION})
    message(FATAL_ERROR
      "HDF5_DIR=${_hdf5_user_dir} does not provide a usable HDF5 >= ${DiFfRG_HDF5_MIN_VERSION} "
      "(found '${HDF5_VERSION}').\n"
      "  Point it at an HDF5 install prefix, or use -DBUILD_HDF5=ON.")
  endif()
  message(STATUS "[HDF5] Using HDF5 ${HDF5_VERSION} from HDF5_DIR=${_hdf5_user_dir}")
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
  message(
    "  ${BoldGreen}[HDF5] Using system HDF5 ${HDF5_VERSION} (${_hdf5_loc}); skipping bundled build.${ColourReset}"
  )
endif()

# ----------------------------------------------------------------------------
# SUNDIALS — needed only by deal.II (DiFfRG links it transitively via
# dealii::dealii). Replicate deal.II's own discovery: the arkode/kinsol/nvecserial
# libraries plus version >= 5.4.0; otherwise build the bundled copy.
# ----------------------------------------------------------------------------
set(_build_sundials OFF)
set(_sundials_prefix "${DEPEND_INSTALL_PREFIX}")
set(_sundials_dep "")
set(_sun_hint "")
set(_sun_explicit OFF)
if(NOT "${SUNDIALS_DIR}" STREQUAL "")
  set(_sun_hint "${SUNDIALS_DIR}")
  set(_sun_explicit ON)
elseif(BUILD_SUNDIALS)
  set(_build_sundials ON)
endif()

if(NOT _build_sundials)
  find_library(_SUN_ARKODE NAMES sundials_arkode HINTS ${_sun_hint} PATH_SUFFIXES lib lib64)
  find_library(_SUN_KINSOL NAMES sundials_kinsol HINTS ${_sun_hint} PATH_SUFFIXES lib lib64)
  find_library(_SUN_NVEC NAMES sundials_nvecserial HINTS ${_sun_hint} PATH_SUFFIXES lib lib64)
  # The version macros live in sundials_config.h on most installs (sundials_version.h
  # only declares the runtime accessors and #includes the config -- e.g. on Fedora
  # it carries no macros), so locate the config header and read whichever of the
  # two actually defines SUNDIALS_VERSION_MAJOR.
  find_path(_SUN_INC sundials/sundials_config.h HINTS ${_sun_hint} PATH_SUFFIXES include)
  set(_sun_ok FALSE)
  set(_sun_ver "")
  set(_sun_reject_reason "")
  if(_SUN_ARKODE AND _SUN_KINSOL AND _SUN_NVEC AND _SUN_INC)
    set(_sun_verhdr "")
    foreach(_h sundials/sundials_config.h sundials/sundials_version.h)
      if(EXISTS "${_SUN_INC}/${_h}")
        file(STRINGS "${_SUN_INC}/${_h}" _sun_probe REGEX "#define[ \t]+SUNDIALS_VERSION_MAJOR")
        if(_sun_probe)
          set(_sun_verhdr "${_SUN_INC}/${_h}")
          break()
        endif()
      endif()
    endforeach()
    if(_sun_verhdr)
      file(STRINGS "${_sun_verhdr}" _sun_mj REGEX "#define[ \t]+SUNDIALS_VERSION_MAJOR")
      file(STRINGS "${_sun_verhdr}" _sun_mn REGEX "#define[ \t]+SUNDIALS_VERSION_MINOR")
      file(STRINGS "${_sun_verhdr}" _sun_pt REGEX "#define[ \t]+SUNDIALS_VERSION_PATCH")
      string(REGEX REPLACE "^.*MAJOR[ \t]+([0-9]+).*" "\\1" _sun_mj "${_sun_mj}")
      string(REGEX REPLACE "^.*MINOR[ \t]+([0-9]+).*" "\\1" _sun_mn "${_sun_mn}")
      string(REGEX REPLACE "^.*PATCH[ \t]+([0-9]+).*" "\\1" _sun_pt "${_sun_pt}")
      set(_sun_ver "${_sun_mj}.${_sun_mn}.${_sun_pt}")
      # deal.II refuses a SUNDIALS whose MPI setting differs from its own
      # (DEAL_II_WITH_MPI). The system package (e.g. Ubuntu's libsundials-dev)
      # is typically MPI-enabled, so a version/library check alone is not enough
      # -- it would be accepted here and then rejected by deal.II at configure.
      # Read SUNDIALS_MPI_ENABLED from sundials_config.h and require it to match.
      set(_sun_mpi 0)
      if(EXISTS "${_SUN_INC}/sundials/sundials_config.h")
        file(STRINGS "${_SUN_INC}/sundials/sundials_config.h" _sun_mpi_line
             REGEX "^[ \t]*#define[ \t]+SUNDIALS_MPI_ENABLED[ \t]+[01]")
        if(_sun_mpi_line)
          string(REGEX REPLACE "^.*SUNDIALS_MPI_ENABLED[ \t]+([01]).*" "\\1"
                 _sun_mpi "${_sun_mpi_line}")
        endif()
      endif()
      if(MPI)
        set(_want_mpi 1)
      else()
        set(_want_mpi 0)
      endif()
      if(NOT _sun_ver MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$"
         OR _sun_ver VERSION_LESS ${DiFfRG_SUNDIALS_MIN_VERSION})
        set(_sun_reject_reason
            "version ${_sun_ver} < required ${DiFfRG_SUNDIALS_MIN_VERSION}")
      elseif(NOT _sun_mpi STREQUAL _want_mpi)
        set(_sun_reject_reason
            "its MPI support (SUNDIALS_MPI_ENABLED=${_sun_mpi}) does not match this build (MPI=${MPI})")
      else()
        set(_sun_ok TRUE)
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
      message(WARNING
        "Using system SUNDIALS ${_sun_ver}, newer than the last version "
        "(${DiFfRG_SUNDIALS_KNOWN_GOOD_MAX}) validated with the bundled deal.II. "
        "If the deal.II build fails, configure with -DBUILD_SUNDIALS=ON.")
    endif()
    message(
      "  ${BoldGreen}[SUNDIALS] Using system SUNDIALS ${_sun_ver} (${_SUN_INC}); skipping bundled build.${ColourReset}"
    )
  elseif(_sun_explicit)
    if(NOT _sun_reject_reason)
      set(_sun_reject_reason
          "the required libraries (arkode, kinsol, nvecserial) or version header were not found")
    endif()
    message(FATAL_ERROR
      "SUNDIALS_DIR=${SUNDIALS_DIR} is not usable: ${_sun_reject_reason}.\n"
      "  Use -DBUILD_SUNDIALS=ON to build the bundled SUNDIALS instead.")
  else()
    if(_sun_reject_reason)
      message(
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
      -DCMAKE_CXX_FLAGS:STRING=-march=native
      -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
      ${CCACHE_CMAKE_ARGS}
      -DMPI:BOOL=${MPI})
else()
  message("  ${BoldGreen}Using system HDF5; skipping bundled HDF5 build.${ColourReset}")
endif()

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
  DEPENDS ${_hdf5_dep}
  # Propagate DEPEND_INSTALL_PREFIX to the dependencies
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=-march=native
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
      -DCMAKE_CXX_FLAGS:STRING=-march=native
      -DCMAKE_C_FLAGS:STRING=-march=native
      -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
      ${CCACHE_CMAKE_ARGS}
      -DMPI:BOOL=${MPI})
else()
  message("  ${BoldGreen}Using system SUNDIALS; skipping bundled SUNDIALS build.${ColourReset}")
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
  CMAKE_CACHE_ARGS -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
                   -DCMAKE_INSTALL_PREFIX:PATH=${DEPEND_INSTALL_PREFIX}
                   ${CCACHE_CMAKE_ARGS})

message("${BoldCyan}[5/8] Configuring Boost...${ColourReset}")
# Boost selection was resolved above; only register the bundled build if needed.
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
  message("  ${BoldGreen}Using system Boost; skipping bundled Boost build.${ColourReset}")
endif()

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
    -DBUILD_SHARED_LIBS:BOOL=OFF
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
# Use the lesser of half cores, RAM-based limit, and hard cap
set(DEALII_MAX_JOBS 16 CACHE STRING "Maximum parallel jobs for deal.II build")
set(_dealii_jobs ${_half_cores})
if(_max_threads_by_ram LESS _dealii_jobs)
  set(_dealii_jobs ${_max_threads_by_ram})
endif()
if(_dealii_jobs GREATER ${DEALII_MAX_JOBS})
  set(_dealii_jobs ${DEALII_MAX_JOBS})
endif()
message(
  "  deal.II build parallelism: ${_dealii_jobs} threads (${_total_ram_mb} MB RAM, ${_nproc} cores)"
)

option(DEVEL "Enable development mode (deal.II DebugRelease build)" OFF)
if(DEVEL)
  set(dealiiConfig "DebugRelease")
else()
  set(dealiiConfig "Release")
endif()

message("${BoldCyan}[7/8] Configuring deal.II...${ColourReset}")
ExternalProject_Add(
  deal.II_dep
  LOG_CONFIGURE ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  GIT_REPOSITORY https://github.com/dealii/dealii.git
  GIT_TAG v9.7.1
  GIT_SHALLOW ON
  # Patch deal.II 9.7.1 for clang-CUDA + Kokkos 5.x + Boost 1.9x (see
  # patches/dealii-9.7.1-clang-cuda.patch and deal.II issue #18797).
  # Idempotent: if already applied, the reverse --check succeeds so re-runs and
  # re-clones are both safe. sh -c keeps it working on Linux and macOS.
  PATCH_COMMAND sh -c "git apply --whitespace=nowarn '${CMAKE_CURRENT_SOURCE_DIR}/patches/dealii-9.7.1-clang-cuda.patch' || git apply --reverse --check --whitespace=nowarn '${CMAKE_CURRENT_SOURCE_DIR}/patches/dealii-9.7.1-clang-cuda.patch'"
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/deal.II
  INSTALL_DIR ${DEPEND_INSTALL_PREFIX}
  DEPENDS general_dep ${_boost_dep} ${is_petsc_dep} ${_sundials_dep}
  BUILD_COMMAND cmake --build . -- -j${_dealii_jobs}
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_Fortran_COMPILER:STRING=${CMAKE_Fortran_COMPILER}
    -DCMAKE_BUILD_TYPE:STRING=${dealiiConfig}
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
    -DCMAKE_POLICY_DEFAULT_CMP0167:STRING=NEW
    -DDEAL_II_UNITY_BUILD:BOOL=ON
    -DCMAKE_WARN_DEPRECATED:BOOL=OFF
    -DDEAL_II_CXX_FLAGS:STRING=-march=native
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
    -DPETSC_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    -DPETSC_ARCH:STRING=arch-linux-c-opt
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
  DEPENDS general_dep ${_boost_dep} deal.II_dep kokkos_dep autodiff_dep
  CMAKE_CACHE_ARGS
    -DMPI:BOOL=${MPI}
    -DHDF5:BOOL=ON
    -DBUNDLED_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    -DBOOST_ROOT:PATH=${_boost_root}
    ${_boost_extra_args}
    -DTBB_DIR:STRING=${_tbb_dir}
    ${_hdf5_extra_args}
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
