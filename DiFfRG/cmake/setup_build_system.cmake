# ##############################################################################
# Setup directories
# ##############################################################################

# We need to find the bundle directory, which contains several external
# dependencies
if(${CMAKE_PROJECT_NAME} STREQUAL "DiFfRG")
  if(NOT DEFINED MPI)
    set(MPI
        OFF
        CACHE BOOL "Whether to build with MPI support (default: OFF)")
  endif()

  # If we are building DiFfRG as a standalone project, we need to set the base
  # directory
  set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(DiFfRG_MPI ${MPI})
else()
  # If we are building a DiFfRG-based project, we need to set the bundle
  # directory relative to the DiFfRG base directory
  set(BASE_DIR ${DiFfRG_BASE_DIR})
endif()

# ##############################################################################
# Validate BUNDLED_DIR
# ##############################################################################

if(NOT DEFINED BUNDLED_DIR OR "${BUNDLED_DIR}" STREQUAL "")
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  BUNDLED_DIR is not set.\n"
      "======================================================================\n"
      "  DiFfRG needs to know where its bundled dependencies are installed.\n"
      "\n"
      "  If you have not built the dependencies yet, build from the top-level\n"
      "  repository directory first:\n"
      "    mkdir build && cd build\n"
      "    cmake .. -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG\n"
      "    cmake --build . -- -j8\n"
      "\n"
      "  Then configure DiFfRG with:\n"
      "    cmake .. -DBUNDLED_DIR=~/.local/share/DiFfRG/bundled\n"
      "======================================================================\n"
  )
endif()

if(NOT EXISTS "${BUNDLED_DIR}")
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  BUNDLED_DIR does not exist: ${BUNDLED_DIR}\n"
      "======================================================================\n"
      "  The specified dependency directory was not found. This usually means\n"
      "  the dependencies have not been built yet.\n"
      "\n"
      "  Build from the top-level repository directory:\n"
      "    mkdir build && cd build\n"
      "    cmake .. -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG\n"
      "    cmake --build . -- -j8\n"
      "\n"
      "  Then re-run this cmake configuration.\n"
      "======================================================================\n"
  )
endif()

set(CMAKE_PREFIX_PATH "${BUNDLED_DIR};${BUNDLED_DIR}/lib;${CMAKE_PREFIX_PATH}")

link_directories(${BUNDLED_DIR}/lib/)
link_directories(${BUNDLED_DIR}/lib64/)
# Emit -I (not -isystem) for the bundled include root so it wins over CPATH
# entries injected by environment modules. Cluster HDF5/MPI modules set CPATH
# to point at parallel HDF5 builds whose hdf5.h pulls in mpi.h and clashes
# with deal.II's MPI stubs when DEAL_II_WITH_MPI=OFF. -isystem would lose to
# CPATH in GCC's search order; -I wins.
include_directories(${BUNDLED_DIR}/include)

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
# Helper macro for find_package with actionable errors
# ##############################################################################

macro(diffrg_find_package pkg)
  # Parse optional arguments: version and extra hints
  cmake_parse_arguments(_DFP "" "VERSION" "HINTS;COMPONENTS" ${ARGN})

  set(_dfp_args "")
  if(_DFP_VERSION)
    list(APPEND _dfp_args "${_DFP_VERSION}")
  endif()
  list(APPEND _dfp_args QUIET)
  if(_DFP_HINTS)
    list(APPEND _dfp_args HINTS ${_DFP_HINTS})
  endif()
  if(_DFP_COMPONENTS)
    list(APPEND _dfp_args COMPONENTS ${_DFP_COMPONENTS})
  endif()

  find_package(${pkg} ${_dfp_args})

  if(NOT ${pkg}_FOUND)
    message(
      FATAL_ERROR
        "\n"
        "======================================================================\n"
        "  Required dependency not found: ${pkg}\n"
        "======================================================================\n"
        "  CMake could not find '${pkg}' in BUNDLED_DIR=${BUNDLED_DIR}\n"
        "\n"
        "  This usually means the bundled dependencies need to be (re)built.\n"
        "  Build from the top-level repository directory:\n"
        "    mkdir build && cd build\n"
        "    cmake .. -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG\n"
        "    cmake --build . -- -j8\n"
        "\n"
        "  Then re-run this cmake configuration.\n"
        "======================================================================\n"
    )
  endif()
endmacro()

# ##############################################################################
# Direct external dependencies
# ##############################################################################

# Find deal.II
diffrg_find_package(deal.II VERSION 9.4.2 HINTS ${BUNDLED_DIR})
deal_ii_initialize_cached_variables()
message(STATUS "Found deal.II in  ${deal.II_DIR}")

# Find TBB
diffrg_find_package(TBB VERSION 2022.0.0 HINTS ${BUNDLED_DIR})
message(STATUS "Found TBB in ${TBB_DIR}")

# Find Kokkos
diffrg_find_package(Kokkos HINTS ${BUNDLED_DIR})
message(STATUS "Found Kokkos in ${Kokkos_DIR}")

# Find Boost
diffrg_find_package(
  Boost
  VERSION
  1.81
  HINTS
  "${BUNDLED_DIR}/"
  "${BUNDLED_DIR}/boost_install/lib/"
  COMPONENTS
  thread
  iostreams
  serialization
  system)
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
# Non-SYSTEM: see comment near include_directories(${BUNDLED_DIR}/include) above.
# Boost_INCLUDE_DIRS typically resolves to ${BUNDLED_DIR}/include, and any
# -isystem on that same path would let CPATH (cluster modules) win for <hdf5.h>.
include_directories(${Boost_INCLUDE_DIRS})

# Find Eigen3
diffrg_find_package(Eigen3 VERSION 3.4.0 HINTS ${BUNDLED_DIR})

# Find GSL (system dependency)
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

# Find autodiff
diffrg_find_package(autodiff VERSION 1.1.0 HINTS ${BUNDLED_DIR})

# Find spdlog
diffrg_find_package(spdlog VERSION 1.14.1 HINTS ${BUNDLED_DIR})

# Find HDF5 (static, minimal)
diffrg_find_package(HDF5 VERSION 2.0.0 HINTS ${BUNDLED_DIR})
message(STATUS "HDF5 include dir: ${HDF5_INCLUDE_DIRS}")
add_compile_definitions(H5CPP)

if(${DiFfRG_MPI})
  find_package(MPI REQUIRED)
endif()

# ##############################################################################
# Dependency summary
# ##############################################################################

message("")
message(
  "${BoldWhite}======================================================================${ColourReset}"
)
message("${BoldWhite}  DiFfRG Dependency Summary${ColourReset}")
message(
  "${BoldWhite}======================================================================${ColourReset}"
)

# Resolve a path string for Boost: prefer Boost_DIR (CMake config package),
# otherwise fall back to the first include directory.
if(DEFINED Boost_DIR AND NOT "${Boost_DIR}" STREQUAL "Boost_DIR-NOTFOUND")
  set(_boost_path "${Boost_DIR}")
else()
  list(GET Boost_INCLUDE_DIRS 0 _boost_path)
endif()

# Resolve a path string for HDF5: prefer HDF5_DIR if it points at a CMake
# config package, otherwise fall back to the first include directory.
if(DEFINED HDF5_DIR AND NOT "${HDF5_DIR}" STREQUAL "HDF5_DIR-NOTFOUND")
  set(_hdf5_path "${HDF5_DIR}")
else()
  list(GET HDF5_INCLUDE_DIRS 0 _hdf5_path)
endif()

# Pretty-print one summary row with fixed-width name and version columns so
# the (path) field aligns identically across every dependency. ARGN is an
# optional trailing annotation (e.g. "[static]").
function(_diffrg_summary_row name version path)
  set(_name_width 10)
  set(_version_width 14)

  string(LENGTH "${name}" _nlen)
  string(LENGTH "${version}" _vlen)
  math(EXPR _npad "${_name_width} - ${_nlen}")
  math(EXPR _vpad "${_version_width} - ${_vlen}")
  if(_npad LESS 1)
    set(_npad 1)
  endif()
  if(_vpad LESS 1)
    set(_vpad 1)
  endif()
  string(REPEAT " " ${_npad} _nspaces)
  string(REPEAT " " ${_vpad} _vspaces)

  set(_suffix "")
  if(ARGN)
    set(_suffix " ${ARGN}")
  endif()

  message(
    "  ${BoldGreen}${name}${ColourReset}${_nspaces}${version}${_vspaces}(${path})${_suffix}"
  )
endfunction()

_diffrg_summary_row("deal.II"  "${deal.II_VERSION}"  "${deal.II_DIR}")
_diffrg_summary_row("TBB"      "${TBB_VERSION}"      "${TBB_DIR}")
_diffrg_summary_row("Kokkos"   "${Kokkos_VERSION}"   "${Kokkos_DIR}")
_diffrg_summary_row("Boost"    "${Boost_VERSION}"    "${_boost_path}")
_diffrg_summary_row("Eigen3"   "${Eigen3_VERSION}"   "${Eigen3_DIR}")
_diffrg_summary_row("GSL"      "${GSL_VERSION}"      "${GSL_INCLUDE_DIRS}")
_diffrg_summary_row("autodiff" "${autodiff_VERSION}" "${autodiff_DIR}")
_diffrg_summary_row("spdlog"   "${spdlog_VERSION}"   "${spdlog_DIR}")
_diffrg_summary_row("HDF5"     "${HDF5_VERSION}"     "${_hdf5_path}" "[static]")
if(${DiFfRG_MPI})
  _diffrg_summary_row("MPI"    "${MPI_CXX_VERSION}"  "${MPI_CXX_INCLUDE_DIRS}")
endif()
message(
  "${BoldWhite}======================================================================${ColourReset}"
)
message("")

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
  # Non-SYSTEM: see comment near include_directories(${BUNDLED_DIR}/include).
  # DEAL_II_INCLUDE_DIRS contains ${BUNDLED_DIR}/include; emitting it as -isystem
  # would let CPATH (cluster HDF5 module) shadow the bundled <hdf5.h>.
  target_include_directories(${TARGET} PUBLIC ${DEAL_II_INCLUDE_DIRS})

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
  # target_link_libraries(${TARGET} PUBLIC petsc)

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

  # Workaround: spdlog's bundled fmt uses consteval for format-string checking,
  # which breaks on newer compilers. constexpr is functionally equivalent.
  target_compile_definitions(${TARGET} PUBLIC FMT_CONSTEVAL=constexpr)
  # Workaround: deal.II's tensor.h uses assert() without including <cassert>.
  target_compile_options(
    ${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-include cassert>)
endfunction()
