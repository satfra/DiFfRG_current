# ##############################################################################
# ## DiFfRG installation script
# ##############################################################################
#
# This script installs the DiFfRG library from a specific branch.
#
# Per default, DiFfRG is installed to $HOME/.local/share/DiFfRG/ However, the
# behavior of this script can be changed by setting the appropriate variables:
#
# * DiFfRG_INSTALL_DIR: The installation directory for DiFfRG (default:
#   $HOME/.local/share/DiFfRG/)
# * DiFfRG_BUILD_DIR: The build directory for DiFfRG (default:
#   $HOME/.local/share/DiFfRG/build/)
# * DiFfRG_SOURCE_DIR: The source directory for DiFfRG (default:
#   $HOME/.local/share/DiFfRG/src/)
# * TRY_DiFfRG_VERSION: The branch of DiFfRG to install (default:
#   Implement-Kokkos)
# * PARALLEL_JOBS: The number of parallel jobs to use for building DiFfRG
#   (default: 8)
#
# ##############################################################################

# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.27)

# Set variables for the installation
if(NOT DEFINED DiFfRG_INSTALL_DIR)
  set(DiFfRG_INSTALL_DIR $ENV{HOME}/.local/share/DiFfRG/)
endif()

if(NOT DEFINED DiFfRG_BUILD_DIR)
  set(DiFfRG_BUILD_DIR $ENV{HOME}/.local/share/DiFfRG/build/)
endif()

if(NOT DEFINED DiFfRG_SOURCE_DIR)
  set(DiFfRG_SOURCE_DIR $ENV{HOME}/.local/share/DiFfRG/src/)
endif()

if(NOT DEFINED TRY_DiFfRG_VERSION)
  set(TRY_DiFfRG_VERSION "Implement-Kokkos")
endif()
if(NOT DEFINED PARALLEL_JOBS)
  set(PARALLEL_JOBS 8)
endif()

# ##############################################################################
# ## Download and install DiFfRG
# ##############################################################################

if(NOT EXISTS ${DiFfRG_SOURCE_DIR})
  message(STATUS "Downloading DiFfRG version ${TRY_DiFfRG_VERSION}...")
  execute_process(
    COMMAND
      bash -c
      "git clone --depth 1 --branch ${TRY_DiFfRG_VERSION} https://github.com/satfra/DiFfRG.git ${DiFfRG_SOURCE_DIR}"
    OUTPUT_QUIET)
else()
  message(STATUS "DiFfRG source directory already exists, skipping download.")
  # still, checkout the branch
  execute_process(
    COMMAND bash -c "git -C ${DiFfRG_SOURCE_DIR} checkout ${TRY_DiFfRG_VERSION}"
    OUTPUT_QUIET)
endif()

message(STATUS "Configuring DiFfRG...")
execute_process(
  COMMAND
    bash -c
    "cmake -S ${DiFfRG_SOURCE_DIR} -B ${DiFfRG_BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${DiFfRG_INSTALL_DIR} -DCMAKE_BUILD_TYPE=Release"
)

message(STATUS "Building DiFfRG...")
execute_process(
  COMMAND
    bash -c
    "cmake --build ${DiFfRG_BUILD_DIR} --config Release --target all -- -j ${PARALLEL_JOBS} | tee ${DiFfRG_BUILD_DIR}/DiFfRG.log"
  )
message(STATUS "DiFfRG installed to ${DiFfRG_INSTALL_DIR}")

# ##############################################################################
# Have fun!
# ##############################################################################

find_package(DiFfRG HINTS ${DiFfRG_INSTALL_DIR} REQUIRED)
