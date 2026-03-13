cmake_minimum_required(VERSION 3.20)

project(boost_dep NONE)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

# boost
set(TRY_BOOST_VERSION "1.81.0")
include(ProcessorCount)
ProcessorCount(PARALLEL_JOBS)

# we need python
find_package(Python REQUIRED)

message(STATUS "Downloading Boost version ${TRY_BOOST_VERSION}...")
execute_process(
  COMMAND
    bash -c
    "git clone --depth 1 --branch boost-${TRY_BOOST_VERSION} https://github.com/boostorg/boost.git"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  OUTPUT_QUIET
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/00_clone.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/00_clone.log
  RESULT_VARIABLE _boost_clone_result)
if(NOT _boost_clone_result EQUAL 0)
  message(FATAL_ERROR "Boost clone failed. See log: ${CMAKE_CURRENT_BINARY_DIR}/00_clone.log")
endif()

message(STATUS "Downloading Boost libraries... (this may take a while)")
execute_process(
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/download_modules.sh
          ${Python_EXECUTABLE}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/
  OUTPUT_QUIET
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/01_download_modules.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/01_download_modules.log
  RESULT_VARIABLE _boost_modules_result)
if(NOT _boost_modules_result EQUAL 0)
  message(FATAL_ERROR "Boost module download failed. See log: ${CMAKE_CURRENT_BINARY_DIR}/01_download_modules.log")
endif()

message(STATUS "Patching boost...")
execute_process(
  COMMAND
    bash -c
    "patch --forward ${CMAKE_CURRENT_BINARY_DIR}/boost/libs/math/include/boost/math/cstdfloat/cstdfloat_limits.hpp < ${CMAKE_CURRENT_SOURCE_DIR}/cstdfloat_limits.hpp.diff"
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/02_patch.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/02_patch.log
  RESULT_VARIABLE _boost_patch_result)
if(NOT _boost_patch_result EQUAL 0)
  # patch --forward exits 1 if already applied; only warn, don't abort
  message(STATUS "Boost patch returned non-zero (may already be applied). See log: ${CMAKE_CURRENT_BINARY_DIR}/02_patch.log")
endif()

set(BOOST_COMPILER_ARGS
    "CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} FC=${CMAKE_Fortran_COMPILER}"
)

add_custom_target(
  build_boost_boostrap
  COMMAND
    bash -c
    "${BOOST_COMPILER_ARGS} ./bootstrap.sh --prefix=${CMAKE_INSTALL_PREFIX} &> ${CMAKE_CURRENT_BINARY_DIR}/03_bootstrap.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/
  VERBATIM)

add_custom_target(
  build_boost_b2
  COMMAND
    bash -c
    "echo 'Building Boost...'; ${BOOST_COMPILER_ARGS} ./b2 --build-dir=${CMAKE_CURRENT_BINARY_DIR}/boost-build \
        --prefix=${CMAKE_INSTALL_PREFIX} \
        --with-headers \
        --with-iostreams \
        --with-json \
        --with-math \
        --with-serialization \
        --with-system \
        --with-thread \
        -j ${PARALLEL_JOBS} \
        cxxflags=\"-fPIC -O3 -ffat-lto-objects -std=c++20\" \
        install &> ${CMAKE_CURRENT_BINARY_DIR}/04_build.log"
  VERBATIM
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/
  DEPENDS build_boost_boostrap)

install(
  CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target build_boost_b2)"
)
