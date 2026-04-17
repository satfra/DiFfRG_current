cmake_minimum_required(VERSION 3.20)

project(boost_dep NONE)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

# boost
set(TRY_BOOST_VERSION "1.81.0")
include(ProcessorCount)
ProcessorCount(PARALLEL_JOBS)
set(BOOST_TARBALL_URL
    "https://sourceforge.net/projects/boost/files/boost/1.81.0/boost_1_81_0.tar.gz/download"
)
set(BOOST_TARBALL
    "${CMAKE_CURRENT_BINARY_DIR}/boost-${TRY_BOOST_VERSION}.tar.gz")
set(BOOST_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/boost_1_81_0")

if(NOT EXISTS ${BOOST_TARBALL})
  message(STATUS "Downloading Boost ${TRY_BOOST_VERSION} tarball...")
  file(
    DOWNLOAD ${BOOST_TARBALL_URL} ${BOOST_TARBALL}
    EXPECTED_HASH
      SHA256=205666dea9f6a7cfed87c7a6dfbeb52a2c1b9de55712c9c1a87735d7181452b6
    SHOW_PROGRESS
    STATUS _download_status
    LOG _download_log)
  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    file(REMOVE ${BOOST_TARBALL})
    message(
      FATAL_ERROR
        "Boost tarball download failed: ${_download_status}\n${_download_log}")
  endif()
endif()

if(NOT EXISTS ${BOOST_SOURCE_DIR})
  message(STATUS "Extracting Boost tarball...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${BOOST_TARBALL}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE _extract_result)
  if(NOT _extract_result EQUAL 0)
    message(FATAL_ERROR "Boost tarball extraction failed.")
  endif()
endif()

message(STATUS "Patching boost...")
execute_process(
  COMMAND
    bash -c
    "patch --forward ${BOOST_SOURCE_DIR}/libs/math/include/boost/math/cstdfloat/cstdfloat_limits.hpp < ${CMAKE_CURRENT_SOURCE_DIR}/cstdfloat_limits.hpp.diff"
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/02_patch.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/02_patch.log
  RESULT_VARIABLE _boost_patch_result)
if(NOT _boost_patch_result EQUAL 0)
  # patch --forward exits 1 if already applied; only warn, don't abort
  message(
    STATUS
      "Boost patch returned non-zero (may already be applied). See log: ${CMAKE_CURRENT_BINARY_DIR}/02_patch.log"
  )
endif()

if(CCACHE_PROGRAM)
  set(BOOST_COMPILER_ARGS
      "CC='${CCACHE_PROGRAM} ${CMAKE_C_COMPILER}' CXX='${CCACHE_PROGRAM} ${CMAKE_CXX_COMPILER}'"
  )
else()
  set(BOOST_COMPILER_ARGS
      "CC='${CMAKE_C_COMPILER}' CXX='${CMAKE_CXX_COMPILER}'")
endif()

add_custom_target(
  build_boost_boostrap
  COMMAND
    bash -c
    "${BOOST_COMPILER_ARGS} ./bootstrap.sh --prefix=${CMAKE_INSTALL_PREFIX} &> ${CMAKE_CURRENT_BINARY_DIR}/03_bootstrap.log"
  WORKING_DIRECTORY ${BOOST_SOURCE_DIR}
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
  WORKING_DIRECTORY ${BOOST_SOURCE_DIR}
  DEPENDS build_boost_boostrap)

install(
  CODE [[
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target build_boost_b2
      RESULT_VARIABLE boost_build_result
      COMMAND_ECHO STDOUT)

    if(NOT boost_build_result EQUAL 0)
      message(
        FATAL_ERROR
          "Boost installation failed while running the nested build target build_boost_b2."
      )
    endif()

    if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/include/boost")
      message(
        FATAL_ERROR
          "Boost installation did not produce ${CMAKE_INSTALL_PREFIX}/include/boost."
      )
    endif()

    if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/lib")
      message(
        FATAL_ERROR
          "Boost installation did not produce ${CMAKE_INSTALL_PREFIX}/lib."
      )
    endif()
  ]]
)
