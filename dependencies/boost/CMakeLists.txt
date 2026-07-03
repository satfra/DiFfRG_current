cmake_minimum_required(VERSION 3.20)

project(boost_dep NONE)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

set(TRY_BOOST_VERSION "1.81.0")
set(TRY_BOOST_VERSION_NUMBER 108100)
set(BOOST_VENDOR_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/boost_1_81_0")
set(BOOST_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/boost_1_81_0")
set(BOOST_PREPARED_STAMP "${BOOST_SOURCE_DIR}/.diffrg_prepared")
set(BOOST_PATCHED_STAMP "${BOOST_SOURCE_DIR}/.diffrg_patched")

include(ProcessorCount)
ProcessorCount(PARALLEL_JOBS)
if(PARALLEL_JOBS EQUAL 0)
  set(PARALLEL_JOBS 1)
endif()

if(NOT EXISTS "${BOOST_VENDOR_SOURCE_DIR}/bootstrap.sh"
   OR NOT EXISTS "${BOOST_VENDOR_SOURCE_DIR}/boost/version.hpp")
  message(
    FATAL_ERROR
      "Vendored Boost ${TRY_BOOST_VERSION} source is missing.\n"
      "Expected ${BOOST_VENDOR_SOURCE_DIR} to contain bootstrap.sh and boost/version.hpp."
  )
endif()

file(STRINGS "${BOOST_VENDOR_SOURCE_DIR}/boost/version.hpp" _boost_version_line
     REGEX "^#define BOOST_VERSION ")
string(REGEX REPLACE ".*BOOST_VERSION[ \t]+([0-9]+).*" "\\1"
                     _boost_version_number "${_boost_version_line}")
if(NOT _boost_version_number STREQUAL "${TRY_BOOST_VERSION_NUMBER}")
  message(
    FATAL_ERROR
      "Vendored Boost version mismatch: expected ${TRY_BOOST_VERSION} "
      "(${TRY_BOOST_VERSION_NUMBER}), found ${_boost_version_number} in "
      "${BOOST_VENDOR_SOURCE_DIR}/boost/version.hpp.")
endif()

if(CCACHE_PROGRAM)
  set(BOOST_CC_ENV "${CCACHE_PROGRAM} ${CMAKE_C_COMPILER}")
  set(BOOST_CXX_ENV "${CCACHE_PROGRAM} ${CMAKE_CXX_COMPILER}")
else()
  set(BOOST_CC_ENV "${CMAKE_C_COMPILER}")
  set(BOOST_CXX_ENV "${CMAKE_CXX_COMPILER}")
endif()

add_custom_command(
  OUTPUT "${BOOST_PREPARED_STAMP}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${BOOST_SOURCE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${BOOST_VENDOR_SOURCE_DIR}"
          "${BOOST_SOURCE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E touch "${BOOST_PREPARED_STAMP}"
  DEPENDS "${BOOST_VENDOR_SOURCE_DIR}/boost/version.hpp"
          "${BOOST_VENDOR_SOURCE_DIR}/bootstrap.sh"
  COMMENT "Preparing build-local Boost ${TRY_BOOST_VERSION} source copy"
  VERBATIM)

add_custom_command(
  OUTPUT "${BOOST_PATCHED_STAMP}"
  COMMAND
    bash -c
    "if grep -q '__GNUC__ < 14' '${BOOST_SOURCE_DIR}/boost/math/cstdfloat/cstdfloat_limits.hpp'; then \
       echo 'Boost cstdfloat patch already applied.'; \
     else \
       patch --batch --forward '${BOOST_SOURCE_DIR}/boost/math/cstdfloat/cstdfloat_limits.hpp' < '${CMAKE_CURRENT_SOURCE_DIR}/cstdfloat_limits.hpp.diff'; \
     fi"
  COMMAND ${CMAKE_COMMAND} -E touch "${BOOST_PATCHED_STAMP}"
  DEPENDS "${BOOST_PREPARED_STAMP}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cstdfloat_limits.hpp.diff"
  COMMENT "Patching build-local Boost ${TRY_BOOST_VERSION} source copy"
  VERBATIM)

add_custom_target(
  build_boost_boostrap
  COMMAND
    bash -c
    "CC=\"${BOOST_CC_ENV}\" CXX=\"${BOOST_CXX_ENV}\" ./bootstrap.sh --prefix='${CMAKE_INSTALL_PREFIX}' > '${CMAKE_CURRENT_BINARY_DIR}/03_bootstrap.log' 2>&1"
  WORKING_DIRECTORY "${BOOST_SOURCE_DIR}"
  DEPENDS "${BOOST_PATCHED_STAMP}"
  VERBATIM)

add_custom_target(
  build_boost_b2
  COMMAND
    bash -c
    "echo 'Building Boost...'; CC=\"${BOOST_CC_ENV}\" CXX=\"${BOOST_CXX_ENV}\" ./b2 --build-dir='${CMAKE_CURRENT_BINARY_DIR}/boost-build' \
        --prefix='${CMAKE_INSTALL_PREFIX}' \
        --with-headers \
        --with-iostreams \
        --with-json \
        --with-math \
        --with-serialization \
        --with-system \
        --with-thread \
        -j ${PARALLEL_JOBS} \
        cxxflags=\"-fPIC -O3 -ffat-lto-objects -std=c++20\" \
        install > '${CMAKE_CURRENT_BINARY_DIR}/04_build.log' 2>&1"
  VERBATIM
  WORKING_DIRECTORY "${BOOST_SOURCE_DIR}"
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
  ]])
