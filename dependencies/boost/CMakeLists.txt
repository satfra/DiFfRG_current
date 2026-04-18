cmake_minimum_required(VERSION 3.20)

project(boost_dep NONE)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

# boost
set(TRY_BOOST_VERSION "1.81.0")
include(ProcessorCount)
ProcessorCount(PARALLEL_JOBS)
set(BOOST_SHA256
    "205666dea9f6a7cfed87c7a6dfbeb52a2c1b9de55712c9c1a87735d7181452b6")
set(BOOST_TARBALL_URLS
    "https://archives.boost.io/release/${TRY_BOOST_VERSION}/source/boost_1_81_0.tar.gz"
    "https://sourceforge.net/projects/boost/files/boost/${TRY_BOOST_VERSION}/boost_1_81_0.tar.gz/download"
)

# Cache the tarball outside the build tree so a fresh build/ dir does not
# re-download it. Honors XDG_CACHE_HOME when set; falls back to $HOME/.cache
# (which works on both Linux and macOS).
if(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
  set(_diffrg_cache_root "$ENV{XDG_CACHE_HOME}/DiFfRG")
else()
  set(_diffrg_cache_root "$ENV{HOME}/.cache/DiFfRG")
endif()
file(MAKE_DIRECTORY "${_diffrg_cache_root}")

set(BOOST_TARBALL "${_diffrg_cache_root}/boost_1_81_0.tar.gz")
set(BOOST_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/boost_1_81_0")

# Reject a cached tarball whose hash does not match — otherwise a prior
# truncated / wrong download silently poisons every subsequent build.
if(EXISTS ${BOOST_TARBALL})
  file(SHA256 ${BOOST_TARBALL} _cached_sha256)
  if(NOT _cached_sha256 STREQUAL ${BOOST_SHA256})
    message(
      STATUS
        "Cached Boost tarball at ${BOOST_TARBALL} has wrong SHA256 (${_cached_sha256}); removing."
    )
    file(REMOVE ${BOOST_TARBALL})
  endif()
endif()

if(NOT EXISTS ${BOOST_TARBALL})
  set(_boost_download_ok FALSE)
  foreach(_url IN LISTS BOOST_TARBALL_URLS)
    message(STATUS "Downloading Boost ${TRY_BOOST_VERSION} from ${_url} ...")
    file(
      DOWNLOAD ${_url} ${BOOST_TARBALL}
      EXPECTED_HASH SHA256=${BOOST_SHA256}
      INACTIVITY_TIMEOUT 30
      SHOW_PROGRESS
      STATUS _download_status
      LOG _download_log)
    list(GET _download_status 0 _download_code)
    if(_download_code EQUAL 0 AND EXISTS ${BOOST_TARBALL})
      file(SHA256 ${BOOST_TARBALL} _downloaded_sha256)
      if(_downloaded_sha256 STREQUAL ${BOOST_SHA256})
        set(_boost_download_ok TRUE)
        break()
      endif()
    endif()
    file(REMOVE ${BOOST_TARBALL})
    message(
      WARNING
        "Boost download from ${_url} failed or returned wrong content (${_download_status}); trying next mirror."
    )
  endforeach()
  if(NOT _boost_download_ok)
    message(
      FATAL_ERROR
        "All Boost tarball mirrors failed.\nLast download log:\n${_download_log}"
    )
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
