cmake_minimum_required(VERSION 3.27)

project(boost_dep)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

# boost
set(TRY_BOOST_VERSION "1.81.0")
set(PARALLEL_JOBS 8)

message(STATUS "Downloading Boost version ${TRY_BOOST_VERSION}...")
execute_process(
  COMMAND
    bash -c
    "git clone --depth 1 --branch boost-${TRY_BOOST_VERSION} https://github.com/boostorg/boost.git &> ${CMAKE_CURRENT_BINARY_DIR}/clone_boost.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  OUTPUT_QUIET)
message(STATUS "Downloading Boost libraries... (this may take a while)")
execute_process(
  COMMAND
    bash -c
    "git submodule update --init --depth 1 --recursive --jobs ${PARALLEL_JOBS} &> ${CMAKE_CURRENT_BINARY_DIR}/boost.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/
  OUTPUT_QUIET)

message(STATUS "Patching boost...")
execute_process(
  COMMAND
    bash -c
    "patch ${CMAKE_CURRENT_BINARY_DIR}/boost/libs/math/include/boost/math/cstdfloat/cstdfloat_limits.hpp < ${CMAKE_CURRENT_SOURCE_DIR}/cstdfloat_limits.hpp.diff"
)

add_custom_target(
  build_boost_boostrap
  COMMAND bash -c "./bootstrap.sh --prefix=${CMAKE_INSTALL_PREFIX} &> /dev/null"
  VERBATIM
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/)
add_custom_target(
  build_boost_b2
  COMMAND
    bash -c
    "echo 'Building Boost...'; ./b2 --build-dir=${CMAKE_CURRENT_BINARY_DIR}/boost-build \
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
        install &>> ${CMAKE_CURRENT_BINARY_DIR}/boost.log"
  VERBATIM
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/
  DEPENDS build_boost_boostrap)

install(
  CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target build_boost_b2)"
)
