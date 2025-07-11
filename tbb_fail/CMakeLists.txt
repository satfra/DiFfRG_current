cmake_minimum_required(VERSION 3.10)

project(testing)

find_package(TBB REQUIRED HINTS $ENV{HOME}/.local/share/DiFfRG/bundled)
message(STATUS "TBB package found at ${TBB_DIR}")

find_package(DiFfRG REQUIRED HINTS $ENV{HOME}/.local/share/DiFfRG)

find_package(deal.II 9.4.2 QUIET REQUIRED HINTS ${BUNDLED_DIR})
deal_ii_initialize_cached_variables()
message(STATUS "Found deal.II in  ${deal.II_DIR}")
function(setup_dealii TARGET)
  target_link_libraries(${TARGET} PUBLIC deal_II)
  target_include_directories(${TARGET} SYSTEM PUBLIC ${DEAL_II_INCLUDE_DIRS})

  # if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # set(_build "DEBUG") elseif(CMAKE_BUILD_TYPE STREQUAL "Release") set(_build
  # "RELEASE") endif()

  # set(_cflags "${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_${_build}}") remove
  # c++20 flag and O2 flag - CMake adds them automatically and we thus avoid the
  # nvcc_wrapper warnings string(REPLACE "-std=c++20" "" _cflags ${_cflags})
  # string(REPLACE "-O2" "" _cflags ${_cflags}) separate_arguments(_cflags)
  # target_compile_options(${TARGET} PUBLIC
  # $<$<COMPILE_LANGUAGE:CXX>:${_cflags}>)

  # set(_lflags "${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_${_build}}")
  # separate_arguments(_lflags) target_link_options(${TARGET} PUBLIC
  # $<$<COMPILE_LANGUAGE:CXX>:${_lflags}>)
endfunction()

add_executable(justTBB main.cc)
target_link_libraries(justTBB TBB::tbb)

add_executable(withDiFfRG main.cc)
target_link_libraries(withDiFfRG DiFfRG::DiFfRG)

add_executable(withdealii main.cc)
target_link_libraries(withdealii PUBLIC TBB::tbb)
setup_dealii(withdealii)
