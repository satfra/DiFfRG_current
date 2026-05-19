cmake_minimum_required(VERSION 3.27.0)

project(ONfiniteT_KT)

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

# Reuse the flows library that already lives next to Examples/ONfiniteT.
# A unique binary-dir alias is needed because CMake forbids adding the same
# source dir twice from one tree.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ONfiniteT/flows
                 ${CMAKE_CURRENT_BINARY_DIR}/ONfiniteT_flows)

add_executable(KT KT.cc)
target_link_libraries(KT PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)
# Make sure model.hh and the flows lib are visible.
target_include_directories(KT PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ONfiniteT)

# Second executable: sigma-coordinate model, sharing the same flows lib.
add_executable(KT_sigma KT_sigma.cc)
target_link_libraries(KT_sigma PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)
target_include_directories(KT_sigma PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ONfiniteT)

# Symlink the parameter files into the build directory.
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink"
  "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
  "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink"
  "${CMAKE_CURRENT_SOURCE_DIR}/parameter_sigma.json"
  "${CMAKE_CURRENT_BINARY_DIR}/parameter_sigma.json")
