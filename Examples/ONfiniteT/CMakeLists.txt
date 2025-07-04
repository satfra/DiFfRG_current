cmake_minimum_required(VERSION 3.27.0)

project(ONfiniteT)

# ##############################################################################
# Setup Application
# ##############################################################################

find_package(DiFfRG REQUIRED HINTS $ENV{HOME}/.local/share/DiFfRG/)

add_subdirectory(flows)

add_executable(LDG LDG.cc)
target_link_libraries(LDG PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

add_executable(CG CG.cc)
target_link_libraries(CG PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

add_executable(dDG dDG.cc)
target_link_libraries(dDG PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
