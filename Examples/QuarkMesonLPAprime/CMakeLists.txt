cmake_minimum_required(VERSION 3.26.4)

project(QuarkMesonLPAprime)

find_package(DiFfRG REQUIRED HINTS /opt/DiFfRG)

add_subdirectory(flows)
add_executable(QuarkMeson QM.cc ${flow_sources})
target_link_libraries(QuarkMeson DiFfRG::DiFfRG)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
