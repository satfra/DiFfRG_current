cmake_minimum_required(VERSION 3.27.0)

project(ONfiniteT LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ##############################################################################
# Setup Application
# ##############################################################################

if(NOT DEFINED DiFfRG_DIR)
  set(DiFfRG_DIR /opt/DiFfRG)
endif()
# find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR})
find_package(DiFfRG REQUIRED HINTS /opt/DiFfRG/)

add_subdirectory(flows)

add_executable(LDG LDG.cc ${flow_sources})
target_link_libraries(LDG PRIVATE DiFfRG::DiFfRG)
# setup_target(LDG)

# add_executable(CG CG.cc ${flow_sources}) setup_application(CG)
# target_link_libraries(CG DiFfRG::DiFfRG)

# add_executable(dDG dDG.cc ${flow_sources}) setup_application(dDG)
# target_link_libraries(dDG DiFfRG::DiFfRG)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
