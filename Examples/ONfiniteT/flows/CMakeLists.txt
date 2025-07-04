set(ONFiniteTFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(ONFiniteTFlows STATIC ${ONFiniteTFlows_SOURCES})
target_link_libraries(ONFiniteTFlows DiFfRG::DiFfRG)
target_compile_options(
  ONFiniteTFlows PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wno-unused-parameter
                         -Wno-unused-variable >)
