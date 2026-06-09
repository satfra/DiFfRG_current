set(ONFiniteTFlows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(ONFiniteTFlows STATIC ${ONFiniteTFlows_SOURCES})
add_library(ONFiniteTFlows_nowarn INTERFACE)
target_compile_options(ONFiniteTFlows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(ONFiniteTFlows DiFfRG::DiFfRG ONFiniteTFlows_nowarn)

set_target_properties(
  ONFiniteTFlows
  PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_MODE GROUP)

foreach(src IN LISTS ONFiniteTFlows_SOURCES)
  if(src MATCHES "/src/CT_map_[0-9]+\.cc$")
    set_source_files_properties("${src}" PROPERTIES UNITY_GROUP "all_ct_map")
  else()
    set_source_files_properties("${src}" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
  endif()
endforeach()
