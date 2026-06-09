set(LPA_QM_flows_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/etaPhi/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaPhi/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaPhi/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaQ/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaQ/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/etaQ/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hPhi0/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hPhi0/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hPhi0/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/constructor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/CT_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/V/src/AD_get.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/flows.cc)

add_library(LPA_QM_flows STATIC ${LPA_QM_flows_SOURCES})
add_library(LPA_QM_flows_nowarn INTERFACE)
target_compile_options(LPA_QM_flows_nowarn INTERFACE -Wno-unused-parameter -Wno-unused-variable)
target_link_libraries(LPA_QM_flows DiFfRG::DiFfRG LPA_QM_flows_nowarn)

set_target_properties(
  LPA_QM_flows
  PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_MODE GROUP)

foreach(src IN LISTS LPA_QM_flows_SOURCES)
  if(src MATCHES "/src/CT_map_[0-9]+\.cc$")
    set_source_files_properties("${src}" PROPERTIES UNITY_GROUP "all_ct_map")
  else()
    set_source_files_properties("${src}" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
  endif()
endforeach()
