# add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)

# boost_json.cc includes boost/json/src.hpp (the compiled boost::json source).
# It must NOT be compiled with CUDA offloading, otherwise the static member
# key_value_pair::empty_ ends up in device code and causes nvlink duplicate
# symbol errors when libDiFfRG.a appears more than once on the link line.
add_library(DiFfRG_boost_json STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/boost_json.cc)
set_target_properties(DiFfRG_boost_json PROPERTIES
    LANGUAGE CXX
    LINKER_LANGUAGE CXX
    CUDA_SEPARABLE_COMPILATION OFF)
target_include_directories(DiFfRG_boost_json SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})

set(DiFfRG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/configuration_helper.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/csv_reader.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/external_data_interpolator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/init.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/interpolation.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/json.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/eigen.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/mpi.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/polynomials.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/csv_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/data_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/fe_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/hdf5_output.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/data/hdf5_input.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/discretization/mesh/rectangular_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/model/model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/solver/kinsol.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/solver/newton.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/boost_rk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/boost_abm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/explicit_euler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/implicit_euler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/rk.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_arkode.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_boost_rk.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_boost_abm.cc
    # ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/sundials_ida_arkode.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/timestepping/trbdf2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/integration/abstract_integrator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/integration/distribution.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/physics/flow_equations.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/matsubara.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/quadrature_provider.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/common/quadrature/quadrature.cc)

add_library(DiFfRG STATIC ${DiFfRG_SOURCES})

target_link_libraries(DiFfRG PUBLIC DiFfRG_boost_json)

setup_target(DiFfRG)

find_package(HDF5 REQUIRED)
target_compile_definitions(DiFfRG PUBLIC H5CPP)
target_link_libraries(DiFfRG PUBLIC hdf5-static)
target_include_directories(DiFfRG PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>)
message(STATUS "HDF5 (static) support enabled")

target_include_directories(
  DiFfRG INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                   $<INSTALL_INTERFACE:include>)
add_library(DiFfRG::DiFfRG ALIAS DiFfRG)
set_property(TARGET DiFfRG PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

include(GNUInstallDirs)
install(
  TARGETS DiFfRG DiFfRG_boost_json
  EXPORT DiFfRGTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
