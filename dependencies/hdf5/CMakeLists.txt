cmake_minimum_required(VERSION 3.20)

project(hdf_dep)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

# if MPI is not set, set it to OFF
if(NOT DEFINED MPI)
  set(MPI
      OFF
      CACHE BOOL "Enable MPI support")
endif()

# Set up external dependencies

# download HDF5, if it is not already downloaded
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/hdf5.tar.gz)
  message(STATUS "Downloading hdf5...")
  file(DOWNLOAD
       https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5.tar.gz
       ${CMAKE_CURRENT_BINARY_DIR}/hdf5.tar.gz)
else()
  message(STATUS "hdf5 already downloaded.")
endif()

# extract hdf5, if it is not already extracted
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/hdf5-2.0.0/)
  message(STATUS "Extracting hdf5...")
  execute_process(COMMAND tar -xzf hdf5.tar.gz
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
  message(STATUS "hdf5 already extracted.")
endif()

# configure hdf5
execute_process(
  COMMAND
    cmake -S hdf5-2.0.0 -B ./build -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -D
    CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} -DCMAKE_CXX_STANDARD=20
    -DHDF5_EXTERNALLY_CONFIGURED=0 -DHDF5_ENABLE_PARALLEL=${MPI}
    -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_HL_LIB=ON
    -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
  # Check if the command was successful
  RESULT_VARIABLE HDF5_CONFIGURE_RESULT)
if(NOT HDF5_CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to configure hdf5.")
endif()

# briefly get the number of processors for parallel build
include(ProcessorCount)
ProcessorCount(PROCESSOR_COUNT)

# build hdf5
execute_process(
  COMMAND make -j ${PROCESSOR_COUNT}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build
  # Check if the command was successful
  RESULT_VARIABLE HDF5_BUILD_RESULT)
if(NOT HDF5_BUILD_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to build hdf5.")
endif()

# install hdf5
execute_process(
  COMMAND make install
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build
  # Check if the command was successful
  RESULT_VARIABLE HDF5_INSTALL_RESULT)
if(NOT HDF5_INSTALL_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to install hdf5.")
endif()
