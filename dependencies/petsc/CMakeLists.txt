cmake_minimum_required(VERSION 3.26.5)

project(petsc_dep)

# petsc
set(TRY_PETSC_VERSION "3.23.1")
set(PARALLEL_JOBS 8)

message(STATUS "Downloading PETSc version ${TRY_PETSC_VERSION}...")
execute_process(
  COMMAND
    bash -c
    "git clone https://gitlab.com/petsc/petsc.git --depth 1 --branch v${TRY_PETSC_VERSION}  2>&1 | tee ${CMAKE_CURRENT_BINARY_DIR}/petsc.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  OUTPUT_QUIET)

message(STATUS "Configuring PETSc...")
if(NOT DEFINED HAVE_OPENMP)
  set(HAVE_OPENMP
      OFF
      CACHE BOOL "Enable OpenMP support")
endif()
set(PETSC_OPTIONS
    "–download-mumps –download-scalapack –download-parmetis –download-metis –download-ptscotch –download-hwloc"
)
if(HAVE_OPENMP)
  message(STATUS "Building PETSc with OpenMP support")
  set(PETSC_OPTIONS "${PETSC_OPTIONS} -with-openmp")
endif()

execute_process(
  COMMAND
    bash -c
    "./configure --with-debugging=0 ${PETSC_OPTIONS} --prefix=${CMAKE_INSTALL_PREFIX} --with-clean 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/petsc)

message(STATUS "Building PETSc...")
execute_process(
  COMMAND
    bash -c
    "make -j ${PARALLEL_JOBS} 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log; \
    make install -j ${PARALLEL_JOBS} 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/petsc)
