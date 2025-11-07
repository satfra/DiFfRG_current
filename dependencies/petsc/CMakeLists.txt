cmake_minimum_required(VERSION 3.20)

project(petsc_dep NONE)

# petsc
set(TRY_PETSC_VERSION "3.23.1")
include(ProcessorCount)
ProcessorCount(PARALLEL_JOBS)

message(STATUS "Downloading PETSc version ${TRY_PETSC_VERSION}...")
execute_process(
  COMMAND
    bash -c
    "git clone https://gitlab.com/petsc/petsc.git --depth 1 --branch v${TRY_PETSC_VERSION} &> ${CMAKE_CURRENT_BINARY_DIR}/00_clone.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  OUTPUT_QUIET)

message(STATUS "Configuring PETSc...")
# config MPI
if(NOT DEFINED HAVE_MPI)
  set(HAVE_MPI
      OFF
      CACHE BOOL "Enable MPI support")
endif()
# setup PETSc options
if(HAVE_MPI)
  message(STATUS "Building PETSc with MPI support")
  set(PETSC_OPTIONS
      "–download-mumps –download-scalapack –download-parmetis –download-metis –download-ptscotch –download-hwloc -download-bison -with-mpi=1"
  )
else()
  message(STATUS "Building PETSc without MPI support")
  set(PETSC_OPTIONS "-with-mpi=0")
endif()

execute_process(
  COMMAND
    bash -c
    "
    ./configure --COPTFLAGS=\"-O3 -march=native -ffp-contract=fast\" \
    --with-mpi-dir=/opt/homebrew/opt/open-mpi \
    --CXXOPTFLAGS=\"-O3 -march=native -ffp-contract=fast\"  \
    --FOPTFLAGS=\"-O3 -march=native -ffp-contract=fast\" \
    --with-blas-lib=${OpenBLAS_LIBRARY} \
    --with-lapack-lib=${OpenBLAS_LIBRARY} \
    --prefix=${CMAKE_INSTALL_PREFIX} ${PETSC_OPTIONS} --with-clean --with-debugging=0 &> ${CMAKE_CURRENT_BINARY_DIR}/01_configure.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/petsc
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to configure PETSc. Check ${CMAKE_CURRENT_BINARY_DIR}/01_configure.log for details."
  )
endif()

message(STATUS "Building PETSc...")
execute_process(
  COMMAND
    bash -c
    "make -j ${PARALLEL_JOBS} &> ${CMAKE_CURRENT_BINARY_DIR}/02_build.log; \
    make install -j ${PARALLEL_JOBS} &>> ${CMAKE_CURRENT_BINARY_DIR}/02_build.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/petsc
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to build PETSc. Check ${CMAKE_CURRENT_BINARY_DIR}/02_build.log for details."
  )
endif()
