cmake_minimum_required(VERSION 3.20)

project(petsc_dep)

# petsc
set(TRY_PETSC_VERSION "3.23.1")
include(ProcessorCount)
ProcessorCount(PARALLEL_JOBS)

message(STATUS "Downloading PETSc version ${TRY_PETSC_VERSION}...")
execute_process(
  COMMAND
    bash -c
    "git clone https://gitlab.com/petsc/petsc.git --depth 1 --branch v${TRY_PETSC_VERSION}  2>&1 | tee ${CMAKE_CURRENT_BINARY_DIR}/petsc.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  OUTPUT_QUIET)

message(STATUS "Configuring PETSc...")
# config OpenMP
if(NOT DEFINED HAVE_OPENMP)
  set(HAVE_OPENMP
      OFF
      CACHE BOOL "Enable OpenMP support")
endif()
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
# if you're on MacOS, we need to turn off OpenMP support
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(HAVE_OPENMP FALSE)
endif()
if(HAVE_OPENMP)
  message(STATUS "Building PETSc with OpenMP support")
  set(PETSC_OPTIONS "${PETSC_OPTIONS} -with-openmp")
endif()

execute_process(
  COMMAND
    bash -c
    "
    ./configure --COPTFLAGS=\"-O3 -march=native -ffp-contract=fast\" \
    --CXXOPTFLAGS=\"-O3 -march=native -ffp-contract=fast\"  \
    --FOPTFLAGS=\"-O3 -march=native -ffp-contract=fast\" \
    --with-blas-lib=${OpenBLAS_LIBRARY} \
    --with-lapack-lib=${OpenBLAS_LIBRARY} \
    --prefix=${CMAKE_INSTALL_PREFIX} --with-clean --with-debugging=0 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/petsc
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to configure PETSc. Check ${CMAKE_CURRENT_BINARY_DIR}/petsc.log for details."
  )
endif()

message(STATUS "Building PETSc...")
execute_process(
  COMMAND
    bash -c
    "make -j ${PARALLEL_JOBS} 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log; \
    make install -j ${PARALLEL_JOBS} 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/petsc
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to build PETSc. Check ${CMAKE_CURRENT_BINARY_DIR}/petsc.log for details."
  )
endif()
