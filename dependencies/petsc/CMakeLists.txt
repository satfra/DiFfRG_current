cmake_minimum_required(VERSION 3.27)

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

message(STATUS "Configuring petsc...")
execute_process(
  COMMAND
    bash -c
    "./configure --with-debugging=0 --prefix=${CMAKE_INSTALL_PREFIX} 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/petsc)

add_custom_target(
  build_petsc
  COMMAND
    bash -c
    "echo 'Building PETSc...'; \
    make -j ${PARALLEL_JOBS} 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log; \
    make install -j ${PARALLEL_JOBS} 2>&1 | tee -a ${CMAKE_CURRENT_BINARY_DIR}/petsc.log"
  VERBATIM
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/petsc)

install(
  CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target build_petsc)"
)
