#!/bin/bash

LIBRARY_NAME="kokkos"
SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

source $SCRIPT_PATH/build_scripts/parse_flags.sh
source $SCRIPT_PATH/build_scripts/populate_paths.sh
source $SCRIPT_PATH/build_scripts/cleanup_build_if_asked.sh
source $SCRIPT_PATH/build_scripts/setup_folders.sh
source $SCRIPT_PATH/../config

cd $BUILD_PATH

cmake \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
  -DKokkos_ARCH_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -fPIC" \
  -DCMAKE_EXE_LINKER_FLAGS="${EXE_LINKER_FLAGS}" \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -S ${SOURCE_PATH} \
  &>$CMAKE_LOG_FILE

make -j $THREADS &>$MAKE_LOG_FILE
$SuperUser make -j $THREADS install >>$MAKE_LOG_FILE 2>&1
