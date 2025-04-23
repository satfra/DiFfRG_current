#!/bin/bash

LIBRARY_NAME="dealii"
SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

source $SCRIPT_PATH/build_scripts/parse_flags.sh
source $SCRIPT_PATH/build_scripts/populate_paths.sh
source $SCRIPT_PATH/build_scripts/cleanup_build_if_asked.sh
source $SCRIPT_PATH/build_scripts/setup_folders.sh
source $SCRIPT_PATH/../config

cuda_path=$(command -v nvcc)
cuda_tail="${cuda_path#/*/*/}"
export CUDA_ROOT="${path%/$cuda_tail}/"

cd $BUILD_PATH

cmake -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_COMPONENT_EXAMPLES=OFF \
  -DDEAL_II_COMPONENT_DOCUMENTATION=OFF \
  -DDEAL_II_WITH_MPI=OFF \
  -DDEAL_II_WITH_UMFPACK=ON \
  -DDEAL_II_WITH_TASKFLOW=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DDEAL_II_WITH_VTK=OFF \
  -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -fPIC" \
  -DCMAKE_CXX_COMPILER="${TOP_INSTALL_PATH}/kokkos_install/bin/nvcc_wrapper" \
  -DTBB_DIR=${TOP_INSTALL_PATH}/oneTBB_install \
  -DKOKKOS_DIR=${TOP_INSTALL_PATH}/kokkos_install \
  -DSUNDIALS_DIR=${TOP_INSTALL_PATH}/sundials_install \
  -DBOOST_DIR=${TOP_INSTALL_PATH}/boost_install \
  -${DEAL_II_CMAKE} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -S ${SOURCE_PATH} \
  &>$CMAKE_LOG_FILE

make -j $THREADS &>$MAKE_LOG_FILE
$SuperUser make -j $THREADS install >>$MAKE_LOG_FILE 2>&1
