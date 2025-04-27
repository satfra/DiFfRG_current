#!/bin/bash

LIBRARY_NAME="dealii"
SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

cuda_path=$(command -v nvcc)
cuda_tail="bin/nvcc"
export CUDA_ROOT="${cuda_path%/$cuda_tail}/"

source $SCRIPT_PATH/build_scripts/parse_flags.sh
source $SCRIPT_PATH/build_scripts/populate_paths.sh
source $SCRIPT_PATH/build_scripts/cleanup_build_if_asked.sh
source $SCRIPT_PATH/build_scripts/setup_folders.sh
source $SCRIPT_PATH/../config

cd $BUILD_PATH

cmake -DCMAKE_BUILD_TYPE=DebugRelease \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DDEAL_II_COMPONENT_EXAMPLES=OFF \
  -DDEAL_II_COMPONENT_DOCUMENTATION=OFF \
  -DDEAL_II_WITH_MPI=OFF \
  -DDEAL_II_WITH_UMFPACK=ON \
  -DDEAL_II_WITH_TASKFLOW=OFF \
  -DCMAKE_CXX_STANDARD=20 \
  -DTBB_DIR=${TOP_INSTALL_PATH}/oneTBB_install \
  -DSUNDIALS_DIR=${TOP_INSTALL_PATH}/sundials_install \
  -DDEAL_II_CXX_FLAGS="-std=c++20" \
  -DBOOST_DIR=${TOP_INSTALL_PATH}/boost_install \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -S ${SOURCE_PATH} \
  &>$CMAKE_LOG_FILE

make -j $THREADS &>$MAKE_LOG_FILE
$SuperUser make -j $THREADS install >>$MAKE_LOG_FILE 2>&1
# This directory is usually created by deal.ii, but if no bundled libraries are used, it is not created.
# We need it to be created so that dependent projects can use DiFfRG
$SuperUser mkdir -p ${INSTALL_PATH}/include/deal.II/bundled
