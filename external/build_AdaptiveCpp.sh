#!/bin/bash

LIBRARY_NAME="AdaptiveCpp"
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
  -DBOOST_ROOT=${TOP_INSTALL_PATH}/boost_install \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -DBoost_DEBUG=ON \
  -S ${SOURCE_PATH}
&>$CMAKE_LOG_FILE

make -j $THREADS &>$MAKE_LOG_FILE
$SuperUser make -j $THREADS install >>$MAKE_LOG_FILE 2>&1
