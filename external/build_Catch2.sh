#!/bin/bash

LIBRARY_NAME="Catch2"

source ./build_scripts/populate_paths.sh
source ./build_scripts/parse_flags.sh
source ./build_scripts/cleanup_build_if_asked.sh
source ./build_scripts/setup_folders.sh

cd $BUILD_PATH

cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
      -S ${SOURCE_PATH} \
      2>&1 | tee $CMAKE_LOG_FILE

make -j $THREADS 2>&1 | tee $MAKE_LOG_FILE
make -j $THREADS install
