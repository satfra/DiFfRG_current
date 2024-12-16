#!/bin/bash

LIBRARY_NAME="thread-pool"

source ./build_scripts/populate_paths.sh
source ./build_scripts/parse_flags.sh
source ./build_scripts/cleanup_build_if_asked.sh
source ./build_scripts/setup_folders.sh

cp -r ${SOURCE_PATH}/include ${INSTALL_PATH}/include
