#!/bin/bash

LIBRARY_NAME="qmc"

source ./build_scripts/populate_paths.sh
source ./build_scripts/parse_flags.sh
source ./build_scripts/cleanup_build_if_asked.sh
source ./build_scripts/setup_folders.sh

cd $SOURCE_PATH

python3 ./make_singleheader.py
mv qmc.hpp ${INSTALL_PATH}/qmc.hpp
