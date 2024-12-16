#!/bin/bash

# ##############################################################################
# Script setup
# ##############################################################################

threads='1'
force='y'
while getopts j:f flag; do
    case "${flag}" in
    j) threads=${OPTARG} ;;
    f) force='y' ;;
    esac
done
scriptpath="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

# check if DiFfRG_build exists
if [ ! -d "${scriptpath}/DiFfRG_build" ]; then
    echo "DiFfRG_build directory does not exist. Please run build_DiFfRG.sh first."
    exit 1
fi

if [[ "$OSTYPE" =~ ^darwin ]]; then
    export OpenMP_ROOT=$(brew --prefix)/opt/libomp
fi

# ##############################################################################
# Build tests
# ##############################################################################

cd ${scriptpath}/DiFfRG_build
cmake -DDiFfRG_BUILD_TESTS=ON .
make -j${threads}

# ##############################################################################
# Run tests
# ##############################################################################

{ ctest | tee "${scriptpath}/logs/DiFfRG_tests.log"; } || { echo "Tests failed."; exit 1; }