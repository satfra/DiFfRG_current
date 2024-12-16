#!/bin/bash

# ##############################################################################
# Script setup
# ##############################################################################

threads='1'
cuda_version=''
cuda=''
while getopts c:j: flag; do
    case "${flag}" in
    j) threads=${OPTARG} ;;
    c) cuda_version=${OPTARG} 
       cuda='-c' ;;
    esac
done
scriptpath="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

# if cuda_version is set, use the cuda base container
if [ -z "${cuda_version}" ]; then
    base="rockylinux:9"
else
    base="nvidia/cuda:${cuda_version}-devel-rockylinux9"
fi

echo
echo "    Docker build Threads: ${threads}"
echo "    CUDA base container: ${base}"
echo

# ##############################################################################
# Run docker build
# ##############################################################################

echo "   Building DiFfRG docker image..."
docker buildx build -t diffrg . --build-arg base=${base} --build-arg threads=${threads} --build-arg cuda=${cuda}
