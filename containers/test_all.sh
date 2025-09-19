#!/bin/bash

# The Base/ directory contains setups with neither MPI, nor CUDA.
# This script tries to build the DiFfRG docker images for all setups in the Base/ directory.

# get all files in Base/
base_images=$(ls Base/)
# get all files in MPI/
mpi_openmp_images=$(ls MPI/)
# get all files in MPI+CUDA/
mpi_openmp_cuda_images=$(ls MPI+CUDA/)
mkdir -p logs

for image in $mpi_openmp_cuda_images; do
    echo "   Building DiFfRG docker image for $image..."
    docker buildx build -t diffrg-$image -f MPI+CUDA/$image . --no-cache --progress=plain &>logs/$image.log
    if [ $? -ne 0 ]; then
        echo "   Error building $image. Check the log file $image.log for details."
    else
        echo "   Successfully built $image."
    fi
    echo "   Cleaning up..."
    # remove the image from the local docker registry
    docker rmi diffrg-$image
done
