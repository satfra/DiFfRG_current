#!/bin/bash

# The Base/ directory contains setups with neither MPI, OpenMP, nor CUDA.
# This script tries to build the DiFfRG docker images for all setups in the Base/ directory.

# get all files in Base/
images=$(ls Base/)

for image in $images; do
    echo "   Building DiFfRG docker image for $image..."
    docker buildx build -t $image -f Base/$image . --no-cache --progress=plain &>$image.log
    if [ $? -ne 0 ]; then
        echo "   Error building $image. Check the log file $image.log for details."
    else
        echo "   Successfully built $image."
    fi
done
