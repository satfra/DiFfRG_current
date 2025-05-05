#!/bin/bash

git clone https://github.com/satfra/DiFfRG.git --depth 1 --branch Implement-Kokkos
mkdir -p build
cd build
cmake ../DiFfRG -DCMAKE_BUILD_TYPE=Release
make -j8
