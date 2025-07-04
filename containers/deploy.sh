#!/bin/bash

git clone https://github.com/satfra/DiFfRG.git --depth 1 --branch Implement-Kokkos
mkdir -p build
cd build
cmake ../DiFfRG -DCMAKE_BUILD_TYPE=Release -DKokkos_ARCH_LIST=Kokkos_ARCH_ADA89
make -j8
