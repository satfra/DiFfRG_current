#!/bin/bash

git clone https://github.com/satfra/DiFfRG_current.git --depth 1 --branch main
mkdir -p build
cd build
cmake ../DiFfRG -DCMAKE_BUILD_TYPE=Release
make -j8
