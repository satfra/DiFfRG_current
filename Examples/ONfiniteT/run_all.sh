#!/bin/bash

cd build
./CG -ss /output/name=CG_out
./dDG -ss /output/name=dDG_out
./LDG -ss /output/name=LDG_out
./KT -ss /output/name=KT_out
./KT_sigma -ss /output/name=KT_sigma_out
