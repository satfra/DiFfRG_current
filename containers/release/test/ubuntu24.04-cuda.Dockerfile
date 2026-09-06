# Environment for validating a CUDA dependency bundle on Ubuntu 24.04.
# nvcc is required (consumers compile device code); still deliberately WITHOUT
# boost/tbb/hdf5/sundials dev packages. Driven by test-tarball.sh -g.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
LABEL type=diffrg-deps-release-test

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install --no-install-recommends \
        git cmake build-essential gfortran \
        g++-14 gcc-14 \
        libopenblas-dev libgsl-dev zlib1g-dev \
        python3 patch ca-certificates curl zstd file pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 defaults to GCC 13, whose libstdc++ trips a nvcc frontend bug
# in C++20 mode; CUDA-bundle consumers must use GCC 14 -- as does this test.
ENV CC=gcc-14 CXX=g++-14
