# DiFfRG dependency image (Ubuntu 24.04, CPU / GPU=OFF).
#
# This image bakes the *bundled* dependencies (deal.II, Kokkos, autodiff, GSL,
# Eigen, spdlog, rapidcsv, ...) so that CI only has to rebuild the DiFfRG library
# itself -- the slow superbuild (deal.II dominates, ~2-3 h) is paid once, here,
# instead of on every PR. Refresh and push with containers/ci/build-and-push.sh.
#
# Build from the repository root as context:
#   docker buildx build -t ghcr.io/satfra/diffrg-deps:ubuntu24.04 \
#       -f containers/ci/ubuntu24.04-deps.Dockerfile .
#
# Multi-stage: the `builder` stage runs the superbuild; the final stage carries
# only the installed dependency tree (/root/.local/share/DiFfRG/bundled) plus the toolchain and
# system libraries the standalone library build needs -- not the multi-GB build
# scratch dir, and no DiFfRG library source.

# --------------------------------------------------------------------------- #
# Stage 1: build the bundled dependencies via the superbuild.
# --------------------------------------------------------------------------- #
FROM ubuntu:24.04 AS builder

ARG threads=4
ARG DEBIAN_FRONTEND=noninteractive

# Toolchain (cmake 3.28, gcc 13 -> C++20) + the system dependencies DiFfRG can
# consume directly: GSL and LAPACK/BLAS are required; Boost/TBB/HDF5/SUNDIALS are
# auto-detected and used when new enough, otherwise the bundled copies are built.
RUN apt-get -y update && apt-get -y install --no-install-recommends \
        git cmake build-essential gfortran \
        libopenblas-dev libgsl-dev \
        libboost-all-dev libtbb-dev libhdf5-dev libsundials-dev \
        doxygen graphviz python3 patch ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Configure the superbuild, then build *only* the dependency ExternalProject
# targets -- not the `DiFfRG`/`finish` targets. The DiFfRG library (CMakeLists.txt)
# DEPENDS on exactly these, so they yield a complete dependency tree while leaving
# the library out of the image. deal.II_dep transitively builds any bundled
# Kokkos/SUNDIALS it needs. With no -DCMAKE_INSTALL_PREFIX override the superbuild's
# default ($HOME/.local/share/DiFfRG, i.e. /root/... here) is used, so deps land in
# ${prefix}/bundled -> /root/.local/share/DiFfRG/bundled.
# NATIVE=OFF: this image is pushed and pulled on arbitrary hosts/runners, so the
# deps must NOT bake in the build machine's ISA (e.g. AVX-512) -- otherwise they
# SIGILL on a CPU without it. Build a generic, portable target instead.
RUN cmake -S /src -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$HOME/.local/share/DiFfRG \
        -DGPU=OFF -DMPI=OFF -DDiFfRG_DOCUMENTATION=OFF \
        -DNATIVE=OFF \
    && cmake --build /build \
        --target general_dep deal.II_dep kokkos_dep autodiff_dep \
        -j ${threads}

# --------------------------------------------------------------------------- #
# Stage 2: slim runtime image with just the dependency tree + toolchain.
# --------------------------------------------------------------------------- #
FROM ubuntu:24.04
LABEL type=diffrg-deps
LABEL org.opencontainers.image.source=https://github.com/satfra/DiFfRG_current
LABEL org.opencontainers.image.description="DiFfRG bundled dependencies (Ubuntu 24.04, CPU) for fast CI library builds"

ARG DEBIAN_FRONTEND=noninteractive

# Same toolchain + system libraries: needed at CI time to build the DiFfRG library
# against the bundled tree (compiler, cmake, and the system Boost/TBB/HDF5/SUNDIALS
# that the standalone build resolves from standard paths).
RUN apt-get -y update && apt-get -y install --no-install-recommends \
        git cmake build-essential gfortran \
        libopenblas-dev libgsl-dev \
        libboost-all-dev libtbb-dev libhdf5-dev libsundials-dev \
        doxygen graphviz python3 patch ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# COPY/ENV can't shell-expand, so the literal /root (= $HOME for the root user) is
# used; it matches the $HOME-based install prefix in the builder stage.
COPY --from=builder /root/.local/share/DiFfRG/bundled /root/.local/share/DiFfRG/bundled

# Consumed by the CI library build: cmake -DBUNDLED_DIR=$DiFfRG_BUNDLED_DIR ...
ENV DiFfRG_BUNDLED_DIR=/root/.local/share/DiFfRG/bundled
