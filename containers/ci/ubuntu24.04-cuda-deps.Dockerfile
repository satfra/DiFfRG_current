# DiFfRG dependency image (Ubuntu 24.04, CUDA / GPU=ON).
#
# Same idea as ubuntu24.04-deps.Dockerfile but with GPU support compiled into the
# bundled deal.II/Kokkos. Intended for *manual* GPU testing (a deps image plus a
# library build + ctest run on a GPU host); the automated GitHub Actions CI stays
# CPU-only because hosted runners have no GPU.
#
# No GPU is present during `docker build`, so Kokkos' native-arch detection cannot
# run -- pin a CUDA architecture via the cuda_arch build-arg (default AMPERE80 /
# compute 8.0). Override with --build-arg cuda_arch=ADA89 etc.
#
# Build from the repository root as context:
#   docker buildx build -t ghcr.io/satfra/diffrg-deps:ubuntu24.04-cuda \
#       --build-arg cuda_arch=AMPERE80 \
#       -f containers/ci/ubuntu24.04-cuda-deps.Dockerfile .

# --------------------------------------------------------------------------- #
# Stage 1: build the bundled dependencies (GPU=ON) via the superbuild.
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

ARG threads=4
ARG cuda_arch=AMPERE80
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install --no-install-recommends \
        git cmake build-essential gfortran \
        libopenblas-dev libgsl-dev \
        libboost-all-dev libtbb-dev libhdf5-dev libsundials-dev \
        doxygen graphviz python3 patch ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Deps-only build (see ubuntu24.04-deps.Dockerfile for the rationale); GPU on with
# a pinned Kokkos architecture since no device is visible at build time.
RUN cmake -S /src -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/DiFfRG \
        -DGPU=ON -DMPI=OFF -DDiFfRG_DOCUMENTATION=OFF \
        -DNATIVE=OFF \
        -DKokkos_ARCH=${cuda_arch} -DKokkos_ARCH_LIST=${cuda_arch} \
    && cmake --build /build \
        --target general_dep deal.II_dep kokkos_dep autodiff_dep \
        -j ${threads}

# --------------------------------------------------------------------------- #
# Stage 2: slim runtime image with the dependency tree + CUDA toolchain.
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
LABEL type=diffrg-deps
LABEL org.opencontainers.image.source=https://github.com/satfra/DiFfRG_current
LABEL org.opencontainers.image.description="DiFfRG bundled dependencies (Ubuntu 24.04, CUDA) for manual GPU builds"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install --no-install-recommends \
        git cmake build-essential gfortran \
        libopenblas-dev libgsl-dev \
        libboost-all-dev libtbb-dev libhdf5-dev libsundials-dev \
        doxygen graphviz python3 patch ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/DiFfRG/bundled /opt/DiFfRG/bundled

ENV DiFfRG_BUNDLED_DIR=/opt/DiFfRG/bundled
