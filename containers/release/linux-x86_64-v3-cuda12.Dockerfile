# Relocatable DiFfRG dependency bundle: linux x86_64, -march=x86-64-v3,
# CUDA 12 (sm_80/Ampere floor), no MPI.
#
# Mirrors linux-x86_64-v3-cpu.Dockerfile with a CUDA-enabled Kokkos. One CUDA
# arch per Kokkos build; AMPERE80 embeds sm_80 SASS plus compute_80 PTX, so
# newer GPUs (Ada, Hopper, Blackwell, ...) run via JIT (cached after first
# launch). Older GPUs (Turing and before) are not covered -- self-build.
#
# Consumers need the CUDA toolkit anyway (their applications compile device
# code), so the bundle does not ship CUDA: its libraries resolve the host's
# libcudart.so.12, and the toolkit minor should be >= the build's (12.8).
#
# Build from the repository root as context:
#   docker buildx build --build-arg bundle_version=1.0.0 \
#       -f containers/release/linux-x86_64-v3-cuda12.Dockerfile .

# --------------------------------------------------------------------------- #
# Stage 1: superbuild of the dependency targets + relocation post-processing.
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:12.8.1-devel-rockylinux9 AS builder

RUN dnf -y install epel-release \
    && dnf -y --enablerepo=devel install \
        gcc-toolset-14 gcc-toolset-14-gcc-gfortran \
        cmake git patch python3 which \
        openblas-devel gsl-devel zlib-devel \
        patchelf zstd xz file binutils \
    && dnf clean all

RUN echo "source /opt/rh/gcc-toolset-14/enable" > /.bashenv
ENV BASH_ENV=/.bashenv
SHELL ["/bin/bash", "-c"]

WORKDIR /src
COPY . /src

ARG threads=6
ARG cuda_arch=AMPERE80

# GPU=ON with a pinned Kokkos arch: no GPU is present at build time, so the
# arch cannot be auto-detected and MUST be given explicitly.
RUN cmake -S /src -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/diffrg \
        -DGPU=ON "-DKokkos_ARCH_LIST=${cuda_arch}" \
        -DMPI=OFF -DDiFfRG_DOCUMENTATION=OFF \
        -DMARCH=x86-64-v3 \
        -DBUILD_BOOST=ON -DBUILD_TBB=ON -DBUILD_HDF5=ON -DBUILD_SUNDIALS=ON \
        -DDEAL_II_GSL=OFF \
        -DUSE_CCACHE=OFF \
        -DBUILD_JOBS=$((2 * threads)) \
        -DDEALII_MAX_JOBS=${threads} \
        -DPETSC_MAX_JOBS=${threads} \
    && cmake --build /build \
        --target general_dep deal.II_dep kokkos_dep autodiff_dep \
        -j ${threads} \
    && chmod -R a+rX /opt/diffrg

ARG bundle_version=0.0.0
ARG git_sha=unknown
RUN GIT_SHA="${git_sha}" bash /src/containers/release/postprocess-bundle.sh \
        /opt/diffrg/bundled /src "${bundle_version}" linux-x86_64-v3-cuda12 x86-64-v3 2.34

# --------------------------------------------------------------------------- #
# Stage 2: runtime image -- proves self-containment, then emits the tarball.
# --------------------------------------------------------------------------- #
# The runtime CUDA image provides libcudart but not the driver's libcuda.so.1
# (that comes from the host driver at run time); link the toolkit's stub so
# the ldd audit can resolve it.
FROM nvidia/cuda:12.8.1-runtime-rockylinux9
LABEL type=diffrg-deps-release
LABEL org.opencontainers.image.source=https://github.com/satfra/DiFfRG_current

ARG bundle_version=0.0.0

RUN dnf -y install openblas-serial zlib tar zstd file \
    && dnf clean all

# The runtime image ships libcudart but not the driver's libcuda.so.1 (host
# driver territory), and not even its stub -- take the stub from the builder's
# devel toolkit so the ldd audit can resolve it.
COPY --from=builder /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so.1
RUN ldconfig

COPY --from=builder /opt/diffrg/bundled /opt/diffrg/bundled
COPY containers/release/check-linkage.sh /usr/local/bin/check-linkage.sh

RUN CHECK_LINKAGE_CUDA=1 bash /usr/local/bin/check-linkage.sh /opt/diffrg/bundled

RUN name="diffrg-deps-${bundle_version}-linux-x86_64-v3-cuda12" \
    && mkdir -p "/dist/${name}" \
    && cp -a /opt/diffrg/bundled "/dist/${name}/bundled" \
    && cp /opt/diffrg/bundled/BUNDLE_MANIFEST.json "/dist/${name}/" \
    && tar -C /dist --sort=name --owner=0 --group=0 --numeric-owner \
           -cf - "${name}" | zstd -19 -T0 -o "/dist/${name}.tar.zst" \
    && rm -rf "/dist/${name}" \
    && cd /dist && sha256sum "${name}.tar.zst" > "${name}.tar.zst.sha256"
