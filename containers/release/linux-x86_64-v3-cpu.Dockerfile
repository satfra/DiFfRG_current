# Relocatable DiFfRG dependency bundle: linux x86_64, -march=x86-64-v3, CPU, no MPI.
#
# Produces the release tarball diffrg-deps-<version>-linux-x86_64-v3-cpu.tar.zst
# in /dist of the final stage; extract it with containers/release/build-release.sh.
#
# Built on Rocky Linux 9 for its glibc 2.34 floor (the tarball then runs on any
# distro at least that new: Rocky 9+, Ubuntu 22.04+, Debian 12+, Fedora, Arch).
# gcc-toolset-14 provides GCC 14 while linking the newer libstdc++ pieces
# statically (libstdc++_nonshared.a), so the host libstdc++ requirement stays at
# the el9 baseline -- postprocess-bundle.sh audits that rather than trusting it.
# The "9" dnf repos track the latest 9.x point release, so gcc-toolset-14 is
# available regardless of which 9.x minor the base image snapshot is.
#
# Unlike the CI deps image (containers/ci/ubuntu24.04-deps.Dockerfile), this
# build force-bundles Boost/TBB/HDF5/SUNDIALS and deliberately installs no dev
# packages for them (nor muparser/suitesparse, so deal.II compiles in its own
# bundled copies) -- the artifact must be self-contained, not pinned to any
# distro's library versions. DEAL_II_GSL=OFF keeps the distro-specific libgsl
# soname out of libdeal_II.so; DiFfRG links GSL directly on the user's host.
#
# Build from the repository root as context:
#   docker buildx build --build-arg bundle_version=1.0.0 \
#       -f containers/release/linux-x86_64-v3-cpu.Dockerfile .

# --------------------------------------------------------------------------- #
# Stage 1: superbuild of the dependency targets + relocation post-processing.
# --------------------------------------------------------------------------- #
FROM rockylinux:9 AS builder

RUN dnf -y install epel-release \
    && dnf -y --enablerepo=devel install \
        gcc-toolset-14 gcc-toolset-14-gcc-gfortran \
        cmake git patch python3 which \
        openblas-devel gsl-devel zlib-devel \
        patchelf zstd xz file binutils \
    && dnf clean all

# Make gcc-toolset-14 the active toolchain: BASH_ENV covers every
# non-interactive bash, and SHELL makes RUN steps use bash in the first place
# (the default /bin/sh -c never reads BASH_ENV, leaving no compiler on PATH).
RUN echo "source /opt/rh/gcc-toolset-14/enable" > /.bashenv
ENV BASH_ENV=/.bashenv
SHELL ["/bin/bash", "-c"]

WORKDIR /src
COPY . /src

# Declared here, not at the top: an ARG invalidates every later layer when its
# value changes, and the dnf/toolset layers above must survive a thread-count
# change.
ARG threads=6

# BUILD_JOBS/-MAX_JOBS caps: the superbuild's own job governor sizes sub-builds
# from the container's view of the host (all cores), which would run deal.II at
# far more jobs than the ${threads} budget -- pin everything to it instead.
RUN cmake -S /src -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/diffrg \
        -DGPU=OFF -DMPI=OFF -DDiFfRG_DOCUMENTATION=OFF \
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

# Relocation fixups + hard audits (ISA, symbol versions, path leaks) + manifest.
# bundle_version/git_sha only matter from here on, so declaring them this late
# keeps the superbuild layer cached across version bumps and commits.
ARG bundle_version=0.0.0
ARG git_sha=unknown
RUN GIT_SHA="${git_sha}" bash /src/containers/release/postprocess-bundle.sh \
        /opt/diffrg/bundled /src "${bundle_version}" linux-x86_64-v3-cpu x86-64-v3 2.34

# --------------------------------------------------------------------------- #
# Stage 2: bare runtime image -- proves self-containment, then emits the tarball.
# --------------------------------------------------------------------------- #
# Deliberately minimal: only the runtime libraries the manifest allowlists plus
# tar/zstd. If the bundle secretly needed any dev package, check-linkage.sh
# fails right here.
FROM rockylinux:9
LABEL type=diffrg-deps-release
LABEL org.opencontainers.image.source=https://github.com/satfra/DiFfRG_current

ARG bundle_version=0.0.0

# openblas-serial, not openblas: on EL9 the latter is a docs-only package and
# libopenblas.so.0 lives in the -serial subpackage.
RUN dnf -y install openblas-serial zlib tar zstd file \
    && dnf clean all

COPY --from=builder /opt/diffrg/bundled /opt/diffrg/bundled
COPY containers/release/check-linkage.sh /usr/local/bin/check-linkage.sh

RUN bash /usr/local/bin/check-linkage.sh /opt/diffrg/bundled

# Deterministic tarball: manifest at top level next to bundled/ (and a second
# copy inside bundled/, which is what survives installation).
RUN name="diffrg-deps-${bundle_version}-linux-x86_64-v3-cpu" \
    && mkdir -p "/dist/${name}" \
    && cp -a /opt/diffrg/bundled "/dist/${name}/bundled" \
    && cp /opt/diffrg/bundled/BUNDLE_MANIFEST.json "/dist/${name}/" \
    && tar -C /dist --sort=name --owner=0 --group=0 --numeric-owner \
           -cf - "${name}" | zstd -19 -T0 -o "/dist/${name}.tar.zst" \
    && rm -rf "/dist/${name}" \
    && cd /dist && sha256sum "${name}.tar.zst" > "${name}.tar.zst.sha256"
