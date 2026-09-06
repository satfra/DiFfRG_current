# Environment for validating a pre-built DiFfRG dependency bundle on Rocky 9 --
# the oldest supported host (glibc 2.34, the bundle's floor). Deliberately
# WITHOUT boost/tbb/hdf5/sundials dev packages: the bundle must be
# self-contained. Driven by containers/release/test-tarball.sh.
FROM rockylinux:9
LABEL type=diffrg-deps-release-test

# Rocky 9's default GCC (11) lacks full C++20; gcc-toolset-14 matches the
# compiler generation the bundle was built with. openblas-devel lives in the
# (disabled-by-default) devel repo.
RUN dnf -y install epel-release \
    && dnf -y --enablerepo=devel install \
        gcc-toolset-14 gcc-toolset-14-gcc-gfortran \
        git cmake \
        openblas-devel gsl-devel zlib-devel \
        python3 patch which zstd file pkgconf-pkg-config \
    && dnf clean all

# Activate the toolset for non-interactive (BASH_ENV) and login (profile.d)
# shells alike -- the test driver enters via `bash -lc`.
RUN echo "source /opt/rh/gcc-toolset-14/enable" > /.bashenv \
    && cp /.bashenv /etc/profile.d/gcc-toolset-14.sh
ENV BASH_ENV=/.bashenv
