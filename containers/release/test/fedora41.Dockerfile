# Environment for validating a pre-built DiFfRG dependency bundle on Fedora 41.
# Deliberately WITHOUT boost/tbb/hdf5/sundials dev packages: the bundle must be
# self-contained. Driven by containers/release/test-tarball.sh.
FROM fedora:41
LABEL type=diffrg-deps-release-test

RUN dnf -y install \
        git cmake gcc-c++ gcc-gfortran \
        openblas-devel gsl-devel zlib-devel \
        python3 patch curl zstd file pkgconf-pkg-config \
    && dnf clean all
