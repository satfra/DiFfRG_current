(Installation)=
# Installation
To compile and run this project, there are very few requirements which you can easily install using your package manager on Linux or MacOS:

- [git](https://git-scm.com/) for external requirements and to clone this repository.
- [CMake](https://www.cmake.org/) for the build systems of DiFfRG, deal.ii and other libraries.
- [GNU Make](https://www.gnu.org/software/make/) or another generator of your choice.
- A compiler supporting at least the C++20 standard. This project is only tested using the [GCC](https://gcc.gnu.org/) compiler suite, as well as with `AppleClang`, but in principle, ICC or standard Clang should also work.
- LAPACK and BLAS in some form, e.g. [OpenBlas](https://www.openblas.net/).
- The GNU Scientific Library [GSL](https://www.gnu.org/software/gsl/). If not found by DiFfRG, it will try to install it by itself.
- [Python](https://www.python.org/) is required by the Boost build system and used for visualization.
- [Doxygen](https://www.doxygen.org/) and [graphviz](https://www.graphviz.org/download/) to build the documentation.

The following requirements are optional:
- [ParaView](https://www.paraview.org/), a program to visualize and post-process the vtk data saved by DiFfRG when treating FEM discretizations.
- A GPU backend for the momentum integration routines, which gives a large speedup for fully momentum-dependent flow equations (10 - 100x). DiFfRG uses [Kokkos](https://kokkos.org/) to abstract the parallel backend, so GPU support is enabled through Kokkos' CUDA (NVIDIA) or HIP (AMD) backends. For the CUDA backend you need a working [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) together with a host compiler compatible with your `nvcc` version (e.g. `g++` <= 13.2 for CUDA 12.5). GPU support is controlled by the `-DGPU` CMake option (see below).

All other requirements are bundled and automatically built with DiFfRG.

## Supported systems

The framework has been tested with the following systems:

### Arch Linux
```bash
$ pacman -S git cmake gcc gcc-fortran blas-openblas paraview python doxygen graphviz gsl
```
In case you want to run with CUDA, as of January 2025 you have to have very specific versions of CUDA and gcc installed. Currently, the gcc13 compiler in the Arch package repository is incompatible with CUDA. To configure a system with a compatible CUDA+gcc configuration, them install directly from the Arch package archive
```bash
$ pacman -U https://archive.archlinux.org/packages/g/gcc12/gcc12-12.3.0-6-x86_64.pkg.tar.zst \
            https://archive.archlinux.org/packages/g/gcc12-libs/gcc12-libs-12.3.0-6-x86_64.pkg.tar.zst \
            https://archive.archlinux.org/packages/c/cuda/cuda-12.3.2-1-x86_64.pkg.tar.zst
```

### Rocky Linux
```bash
$ dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel doxygen doxygen-latex python3 python3-pip gsl-devel
$ scl enable gcc-toolset-12 bash
```

The second line is necessary to switch into a shell where `g++-12` is available

### Ubuntu
```bash
$ apt-get update
$ apt-get install git cmake gfortran libopenblas-dev paraview build-essential python3 doxygen graphviz libgsl-dev
```

### MacOS
First, install xcode and homebrew, then run
```bash
$ brew install cmake gcc doxygen paraview graphviz gsl python3 bash
```

### Windows

If using Windows, instead of running the project directly, it is recommended to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/setup/environment) and then go through the installation as if on Linux (e.g. Arch or Ubuntu).

### Docker and other container runtime environments

Although a native install should be unproblematic in most cases, the setup with CUDA functionality may be daunting. Especially on high-performance clusters, and also depending on the packages available for  chosen distribution, it may be much easier to work with the framework inside a container.

The specific choice of runtime environment is up to the user, however we provide a small build script to create a Docker/OCI container in which DiFfRG will be built and then tested through Singularity/Apptainer when available.
To do this, you will need `docker`, `docker-buildx`, and `singularity` or `apptainer`. For CUDA-compatible image execution you also need the host NVIDIA driver available to Singularity/Apptainer.

To build a Docker image and test it through Singularity/Apptainer, you can run the script `build-container.sh` in the `containers/` folder:
```bash
$ cd containers
$ bash build-container.sh
```

If using other environments, e.g. [ENROOT](https://github.com/NVIDIA/enroot), the preferred approach is simply to build an image on top of the [CUDA images by NVIDIA](https://hub.docker.com/r/nvidia/cuda/tags). Optimal compatibility is given using `nvidia/cuda:12.5.1-devel-rockylinux`. Proceed with the installation setup for  Rocky Linux above.

## Setup

### Quick install (recommended)

From the shell, run (this requires `curl` to be available on your system):
```bash
$ bash <(curl -s -L https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install.sh)
```

You can optionally specify the installation folder or the number of threads:
```bash
$ THREADS=6 FOLDER=${HOME}/.local/share/DiFfRG/ bash <(curl -s -L https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install.sh)
```

Run `bash install.sh --help` for more options.

### Manual installation

Clone the repository:
```bash
$ git clone https://github.com/satfra/DiFfRG_current.git
```

Then, create a build directory and run cmake:
```bash
$ cd DiFfRG_current
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG/ -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./ -- -j8
```

By default, the library will install itself to `$HOME/.local/share/DiFfRG`, but you can control the destination by pointing `CMAKE_INSTALL_PREFIX` to a directory of your choice.

This top-level build is a *superbuild*: it builds the bundled dependencies (deal.II, Kokkos, and any of Boost/TBB/HDF5/SUNDIALS not found on the system) into `${CMAKE_INSTALL_PREFIX}/bundled` and then builds and installs the DiFfRG library itself.

### Build options

The most important options to pass to the top-level `cmake` invocation are:

- `-DGPU=ON/OFF` — GPU support via the Kokkos CUDA/HIP backend (default `ON`).
- `-DMPI=ON/OFF` — MPI support (default `OFF`).
- `-DNATIVE=ON/OFF` — optimize for the build machine's CPU (`-march=native`). Disable for portable binaries (default `ON`).
- `-DDiFfRG_TEST=ON` — build the test suite (default `OFF`).
- `-DDiFfRG_DOCUMENTATION=ON` — build this documentation (default `ON`).
- `-DBUILD_PETSC=ON` / `-DBUILD_OpenBLAS=ON` — additionally build PETSc / OpenBLAS (default `OFF`).

Boost, TBB, HDF5 and SUNDIALS are taken from the system when a viable version is found, and otherwise built from the bundled, pinned sources. For each library `<LIB>` ∈ {`BOOST`, `TBB`, `HDF5`, `SUNDIALS`} you can override this:

- `-D<LIB>_DIR=<prefix>` — use the install at this prefix (a fatal error is raised if it is not usable).
- `-DBUILD_<LIB>=ON` — always build the bundled, pinned copy, ignoring any system install.

The minimum supported versions are Boost ≥ 1.81, oneTBB ≥ 2021, HDF5 ≥ 1.12 and SUNDIALS ≥ 5.4.

### Rebuilding the library only

Once the dependencies have been installed by a full build, you usually do not want to rebuild them when working on DiFfRG itself. You can then configure directly from the `DiFfRG/` subfolder, pointing CMake at the already-installed bundled dependencies via `BUNDLED_DIR`:

```bash
$ cd DiFfRG
$ mkdir build && cd build
$ cmake .. -DBUNDLED_DIR=$HOME/.local/share/DiFfRG/bundled/ -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./ -- -j8
```

Note the `/bundled/` suffix: `BUNDLED_DIR` points at the directory the superbuild installed the dependencies into, which defaults to `${CMAKE_INSTALL_PREFIX}/bundled`.

### Verifying your installation

After installation, you can verify that all dependencies are correctly found:
```bash
$ cmake -DBUNDLED_DIR=~/.local/share/DiFfRG/bundled -P ~/.local/share/DiFfRG/cmake/verify_install.cmake
```

This prints a pass/fail table for each dependency.
