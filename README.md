[![arXiv](https://img.shields.io/badge/arXiv-2412.13043-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2412.13043)
[![Doxygen](https://img.shields.io/badge/doxygen-2C4AA8?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://satfra.github.io/DiFfRG/cpp/index.html)
[![Wolfram](https://img.shields.io/badge/wolfram_doc-cf1c10?style=for-the-badge&logo=wolfram)](https://satfra.github.io/DiFfRG/wolfram/html/guide/DiFfRG.html)
[![Python](https://img.shields.io/badge/python_doc-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://satfra.github.io/DiFfRG/python/index.html)

# DiFfRG - A Discretization Framework for functional Renormalization Group flows

DiFfRG is a set of tools for the discretization of flow equations arising in the functional Renormalization Group (fRG).
It supports the setup and calculation of large systems of flow equations allowing for complex combinations of vertex and derivative expansions.

For spatial discretizations, i.e. discretizations of field space mostly used for derivative expansions, DiFfRG makes different finite element (FE) methods available. These include:
- Continuous Galerkin FE
- Discontinuos Galerkin FE
- Direct discontinuous Galerkin FE
- Local discontinuous Galerkin FE (including derived finite volume (FV) schemes)

The FEM methods included in DiFfRG are built upon the [deal.ii](https://www.dealii.org/) finite element library, which is highly parallelized and allows for great performance and flexibility.
PDEs consisting of RG-time dependent equations, as well as stationary equations can be solved together during the flow, allowing for techniques like flowing fields in a very accessible way.

Both explicit and implicit time-stepping methods are available and allow thus for efficient RG-time integration in the symmetric and symmetry-broken regime.

We also include a set of tools for the evaluation of integrals and discretization of momentum dependencies.

For an overview, please see the [accompanying paper](https://arxiv.org/abs/2412.13043), the ***[tutorial page](https://satfra.github.io/DiFfRG/cpp/TutorialTOC.html)*** in the [documentation](https://satfra.github.io/DiFfRG/cpp/index.html) and the examples in `Examples/`. 

## Citation

If you use DiFfRG in your scientific work, please cite the corresponding paper:
```
@article{Sattler:2024ozv,
    author = "Sattler, Franz R. and Pawlowski, Jan M.",
    title = "{DiFfRG: A Discretisation Framework for functional Renormalisation Group flows}",
    eprint = "2412.13043",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "12",
    year = "2024"
}
```


## Requirements

To compile and run this project, there are very few requirements which you can easily install using your package manager on Linux or MacOS:

- [git](https://git-scm.com/) for external requirements and to clone this repository.
- [CMake](https://www.cmake.org/) for the build systems of DiFfRG, deal.ii and other libraries.
- [GNU Make](https://www.gnu.org/software/make/) or another generator of your choice.
- A compiler supporting at least the C++20 standard. This project is only tested using the [GCC](https://gcc.gnu.org/) compiler suite, as well as with `AppleClang`, but in principle, ICC or standard Clang should also work.
- LAPACK and BLAS in some form, e.g. [OpenBlas](https://www.openblas.net/).
- The GNU Scientific Library [GSL](https://www.gnu.org/software/gsl/). If not found by DiFfRG, it will try to install it by itself.
- [Doxygen](https://www.doxygen.org/) and [graphviz](https://www.graphviz.org/download/) to build the documentation.

The following requirements are optional:
- [Python](https://www.python.org/) is used in the library for visualization purposes. Furthermore, adaptive phase diagram calculation is implemented as a python routine.
- [ParaView](https://www.paraview.org/), a program to visualize and post-process the vtk data saved by DiFfRG when treating FEM discretizations.
- [CUDA](https://developer.nvidia.com/cuda-toolkit) for integration routines on the GPU, which gives a huge speedup for the calculation of fully momentum dependent flow equations (10 - 100x). In case you wish to use CUDA, make sure you have a compiler available on your system compatible with your version of `nvcc`, e.g. `g++`<=13.2 for CUDA 12.5

All other requirements are bundled and automatically built with DiFfRG.
The framework has been tested with the following systems:

#### Arch Linux
```bash
$ pacman -S git cmake gcc blas-openblas blas64-openblas paraview python doxygen graphviz gsl
```
For a CUDA-enabled build, additionally 
```bash
$ pacman -S cuda
```


#### Rocky Linux
```bash
$ dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel doxygen doxygen-latex python3 python3-pip gsl-devel patch
$ scl enable gcc-toolset-12 bash
```

The second line is necessary to switch into a shell where `g++-12` is available

#### Ubuntu
```bash
$ apt-get update
$ apt-get install git cmake libopenblas-dev paraview build-essential python3 doxygen graphviz libgsl-dev
```
For a CUDA-enabled build, additionally 
```bash
$ apt-get install cuda
```

#### MacOS
First, install xcode and homebrew, then run
```bash
$ brew install cmake doxygen paraview graphviz gsl
```

For better performance, it is recommended to install OpenMP. 
```bash
$ brew install libomp
```

To let the build system know about it, you need to set the environment variable `OpenMP_ROOT` to the OpenMP installation directory, e.g.
```bash
$ export OpenMP_ROOT=$(brew --prefix)/opt/libomp
```
This can be automatically done by adding the above line to your `~/.zshrc` file,
```bash
echo 'export OpenMP_ROOT=$(brew --prefix)/opt/libomp' >> ~/.zshrc
```
After adding this line, you can either restart your terminal or run
```bash
$ source ~/.zshrc
```

#### Windows

If using Windows, instead of running the project directly, it is recommended to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/setup/environment) and then go through the installation as if on Linux (e.g. Arch or Ubuntu).

## Installation

### CMake

You can download a script to install DiFfRG locally directly from a CMake file by putting into your `CMakeLists.txt` the lines
```CMake
file(DOWNLOAD
  https://raw.githubusercontent.com/satfra/DiFfRG/refs/heads/Implement-Kokkos/DiFfRG/cmake/InstallDiFfRG.cmake
  ${CURRENT_BINARY_DIR}/cmake/InstallDiFfRG.cmake)
include(${CURRENT_BINARY_DIR}/cmake/InstallDiFfRG.cmake)
```
This will fetch a script, which will automatically download and install DiFfRG and all of its dependencies to `$HOME/.local/share/DiFfRG`.
If you wish to change this directory, or some other default values, you can set the following optional variables:
```CMake
set(DiFfRG_INSTALL_DIR $ENV{HOME}/.local/share/DiFfRG/)
set(DiFfRG_BUILD_DIR $ENV{HOME}/.local/share/DiFfRG/build/)
set(DiFfRG_SOURCE_DIR $ENV{HOME}/.local/share/DiFfRG/src/)
set(TRY_DiFfRG_VERSION Implement-Kokkos)
set(PARALLEL_JOBS 8)
```

### Manual installation

You can also manually clone DiFfRG to a directory of your choice
```bash
$ git clone https://github.com/satfra/DiFfRG.git
```
Then, create a build directory and run cmake
```bash
$ cd DiFfRG
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG/ -DCMAKE_BUILD_TYPE=Release
$ cmake --build ./ -- -j8"
```
By default, the library will install itself to `$HOME/.local/shared/DiFfRG`, but you can control the destination by pointing `CMAKE_INSTALL_PREFIX` to a directory of your choice.

### Docker and other container runtime environments

Although a native install should be unproblematic in most cases, the setup with CUDA functionality may be daunting. Especially on high-performance clusters, and also depending on the packages available for  chosen distribution, it may be much easier to work with the framework inside a container to avoid conflicting dependencies.

Besides the manual setup described below, we recommend using [development containers](https://code.visualstudio.com/docs/devcontainers/containers) if you are using VSCode. An appropriate `.devcontainers` configuration can be adapted from the one found in the DiFfRG top level directory.

The specific choice of container runtime environment is up to the user, however we provide a small build script to create a docker container for DiFfRG.
To do this, you will need `docker`, `docker-buildx` and the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#) in case you wish to create a CUDA-compatible image.

To build a docker image, you can run the script `build-container.sh` in the `containers/` folder, which will guide you through the process, i.e.
```bash
$ cd containers
$ bash build-container.sh
```

If using other environments, e.g. [ENROOT](https://github.com/NVIDIA/enroot), the preferred approach is simply to build an image on top of one of the [CUDA images by NVIDIA](https://hub.docker.com/r/nvidia/cuda/tags).

For example, with ENROOT a DiFfRG image can be built on top of rockylinux9 by following these steps:
```bash
$ enroot import docker://nvidia/cuda:12.8.1-devel-rockylinux9
$ enroot create --name DiFfRG nvidia+cuda+12.5.1-devel-rockylinux9.sqsh
$ enroot start --root --rw -m ./:/DiFfRG DiFfRG bash
```
Afterwards, one proceeds with the above Rocky Linux setup.

## Getting started with simulating fRG flows

For an overview, please see the ***[tutorial page](https://satfra.github.io/DiFfRG/cpp/TutorialTOC.html)*** in the [documentation](https://satfra.github.io/DiFfRG/cpp/index.html). A local documentation is also always built automatically when running the setup script, but can also be built manually by running
```bash
$ make documentation
```
inside the `DiFfRG_build` directory. You can find then a code reference in the top directory.

All backend code is contained in the DiFfRG directory.

Several simulations are defined in the Applications directory, which can be used as a starting point for your own simulations.

# Tips and FAQ

## Logfiles and install issues

If DiFfRG fails to build on your machine, first check the appropriate logs. You find the main log at `~/.local/share/DiFfRG/build/DiFfRG.log` if you are using the CMake install. Otherwise, you can redirect the output of the build, e.g.
```bash
cmake --build ./ -- -j8 | tee DiFfRG.log
```
and analyze the result.

If DiFfRG proves to be incompatible with your machine, please open an Issue on GitHub [here](https://github.com/satfra/DiFfRG/issues), or, alternatively, send an email to the author (see the [publication](https://arxiv.org/abs/2412.13043)).


## Contributing

DiFfRG is a work in progress. If you find some feature missing, a bug, or some other kind of improvement, you can get involved in the further development of DiFfRG. 

Thanks to the collaborative nature of GitHub, you can simply fork the project and work on a private copy on your own GitHub account. Feel also encouraged to open an [issue](https://github.com/satfra/DiFfRG/issues), or if you already have a (partially) ready contribution, open a [pull request](https://github.com/satfra/DiFfRG/pulls).


## Configuration files

A DiFfRG simulation requires you to provide a valid `parameters.json` file in the execution path, or alternatively provide another JSON-file using the `-p` flag (see below).

To generate a "stock" `parameters.json` in the current folder, you can call any DiFfRG application as
```bash
$ ./my_simulation --generate-parameter-file
```
Before usage, don't forget to put in the parameters you defined in your own simulation!


## Progress output

To monitor the progress of the simulation, one can set the `verbosity` parameter either in the parameter file,
```json
{
  "output": {
    "verbosity": 1
  },
}
```
or from the CLI,
```bash
$ ./my_simulation -si /output/verbosity=1
```


## Modifying parameters from the CLI

Any DiFfRG simulation using the `DiFfRG::ConfigurationHelper` class can be asked to give some syntax pertaining to the configuration:
```bash
$ ./my_simulation --help
This is a DiFfRG simulation. You can pass the following optional parameters:
  --help                      shows this text
  --generate-parameter-file   generates a parameter file with some default values
  -p                          specifiy a parameter file other than the standard parameter.json
  -sd                         overwrite a double parameter. This should be in the format '-s physical/T=0.1'
  -si                         overwrite an integer parameter. This should be in the format '-s physical/Nc=1'
  -sb                         overwrite a boolean parameter. This should be in the format '-s physical/use_sth=true'
  -ss                         overwrite a string parameter. This should be in the format '-s physical/a=hello'
```
e.g.
```bash
$ ./my_simulation -sd /physical/Lambda=1.0
```

## Timestepper choice

In general, the `IDA` timestepper from the `SUNDIALS`-suite has proven to be the optimal choice for any fRG-flow with convexity restoration. Additionally, this solver allows for out-of-the-box solving of additional algebraic systems, which is handy for more complicated fRG setups.

If solving purely variable-dependent systems, one of the `Boost` time steppers, `Boost_RK45`, `Boost_RK78` or `Boost_ABM`. The latter is especially excellent for extremely large systems which have no extremely fast dynamics, but lacks adaptive timestepping. In practice, choosing `Boost_ABM` over one of the RK steppers may speed up a Yang-Mills simulation with full momentum dependences by more than a factor of 10.

For systems with both spatial discretisations and variables, consider one of the implicit-explicit mixtures, `SUNDIALS_IDA_Boost_RK45`,  `SUNDIALS_IDA_Boost_RK78` or `SUNDIALS_IDA_Boost_ABM`.

# Other Libraries used

The following third-party libraries are utilised by DiFfRG. They are automatically built and installed DiFfRG during the build process.

- The main backend for field-space discretization is [deal.II](https://www.dealii.org/), which provides the entire FEM-machinery as well as many other utility components.
- For performant and convenient calculation of Jacobian matrices we use the [autodiff](https://github.com/autodiff/autodiff) library, which implements automatic forward and backwards differentiation in C++ and also in CUDA.
- [Kokkos](https://github.com/kokkos/kokkos), a performance portability framework for shared-memory parallelization on GPU and CPU. We use it for the integration routines for flow equations.
- Time integration relies heavily on the [SUNDIALS](https://computing.llnl.gov/projects/sundials) suite, specifically on the IDAs solver.
- [Boost](https://www.boost.org/) provides explicit time-stepping and various math algorithms.
- [Rapidcsv](https://github.com/d99kris/rapidcsv) for quick processing of .csv files.
- [Catch2](https://github.com/catchorg/Catch2) for unit testing.
- [spdlog](https://github.com/gabime/spdlog) for logging.
- [Doxygen Awesome](https://github.com/jothepro/doxygen-awesome-css) for a modern doxygen theme.
- [Eigen](https://eigen.tuxfamily.org/) for some linear-algebra related tasks.