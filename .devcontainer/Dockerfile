FROM "nvidia/cuda:12.5.1-devel-rockylinux9" as base

# set a directory for the app
WORKDIR /DiFfRG/

# install dependencies
RUN dnf --enablerepo=devel install -y gcc-toolset-12 cmake git openblas-devel python3 python3-pip gsl-devel vim

# run the command
CMD ["scl", "enable", "gcc-toolset-12", "bash"]
