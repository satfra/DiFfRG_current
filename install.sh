#!/bin/bash

tempFolder="/tmp/"
repoName="DiFfRG_current"
repo="https://github.com/satfra/${repoName}.git"

if [[ -v FOLDER ]]; then
  installFolder=${FOLDER}
else
  installFolder="${HOME}/.local/share/DiFfRG"
fi
echo "Installation directory set to ${installFolder}"

echo "Cloning to temporary directory ${tempFolder}${repoName}..."

cd ${tempFolder}
git clone ${repo} ${tempFolder}${repoName} --depth 1

echo "Running CMake in ${tempFolder}${repoName}/build..."
cd ${repoName}
mkdir -p ${tempFolder}${repoName}/build
cd ${tempFolder}${repoName}/build
cmake ${tempFolder}${repoName} -DCMAKE_INSTALL_PREFIX=${installFolder}

if [[ -v THREADS ]]; then
  bcores=$THREADS
else
  ncores="$(($(lscpu | awk '/^Socket\(s\)/{ print $2 }') * $(lscpu | awk '/^Core\(s\) per socket/{ print $4 }')))"
  bcores=$((ncores / 2))
fi
echo "Building DiFfRG with ${bcores} cores..."
make -j${bcores}
