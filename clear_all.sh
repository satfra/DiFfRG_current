#!/bin/bash

read -p "Delete full build tree of DiFfRG library? [y/N] " option_clear
option_clear=${option_clear:-N}

if [[ ${option_clear} = "y" ]] || [[ ${option_clear} = "Y" ]]; then
  echo "Deleting..."
  cd external
  ./build_Catch2.sh -c
  ./build_autodiff.sh -c
  ./build_dealii.sh -c
  ./build_kokkos.sh -c
  ./build_rmm.sh -c
  ./build_sundials.sh -c
  ./build_boost.sh -c
  ./build_spdlog.sh -c
  ./build_eigen.sh -c
  ./build_thread-pool.sh -c
  ./build_qmc.sh -c
  cd ..
  rm -rf ./DiFfRG_build ./DiFfRG_install
  rm -rf ./logs
  rm -f ./external/*.log
fi

echo "    Done"
