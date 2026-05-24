#!/bin/bash
# ##############################################################################
# Build a DiFfRG dependency image, verify it, and (optionally) push it to GHCR.
#
# The dependency image bakes the slow superbuild (deal.II, Kokkos, autodiff, ...)
# so that CI only rebuilds the DiFfRG library. Run this locally whenever the
# bundled dependencies change (dependencies/** or the top-level CMakeLists.txt),
# then push so CI picks up the new tree.
#
# Usage: build-and-push.sh [-c] [-j <threads>] [-a <cuda_arch>] [-t <tag>] [-n]
#   -c            build the CUDA image (ubuntu24.04-cuda) instead of CPU (ubuntu24.04)
#   -j <threads>  build threads (default: half the host cores)
#   -a <cuda_arch> Kokkos CUDA arch for the CUDA image (default: AMPERE80)
#   -t <tag>      base tag name (default: ubuntu24.04, or ubuntu24.04-cuda with -c)
#   -n            no-push: build + smoke-test only (don't push to GHCR)
#
# Images are tagged twice under ghcr.io/satfra/diffrg-deps:
#   <tag>             moving tag consumed by CI (.github/workflows/ci.yml)
#   <tag>-<YYYYMMDD>  immutable dated record
#
# Pushing requires a one-time login with a Personal Access Token that has the
# `write:packages` scope (classic PAT) -- and, for the first push of a new
# package, `repo` as well:
#   echo $GHCR_PAT | docker login ghcr.io -u <github-user> --password-stdin
# After the first push, make the package public in the GitHub UI so CI can pull
# it without credentials.
# ##############################################################################
set -euo pipefail

scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
repo="$(cd -- "${scriptpath}/../.." >/dev/null 2>&1 && pwd -P)"

registry="ghcr.io/satfra/diffrg-deps"
cuda=0
threads=''
cuda_arch='AMPERE80'
tag=''
push=1

while getopts cj:a:t:n flag; do
  case "${flag}" in
  c) cuda=1 ;;
  j) threads=${OPTARG} ;;
  a) cuda_arch=${OPTARG} ;;
  t) tag=${OPTARG} ;;
  n) push=0 ;;
  *)
    echo "Unknown flag." >&2
    exit 1
    ;;
  esac
done

# Default to half the host cores.
if [[ -z ${threads} ]]; then
  ncores=''
  if command -v nproc >/dev/null 2>&1; then
    ncores=$(nproc)
  elif command -v sysctl >/dev/null 2>&1; then
    ncores=$(sysctl -n hw.ncpu 2>/dev/null)
  fi
  ncores=${ncores:-2}
  threads=$((ncores / 2))
  [[ ${threads} -lt 1 ]] && threads=1
fi

if [[ ${cuda} -eq 1 ]]; then
  dockerfile="${scriptpath}/ubuntu24.04-cuda-deps.Dockerfile"
  : "${tag:=ubuntu24.04-cuda}"
  build_args=(--build-arg "threads=${threads}" --build-arg "cuda_arch=${cuda_arch}")
else
  dockerfile="${scriptpath}/ubuntu24.04-deps.Dockerfile"
  : "${tag:=ubuntu24.04}"
  build_args=(--build-arg "threads=${threads}")
fi

moving="${registry}:${tag}"
dated="${registry}:${tag}-$(date +%Y%m%d)"

echo "Building dependency image with ${threads} threads (context: ${repo})"
echo "  Dockerfile: ${dockerfile}"
echo "  Tags:       ${moving}, ${dated}"
echo

docker buildx build --load \
  -t "${moving}" -t "${dated}" \
  -f "${dockerfile}" \
  "${repo}" \
  --progress=plain \
  "${build_args[@]}"

# Smoke test: build the DiFfRG library + run the test suite *inside* the freshly
# built image, against the baked dependency tree. This proves the multi-stage
# COPY of /root/.local/share/DiFfRG/bundled yields a usable tree before we publish it.
# The repo is mounted read-only and built in a tmpfs-style /tmp dir so the image
# stays clean. CUDA images skip ctest unless a host GPU is present.
echo
echo "Smoke-testing ${moving} (build DiFfRG library + ctest against baked deps)..."
run_args=(--rm -v "${repo}:/src:ro")
ctest_step="&& ctest --test-dir /tmp/diffrg-bin -j ${threads} --output-on-failure"
if [[ ${cuda} -eq 1 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    run_args+=(--gpus all)
  else
    echo "No host GPU (nvidia-smi); building library only, skipping ctest."
    ctest_step=""
  fi
fi

docker run "${run_args[@]}" "${moving}" bash -c "
  set -e
  cp -r /src/DiFfRG /tmp/DiFfRG-src
  cmake -S /tmp/DiFfRG-src -B /tmp/diffrg-bin \
    -DBUNDLED_DIR=\"\${DiFfRG_BUNDLED_DIR}\" \
    -DCMAKE_BUILD_TYPE=Release -DDiFfRG_TEST=ON -DDiFfRG_DOCUMENTATION=OFF -DNATIVE=OFF
  cmake --build /tmp/diffrg-bin -j ${threads} ${ctest_step}
"
echo "   Smoke test PASSED for ${moving}."

if [[ ${push} -eq 0 ]]; then
  echo
  echo "No-push mode (-n): built and verified ${moving} and ${dated}, not pushing."
  exit 0
fi

echo
echo "Pushing ${moving} and ${dated} to GHCR..."
docker push "${moving}"
docker push "${dated}"
echo "   Pushed. If this is a new package, make it public in the GitHub UI so CI"
echo "   can pull it without credentials."
