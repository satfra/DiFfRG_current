#!/bin/bash
# ##############################################################################
# DiFfRG interactive installer.
#
# Run from anywhere:
#   bash <(curl -sL https://raw.githubusercontent.com/satfra/DiFfRG_current/main/install_diffrg.sh)
# or
#   wget -qO install_diffrg.sh https://raw.githubusercontent.com/satfra/DiFfRG_current/main/install_diffrg.sh
#   bash install_diffrg.sh
#
# A wizard that walks through the typical choices:
#   - pre-built dependency bundle (fast; Linux x86_64 with AVX2) or full
#     self-build of the dependency superbuild,
#   - install prefix and temporary build folder,
#   - features for self-builds (MPI, GPU, MUMPS, documentation, -march),
#   - optional copy of the Examples/Tutorials and documentation sources.
#
# Every question can instead be answered with a flag for non-interactive use:
#   --mode prebuilt|source     installation mode
#   --prefix DIR               install prefix        (default ~/.local/share/DiFfRG)
#   --build-dir DIR            temporary build dir   (default /tmp/diffrg-build)
#   --deps-version X.Y.Z       prebuilt bundle version (default: latest)
#   --deps-variant NAME        prebuilt bundle variant (default: platform CPU
#                              bundle; wizard offers the CUDA one when a GPU is found)
#   --deps-file TARBALL        prebuilt: install from a local bundle tarball
#   --threads N                build threads         (default 6)
#   --mpi / --no-mpi           self-build: MPI support
#   --gpu / --no-gpu           self-build: GPU (CUDA) support
#   --mumps / --no-mumps       self-build: PETSc MUMPS solver (default: follow MPI)
#   --docs / --no-docs         build the documentation
#   --march VALUE              self-build: native | x86-64-v3 | none
#   --examples DIR             copy Examples/Tutorials (and docs) there
#   --mathematica DIR          install the DiFfRG Mathematica package there
#   --no-mathematica           skip the Mathematica package
#   --force                    replace an existing <prefix>/bundled without asking
#   --yes                      accept all remaining defaults, no prompts
# ##############################################################################
set -euo pipefail

usage() {
  cat <<'EOF'
DiFfRG interactive installer.

Run from anywhere:
  bash <(curl -sL https://raw.githubusercontent.com/satfra/DiFfRG_current/main/install_diffrg.sh)

Walks through the typical choices (pre-built dependency bundle or full
self-build, install prefix, build folder, features, examples) and performs the
complete installation. Flags for non-interactive use:

  --mode prebuilt|source     installation mode
  --prefix DIR               install prefix        (default ~/.local/share/DiFfRG)
  --build-dir DIR            temporary build dir   (default /tmp/diffrg-build)
  --deps-version X.Y.Z       prebuilt bundle version (default: latest)
  --deps-variant NAME        prebuilt bundle variant (default: platform CPU bundle)
  --deps-file TARBALL        prebuilt: install from a local bundle tarball
  --threads N                build threads         (default 6)
  --mpi / --no-mpi           self-build: MPI support
  --gpu / --no-gpu           self-build: GPU (CUDA) support
  --mumps / --no-mumps       self-build: PETSc MUMPS solver (default: follow MPI)
  --docs / --no-docs         build the documentation
  --march VALUE              self-build: native | x86-64-v3 | none
  --examples DIR             copy Examples/Tutorials (and docs) there
  --mathematica DIR          install the DiFfRG Mathematica package there
  --no-mathematica           skip the Mathematica package
  --force                    replace an existing <prefix>/bundled without asking
  --yes                      accept all remaining defaults, no prompts
EOF
}

REPO_URL="${DIFFRG_REPO_URL:-https://github.com/satfra/DiFfRG_current.git}"
REPO_API="https://api.github.com/repos/satfra/DiFfRG_current"
DOCS_URL="https://satfra.github.io/DiFfRG_current"

err() { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }
info() { echo -e "\033[1;32m==>\033[0m $*"; }
warn() { echo -e "\033[1;33mWARNING:\033[0m $*" >&2; }

# ------------------------------------------------------------------ defaults --
mode=''
prefix="${FOLDER:-$HOME/.local/share/DiFfRG}"
build_dir="/tmp/diffrg-build"
deps_version=''
deps_variant=''
deps_file=''
threads="${THREADS:-6}"
opt_mpi=0
opt_gpu=2 # 2 = auto (on when nvcc is found)
opt_mumps=2 # 2 = follow MPI
opt_docs=0
march="native"
examples_dir=''
mathematica_dir=''
mathematica_asked=0
assume_yes=0
force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  --mode) mode="$2"; shift 2 ;;
  --prefix) prefix="$2"; shift 2 ;;
  --build-dir) build_dir="$2"; shift 2 ;;
  --deps-version) deps_version="$2"; shift 2 ;;
  --deps-variant) deps_variant="$2"; shift 2 ;;
  --deps-file) deps_file="$2"; shift 2 ;;
  --threads) threads="$2"; shift 2 ;;
  --mpi) opt_mpi=1; shift ;;
  --no-mpi) opt_mpi=0; shift ;;
  --gpu) opt_gpu=1; shift ;;
  --no-gpu) opt_gpu=0; shift ;;
  --mumps) opt_mumps=1; shift ;;
  --no-mumps) opt_mumps=0; shift ;;
  --docs) opt_docs=1; shift ;;
  --no-docs) opt_docs=0; shift ;;
  --march) march="$2"; shift 2 ;;
  --examples) examples_dir="$2"; shift 2 ;;
  --mathematica) mathematica_dir="$2"; mathematica_asked=1; shift 2 ;;
  --no-mathematica) mathematica_dir=''; mathematica_asked=1; shift ;;
  --force) force=1; shift ;;
  --yes) assume_yes=1; shift ;;
  -h | --help) usage; exit 0 ;;
  *) err "Unknown option: $1 (see --help)" ;;
  esac
done

# --------------------------------------------------------------- interaction --
# All prompts talk to the terminal directly so `curl | bash` works: stdin may
# be the script itself. Without a terminal, defaults apply (as with --yes).
interactive=0
if [[ ${assume_yes} -eq 0 ]] && { exec 3</dev/tty 4>/dev/tty; } 2>/dev/null; then
  interactive=1
fi
_c=0 # menu-selection result of choose()

# choose <varname> <title> <option>... -- arrow-key menu (falls back to a
# numbered prompt on terminals without ANSI support). Sets <varname> to the
# 0-based selected index; the first option is the default.
choose() {
  local __var="$1" title="$2"; shift 2
  local opts=("$@") cur=0 key i
  if [[ ${interactive} -eq 0 ]]; then printf -v "${__var}" 0; return; fi
  printf '\n\033[1m%s\033[0m  (arrows + enter, or number)\n' "${title}" >&4
  while true; do
    for i in "${!opts[@]}"; do
      if [[ ${i} -eq ${cur} ]]; then printf '  \033[7m %s \033[0m\n' "${opts[${i}]}" >&4
      else printf '   %s\n' "${opts[${i}]}" >&4; fi
    done
    IFS= read -rsn1 key <&3 || { key=''; }
    [[ ${key} == $'\x1b' ]] && { IFS= read -rsn2 -t 0.05 key <&3 || key=''; }
    case "${key}" in
    '[A') [[ ${cur} -gt 0 ]] && cur=$((cur - 1)) ;;
    '[B') [[ ${cur} -lt $((${#opts[@]} - 1)) ]] && cur=$((cur + 1)) ;;
    '') break ;;
    [1-9]) [[ ${key} -le ${#opts[@]} ]] && { cur=$((key - 1)); break; } ;;
    esac
    printf '\033[%dA' "${#opts[@]}" >&4
  done
  printf '\033[%dA' "${#opts[@]}" >&4
  for i in "${!opts[@]}"; do
    if [[ ${i} -eq ${cur} ]]; then printf '  \033[1m> %s\033[0m\033[K\n' "${opts[${i}]}" >&4
    else printf '   %s\033[K\n' "${opts[${i}]}" >&4; fi
  done
  printf -v "${__var}" '%s' "${cur}"
}

# ask <varname> <question> <default>
ask() {
  local __var="$1" q="$2" def="$3" reply=''
  if [[ ${interactive} -eq 1 ]]; then
    printf '\033[1m%s\033[0m [%s]: ' "${q}" "${def}" >&4
    IFS= read -r reply <&3 || reply=''
  fi
  printf -v "${__var}" '%s' "${reply:-${def}}"
}

# toggles <title> <name:state>... -- multi-select; echoes final states as
# "name=0/1" lines on stdout. Space toggles, enter confirms.
toggles() {
  local title="$1"; shift
  local names=() states=() cur=0 key i spec
  for spec in "$@"; do names+=("${spec%%:*}"); states+=("${spec##*:}"); done
  if [[ ${interactive} -eq 1 ]]; then
    printf '\n\033[1m%s\033[0m  (arrows, space toggles, enter confirms)\n' "${title}" >&4
    while true; do
      for i in "${!names[@]}"; do
        local box='[ ]'; [[ ${states[${i}]} -eq 1 ]] && box='[x]'
        if [[ ${i} -eq ${cur} ]]; then printf '  \033[7m %s %s \033[0m\n' "${box}" "${names[${i}]}" >&4
        else printf '   %s %s\n' "${box}" "${names[${i}]}" >&4; fi
      done
      IFS= read -rsn1 key <&3 || key=''
      [[ ${key} == $'\x1b' ]] && { IFS= read -rsn2 -t 0.05 key <&3 || key=''; }
      case "${key}" in
      '[A') [[ ${cur} -gt 0 ]] && cur=$((cur - 1)) ;;
      '[B') [[ ${cur} -lt $((${#names[@]} - 1)) ]] && cur=$((cur + 1)) ;;
      ' ') states[${cur}]=$((1 - states[${cur}])) ;;
      '') break ;;
      esac
      printf '\033[%dA' "${#names[@]}" >&4
    done
  fi
  for i in "${!names[@]}"; do echo "${names[${i}]}=${states[${i}]}"; done
}

# ------------------------------------------------------------------ preflight --
for tool in git cmake curl tar make; do
  command -v "${tool}" >/dev/null || err "'${tool}' is required but not installed."
done
command -v c++ >/dev/null || command -v g++ >/dev/null \
  || warn "No C++ compiler found on PATH -- the build will fail until one is installed."

prebuilt_ok=1
prebuilt_reason=''
case "$(uname -s)-$(uname -m)" in
Linux-x86_64)
  if ! grep -qm1 avx2 /proc/cpuinfo || ! grep -qm1 fma /proc/cpuinfo; then
    prebuilt_ok=0 prebuilt_reason="this CPU lacks AVX2/FMA (x86-64-v3)"
  fi
  ;;
Darwin-arm64) ;; # Apple Silicon bundles (experimental)
*) prebuilt_ok=0 prebuilt_reason="bundles exist for Linux x86_64 and macOS arm64 only" ;;
esac

echo
echo "  ============================================="
echo "   DiFfRG installer"
echo "  ============================================="

# --------------------------------------------------------------------- wizard --
if [[ -z ${mode} ]]; then
  if [[ ${prebuilt_ok} -eq 1 ]]; then
    choose _c "How should DiFfRG's dependencies be installed?" \
      "Pre-built bundle -- download deal.II, Kokkos, Boost, ... (~50 MB, minutes)" \
      "Self-build -- compile the full dependency superbuild (hours, all features)"
    [[ ${_c} -eq 0 ]] && mode=prebuilt || mode=source
  else
    info "Pre-built bundles unavailable: ${prebuilt_reason}. Using self-build."
    mode=source
  fi
fi
[[ ${mode} == prebuilt || ${mode} == source ]] || err "--mode must be 'prebuilt' or 'source'"
[[ ${mode} == prebuilt && ${prebuilt_ok} -eq 0 ]] && err "Pre-built bundles unavailable: ${prebuilt_reason}"

ask prefix "Install prefix" "${prefix}"
ask build_dir "Temporary build folder" "${build_dir}"
case "${prefix}" in /*) ;; *) prefix="$(pwd)/${prefix}" ;; esac
case "${build_dir}" in /*) ;; *) build_dir="$(pwd)/${build_dir}" ;; esac

# With an NVIDIA GPU present, offer the CUDA bundle (Ampere/sm_80 or newer;
# needs the CUDA 12 toolkit installed to build applications).
if [[ ${mode} == prebuilt && -z ${deps_variant} && -z ${deps_file} && "$(uname -s)" == Linux ]] \
  && command -v nvidia-smi >/dev/null 2>&1; then
  choose _c "An NVIDIA GPU was detected -- which bundle?" \
    "CPU bundle -- no GPU support" \
    "CUDA bundle -- GPU-enabled (Ampere/RTX 30xx or newer; requires the CUDA 12 toolkit)"
  [[ ${_c} -eq 1 ]] && deps_variant="linux-x86_64-v3-cuda12"
fi

if [[ ${mode} == prebuilt && -z ${deps_version} && -z ${deps_file} ]]; then
  info "Fetching available dependency bundles..."
  mapfile -t versions < <(curl -fsSL "${REPO_API}/releases?per_page=100" 2>/dev/null |
    grep -oE '"tag_name": *"deps-v[0-9]+\.[0-9]+\.[0-9]+"' |
    grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | sort -rV | head -5)
  if [[ ${#versions[@]} -eq 0 ]]; then
    err "No dependency bundle releases found -- re-run with --mode source, or check your network."
  elif [[ ${#versions[@]} -eq 1 || ${interactive} -eq 0 ]]; then
    deps_version="${versions[0]}"
  else
    choose _c "Which dependency bundle version?" "${versions[@]/#/deps-v}"
    deps_version="${versions[${_c}]}"
  fi
  info "Using dependency bundle deps-v${deps_version}"
fi

if [[ ${mode} == source ]]; then
  [[ ${opt_gpu} -eq 2 ]] && { command -v nvcc >/dev/null && opt_gpu=1 || opt_gpu=0; }
  while IFS='=' read -r name state; do
    case "${name}" in
    "MPI support") opt_mpi=${state} ;;
    "GPU (CUDA)") opt_gpu=${state} ;;
    "PETSc MUMPS solver") opt_mumps=${state} ;;
    "Build documentation") opt_docs=${state} ;;
    esac
  done < <(toggles "Features" \
    "MPI support:${opt_mpi}" \
    "GPU (CUDA):${opt_gpu}" \
    "PETSc MUMPS solver:$([[ ${opt_mumps} -eq 2 ]] && echo "${opt_mpi}" || echo "${opt_mumps}")" \
    "Build documentation:${opt_docs}")
  choose _c "Optimize for which CPU target?" \
    "native -- fastest, this machine only" \
    "x86-64-v3 -- portable to consumer CPUs from ~2013 on" \
    "none -- fully generic (slowest)"
  march=$(sed -n "$((_c + 1))p" <<<$'native\nx86-64-v3\nnone')
fi

if [[ -z ${examples_dir} ]]; then
  choose _c "Copy the examples, tutorials and documentation sources somewhere?" \
    "No -- I'll use the online docs (${DOCS_URL})" \
    "Yes -- choose a folder"
  [[ ${_c} -eq 1 ]] && ask examples_dir "Examples/docs folder" "${HOME}/DiFfRG-examples"
fi

# Mathematica package (flow-equation derivation). Offer it whenever a Wolfram
# installation is detectable; the default destination is the per-user Wolfram
# applications directory.
if [[ ${mathematica_asked} -eq 0 ]]; then
  if [[ "$(uname -s)" == Darwin ]]; then
    wolfram_apps="${HOME}/Library/Mathematica/Applications"
  else
    wolfram_apps="${HOME}/.Wolfram/Applications"
  fi
  if command -v wolframscript >/dev/null 2>&1 || command -v wolfram >/dev/null 2>&1 \
    || [[ -d ${wolfram_apps%/*} ]]; then
    choose _c "Install the DiFfRG Mathematica package (flow-equation derivation)?" \
      "Yes -- into the Wolfram applications directory" \
      "Yes -- choose a folder" \
      "No"
    case ${_c} in
    0) mathematica_dir="${wolfram_apps}" ;;
    1) ask mathematica_dir "Mathematica package folder" "${wolfram_apps}" ;;
    esac
  fi
fi
ask threads "Build threads" "${threads}"
[[ ${threads} =~ ^[0-9]+$ ]] || err "Threads must be a number."

# -------------------------------------------------------------------- summary --
echo
echo "  ---------------------------------------------"
echo "   Mode:            ${mode}"
[[ ${mode} == prebuilt ]] && echo "   Bundle:          ${deps_file:-deps-v${deps_version}} (${deps_variant:-platform default})"
if [[ ${mode} == source ]]; then
  echo "   MPI:             $([[ ${opt_mpi} -eq 1 ]] && echo on || echo off)"
  echo "   GPU (CUDA):      $([[ ${opt_gpu} -eq 1 ]] && echo on || echo off)"
  echo "   MUMPS:           $([[ ${opt_mumps} -eq 1 ]] && echo on || echo off)"
  echo "   Documentation:   $([[ ${opt_docs} -eq 1 ]] && echo on || echo off)"
  echo "   CPU target:      ${march}"
fi
echo "   Install prefix:  ${prefix}"
echo "   Build folder:    ${build_dir}"
echo "   Examples/docs:   ${examples_dir:-not copied}"
echo "   Mathematica:     ${mathematica_dir:-not installed}"
echo "   Threads:         ${threads}"
echo "  ---------------------------------------------"
if [[ ${interactive} -eq 1 ]]; then
  printf '\033[1mProceed?\033[0m [Y/n]: ' >&4
  IFS= read -r _go <&3 || _go=''
  [[ -z ${_go} || ${_go} == y || ${_go} == Y ]] || { echo "Aborted."; exit 0; }
fi

# ------------------------------------------------------------------ checkout --
mkdir -p "${build_dir}"
src="${build_dir}/DiFfRG_current"
if [[ -d ${src}/.git ]]; then
  info "Updating existing checkout in ${src}..."
  git -C "${src}" pull --ff-only || warn "Could not update ${src}; using it as-is."
else
  info "Cloning DiFfRG into ${src}..."
  git clone --depth 1 "${REPO_URL}" "${src}"
fi

# --------------------------------------------------------------------- build --
if [[ ${mode} == prebuilt ]]; then
  # Replacing an existing bundle needs explicit consent: ask when interactive,
  # otherwise require --force.
  force_arg=''
  if [[ -e ${prefix}/bundled ]]; then
    if [[ ${force} -eq 1 ]]; then
      force_arg='--force'
    elif [[ ${interactive} -eq 1 ]]; then
      printf '\033[1m%s/bundled already exists. Replace it?\033[0m [y/N]: ' "${prefix}" >&4
      IFS= read -r _rep <&3 || _rep=''
      [[ ${_rep} == y || ${_rep} == Y ]] || err "Keeping the existing bundle. Re-run with another --prefix, or --force to replace."
      force_arg='--force'
    else
      err "${prefix}/bundled already exists. Pass --force to replace it."
    fi
  fi

  info "Installing the pre-built dependency bundle..."
  variant_arg=''
  [[ -n ${deps_variant} ]] && variant_arg="--variant ${deps_variant}"
  if [[ -n ${deps_file} ]]; then
    bash "${src}/install-diffrg-deps.sh" --file "${deps_file}" --prefix "${prefix}" ${variant_arg} ${force_arg}
  else
    bash "${src}/install-diffrg-deps.sh" --version "${deps_version}" --prefix "${prefix}" ${variant_arg} ${force_arg}
  fi

  info "Building the DiFfRG library against the bundle..."
  # CMAKE_INSTALL_LIBDIR=lib keeps the library's cmake config at
  # <prefix>/lib/cmake/DiFfRG on every distro (EL-family GNUInstallDirs would
  # otherwise choose lib64), matching the find_package hint printed below.
  cmake -S "${src}/DiFfRG" -B "${build_dir}/library-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUNDLED_DIR="${prefix}/bundled" \
    -DCMAKE_INSTALL_PREFIX="${prefix}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    ${mathematica_dir:+-DDiFfRG_MATHEMATICA_INSTALL_DIR="${mathematica_dir}"} \
    -DDiFfRG_DOCUMENTATION="$([[ ${opt_docs} -eq 1 ]] && echo ON || echo OFF)"
  cmake --build "${build_dir}/library-build" -j "${threads}"
  cmake --install "${build_dir}/library-build"
else
  info "Configuring the full superbuild (this will take a while)..."
  [[ ${opt_mumps} -eq 2 ]] && opt_mumps=${opt_mpi}
  # BUILD_JOBS is a core-count hint the superbuild halves for its sub-builds;
  # 2x threads makes those sub-builds use exactly ${threads} jobs.
  cmake -S "${src}" -B "${build_dir}/superbuild" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${prefix}" \
    -DMPI="$([[ ${opt_mpi} -eq 1 ]] && echo ON || echo OFF)" \
    -DGPU="$([[ ${opt_gpu} -eq 1 ]] && echo ON || echo OFF)" \
    -DPETSC_MUMPS="$([[ ${opt_mumps} -eq 1 ]] && echo ON || echo OFF)" \
    -DDiFfRG_DOCUMENTATION="$([[ ${opt_docs} -eq 1 ]] && echo ON || echo OFF)" \
    -DMARCH="${march}" \
    ${mathematica_dir:+-DDiFfRG_MATHEMATICA_INSTALL_DIR="${mathematica_dir}"} \
    -DBUILD_JOBS=$((2 * threads)) \
    -DDEALII_MAX_JOBS="${threads}" \
    -DPETSC_MAX_JOBS="${threads}"
  info "Building (logs of individual dependencies are in ${build_dir}/superbuild)..."
  cmake --build "${build_dir}/superbuild" -- -j "${threads}"
fi

# ------------------------------------------------------------ examples & docs --
if [[ -n ${examples_dir} ]]; then
  info "Copying examples, tutorials and documentation sources to ${examples_dir}..."
  mkdir -p "${examples_dir}"
  cp -a "${src}/Examples" "${examples_dir}/Examples"
  cp -a "${src}/Tutorials" "${examples_dir}/Tutorials"
  cp -a "${src}/DiFfRG/documentation" "${examples_dir}/documentation"
fi

# -------------------------------------------------------------------- verify --
info "Verifying the installation..."
cmake -DBUNDLED_DIR="${prefix}/bundled" -P "${prefix}/cmake/verify_install.cmake" \
  || err "Verification failed -- see output above."

echo
info "DiFfRG is installed at ${prefix}."
echo "  Use it from your own project's CMakeLists.txt with:"
echo "      find_package(DiFfRG REQUIRED HINTS ${prefix}/lib/cmake/DiFfRG)"
[[ -n ${examples_dir} ]] && echo "  Examples and tutorials: ${examples_dir}"
echo "  Documentation: ${DOCS_URL}"
echo "  The build folder ${build_dir} can be deleted to reclaim space"
echo "  (keep it to speed up future updates)."
