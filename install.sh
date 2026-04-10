#!/bin/bash
set -euo pipefail

# ==============================================================================
# DiFfRG Installation Script
# ==============================================================================

print_usage() {
  cat <<'USAGE'
Usage: [THREADS=N] [FOLDER=path] bash install.sh [--help]

Install DiFfRG and all dependencies from source.

Environment variables:
  THREADS   Number of build threads (default: half of available cores, max 8)
  FOLDER    Installation directory (default: $HOME/.local/share/DiFfRG)

Examples:
  bash install.sh
  THREADS=4 FOLDER=/opt/DiFfRG bash install.sh
  bash <(curl -sL https://github.com/satfra/DiFfRG_current/raw/refs/heads/main/install.sh)

Report issues: https://github.com/satfra/DiFfRG/issues
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

# ==============================================================================
# Pre-flight checks
# ==============================================================================

check_command() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is required but not found."
    echo "  Install hint: $hint"
    echo ""
    return 1
  fi
}

preflight_ok=true

check_command git      "apt install git  |  pacman -S git  |  dnf install git  |  brew install git"              || preflight_ok=false
check_command cmake    "apt install cmake  |  pacman -S cmake  |  dnf install cmake  |  brew install cmake"    || preflight_ok=false
check_command make     "apt install build-essential  |  pacman -S make  |  dnf install make  |  brew install make" || preflight_ok=false

# Check for a C++ compiler (g++ or clang++)
if ! command -v g++ &>/dev/null && ! command -v clang++ &>/dev/null; then
  echo "ERROR: A C++ compiler (g++ or clang++) is required but not found."
  echo "  Install hint: apt install build-essential | pacman -S gcc | dnf install gcc-c++ | brew install gcc"
  echo ""
  preflight_ok=false
fi

# Check for a Fortran compiler (gfortran or flang)
if ! command -v gfortran &>/dev/null && ! command -v flang &>/dev/null; then
  echo "ERROR: A Fortran compiler (gfortran or flang) is required but not found."
  echo "  Install hint: apt install gfortran | pacman -S gcc-fortran | dnf install gcc-gfortran | brew install gcc"
  echo ""
  preflight_ok=false
fi

# Check for Python (python3 or python)
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo "ERROR: Python (python3 or python) is required but not found."
  echo "  Install hint: apt install python3 | pacman -S python | dnf install python3 | brew install python3"
  echo ""
  preflight_ok=false
fi

if [[ "$preflight_ok" != "true" ]]; then
  echo "Please install the missing tools above and re-run this script."
  exit 1
fi

# Check for GSL (library, not a command)
gsl_found=false
if pkg-config --exists gsl 2>/dev/null; then
  gsl_found=true
elif ldconfig -p 2>/dev/null | grep -q libgsl; then
  gsl_found=true
elif [[ -f /usr/local/lib/libgsl.dylib || -f /opt/homebrew/lib/libgsl.dylib ]]; then
  # macOS Homebrew paths
  gsl_found=true
fi
if [[ "$gsl_found" != "true" ]]; then
  echo "WARNING: GSL (GNU Scientific Library) not detected."
  echo "  Install: sudo apt install libgsl-dev | pacman -S gsl | dnf install gsl-devel | brew install gsl"
  echo ""
fi

# ==============================================================================
# Configuration
# ==============================================================================

tempFolder="/tmp/"
repoName="DiFfRG_current"
repo="https://github.com/satfra/${repoName}.git"

if [[ "${FOLDER+set}" == "set" ]]; then
  installFolder="${FOLDER}"
else
  installFolder="${HOME}/.local/share/DiFfRG"
fi
echo "Installation directory set to ${installFolder}"

# Determine number of build threads
if [[ "${THREADS+set}" == "set" ]]; then
  bcores="$THREADS"
else
  ncores="$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  bcores=$(( ncores / 2 ))
  if (( bcores < 1 )); then bcores=1; fi
  if (( bcores > 8 )); then bcores=8; fi
fi

# Cap threads by available RAM (~2GB per thread)
mem_kb=0
if [[ -f /proc/meminfo ]]; then
  mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
elif command -v sysctl &>/dev/null; then
  mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
  mem_kb=$(( mem_bytes / 1024 ))
fi
if (( mem_kb > 0 )); then
  max_by_ram=$(( mem_kb / 2000000 ))
  if (( max_by_ram < 1 )); then max_by_ram=1; fi
  if (( max_by_ram < bcores )); then
    bcores=$max_by_ram
    echo "Limiting to ${bcores} threads due to available RAM ($(( mem_kb / 1024 / 1024 ))GB)."
  fi
fi

echo "Will build with ${bcores} threads."

# ==============================================================================
# Clone and build
# ==============================================================================

echo "Preparing source in ${tempFolder}${repoName}..."
if [[ -d "${tempFolder}${repoName}" ]]; then
  echo "Source directory exists from previous attempt. Updating..."
  if ! (cd "${tempFolder}${repoName}" && git fetch --depth 1 origin && git reset --hard origin/HEAD) 2>/dev/null; then
    echo "  Could not update existing source. Removing and re-cloning..."
    rm -rf "${tempFolder}${repoName}"
  fi
fi
if [[ ! -d "${tempFolder}${repoName}" ]]; then
  cd "${tempFolder}"
  git clone "${repo}" "${tempFolder}${repoName}" --depth 1
fi

echo "Running CMake in ${tempFolder}${repoName}/build..."
mkdir -p "${tempFolder}${repoName}/build"
cd "${tempFolder}${repoName}/build"
cmake "${tempFolder}${repoName}" -DCMAKE_INSTALL_PREFIX="${installFolder}"

echo "Building DiFfRG with ${bcores} threads..."
if ! make -j"${bcores}"; then
  echo ""
  echo "========================================"
  echo "  BUILD FAILED"
  echo "========================================"
  echo "Check the build log above for errors."
  echo "Common issues:"
  echo "  - Missing system dependency (GSL, LAPACK/BLAS)"
  echo "  - Insufficient RAM (try reducing THREADS)"
  echo "  - Incompatible compiler version (especially with CUDA)"
  echo ""
  echo "Report issues: https://github.com/satfra/DiFfRG/issues"
  exit 1
fi

# ==============================================================================
# Verify installation
# ==============================================================================

config_file="${installFolder}/lib/cmake/DiFfRG/DiFfRGConfig.cmake"
if [[ -f "$config_file" ]]; then
  echo ""
  echo "========================================"
  echo "  DiFfRG installed successfully!"
  echo "========================================"
  echo "  Location: ${installFolder}"
  echo ""
  echo "To use DiFfRG in your project, add to your CMakeLists.txt:"
  echo "  find_package(DiFfRG REQUIRED HINTS ${installFolder})"
  echo ""

  # Run verification script if available
  verify_script="${installFolder}/cmake/verify_install.cmake"
  if [[ -f "$verify_script" ]]; then
    echo "Running dependency verification..."
    cmake -DBUNDLED_DIR="${installFolder}/bundled" -P "$verify_script" || true
    echo ""
  fi
else
  echo ""
  echo "========================================"
  echo "  WARNING: Installation may be incomplete"
  echo "========================================"
  echo "  Expected config file not found:"
  echo "    ${config_file}"
  echo ""
  echo "  The build may have succeeded but installation failed."
  echo "  Try re-running: cd ${tempFolder}${repoName}/build && make install"
  echo ""
  echo "Report issues: https://github.com/satfra/DiFfRG/issues"
  exit 1
fi
