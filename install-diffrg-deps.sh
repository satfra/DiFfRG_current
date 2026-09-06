#!/bin/bash
# ##############################################################################
# Install a pre-built DiFfRG dependency bundle.
#
# Downloads a relocatable binary bundle of DiFfRG's dependencies (deal.II,
# Kokkos, Boost, TBB, SUNDIALS, HDF5, ...) from GitHub Releases and installs it
# so the multi-hour dependency superbuild is skipped entirely. The DiFfRG
# library itself is then built from source against the bundle (minutes, not
# hours) -- either by hand or with --build-library.
#
# Usage:
#   bash <(curl -sL https://raw.githubusercontent.com/satfra/DiFfRG_current/main/install-diffrg-deps.sh) [options]
#
# Options:
#   --prefix DIR      install prefix (default: $HOME/.local/share/DiFfRG; env FOLDER)
#   --version X.Y.Z   bundle version (default: latest deps-v* release)
#   --variant NAME    bundle variant (default: auto-detected, linux-x86_64-v3-cpu)
#   --file TARBALL    install from a local tarball instead of downloading
#   --force           replace an existing <prefix>/bundled tree
#   --skip-cpu-check  do not gate on x86-64-v3 CPU support (the binaries will
#                     still SIGILL on unsupported CPUs when run)
#   --build-library   also configure+build+install the DiFfRG library (requires
#                     running inside a DiFfRG checkout)
#   -h, --help        this text
#
# For CPUs without AVX2 (pre-2013), CUDA builds, MPI builds, or any other
# configuration the binary bundles do not cover, use the source installer
# instead: install.sh in the same repository.
# ##############################################################################
set -euo pipefail

usage() {
  cat <<'EOF'
Install a pre-built DiFfRG dependency bundle from GitHub Releases, so the
multi-hour dependency superbuild is skipped; the DiFfRG library itself is then
built from source against the bundle (minutes).

Usage:
  bash <(curl -sL https://raw.githubusercontent.com/satfra/DiFfRG_current/main/install-diffrg-deps.sh) [options]

Options:
  --prefix DIR      install prefix (default: $HOME/.local/share/DiFfRG; env FOLDER)
  --version X.Y.Z   bundle version (default: latest deps-v* release)
  --variant NAME    bundle variant (default: auto-detected, linux-x86_64-v3-cpu)
  --file TARBALL    install from a local tarball instead of downloading
  --force           replace an existing <prefix>/bundled tree
  --skip-cpu-check  do not gate on x86-64-v3 CPU support
  --build-library   also configure+build+install the DiFfRG library (requires
                    running inside a DiFfRG checkout)
  -h, --help        this text

For CPUs without AVX2 (pre-2013), CUDA builds, MPI builds, or any other
configuration the binary bundles do not cover, use the source installer
instead: install.sh in the same repository.
EOF
}

REPO="satfra/DiFfRG_current"
BUILD_PREFIX="/opt/diffrg" # canonical prefix baked into release bundles

os_name="$(uname -s)"
case "${os_name}-$(uname -m)" in
Linux-x86_64) DEFAULT_VARIANT="linux-x86_64-v3-cpu" ;;
Darwin-arm64) DEFAULT_VARIANT="macos-arm64-cpu" ;;
*) DEFAULT_VARIANT="" ;;
esac

# Portability helpers: macOS ships BSD sed (needs -i '') and shasum instead of
# sha256sum.
sed_inplace() {
  if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi
}
sha256_verify() { # <checksum-file> (run from its directory)
  if command -v sha256sum >/dev/null; then sha256sum -c "$1" >/dev/null
  else shasum -a 256 -c "$1" >/dev/null; fi
}

prefix="${FOLDER:-$HOME/.local/share/DiFfRG}"
version=''
variant=''
tarball=''
force=0
skip_cpu_check=0
build_library=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  --prefix) prefix="$2"; shift 2 ;;
  --version) version="$2"; shift 2 ;;
  --variant) variant="$2"; shift 2 ;;
  --file) tarball="$2"; shift 2 ;;
  --force) force=1; shift ;;
  --skip-cpu-check) skip_cpu_check=1; shift ;;
  --build-library) build_library=1; shift ;;
  -h | --help) usage; exit 0 ;;
  *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

err() { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }
warn() { echo -e "\033[1;33mWARNING:\033[0m $*" >&2; }
info() { echo -e "\033[1;32m==>\033[0m $*"; }

# '#' delimits the relocation sed, '&' and '\' are special in its replacement.
case "$prefix" in
*'#'* | *'&'* | *'\'* | *$'\n'*) err "Install prefix must not contain '#', '&', '\\' or newlines: $prefix" ;;
/*) ;;
*) prefix="$(pwd)/$prefix" ;;
esac

# ------------------------------------------------------------ platform gate --
[[ -n $DEFAULT_VARIANT ]] \
  || err "Pre-built bundles exist for Linux x86_64 and macOS arm64 only ($(uname -s) $(uname -m) detected). Use install.sh to build from source."

[[ -z $variant ]] && variant="$DEFAULT_VARIANT"
if [[ $skip_cpu_check -eq 0 && ${os_name} == Linux ]]; then
  # The v3 bundles need AVX2/FMA (any consumer CPU from ~2013 on). Prefer the
  # authoritative glibc probe; fall back to /proc/cpuinfo flags.
  if ld.so --help 2>/dev/null | grep -q 'x86-64-v3'; then
    ld.so --help 2>/dev/null | grep 'x86-64-v3' | grep -q supported \
      || err "This CPU does not support x86-64-v3 (AVX2+FMA), which the pre-built bundles require.
Use the source installer instead: install.sh"
  elif ! grep -qm1 avx2 /proc/cpuinfo || ! grep -qm1 fma /proc/cpuinfo; then
    err "This CPU does not support x86-64-v3 (AVX2+FMA), which the pre-built bundles require.
Use the source installer instead: install.sh"
  fi
fi

# curl/sha256sum are only needed for the download path; a local --file install
# (e.g. air-gapped) works without them.
required_tools=(tar sed grep cmake)
[[ -z $tarball ]] && required_tools+=(curl sha256sum)
for tool in "${required_tools[@]}"; do
  command -v "$tool" >/dev/null || err "'$tool' is required but not installed."
done
tar --zstd --version >/dev/null 2>&1 || command -v zstd >/dev/null \
  || err "zstd is required to unpack the bundle (install the 'zstd' package)."

# ----------------------------------------------------------------- download --
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

if [[ -z $tarball ]]; then
  if [[ -z $version ]]; then
    info "Looking up the latest dependency release..."
    # The || true keeps set -e/pipefail from aborting before the friendly
    # error below when the API is unreachable or no release matches.
    version="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases?per_page=100" 2>/dev/null |
      grep -oE '"tag_name": *"deps-v[0-9]+\.[0-9]+\.[0-9]+"' |
      grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | sort -V | tail -1 || true)"
    [[ -n $version ]] || err "No deps-v* release found on github.com/${REPO}."
  fi
  name="diffrg-deps-${version}-${variant}"
  url="https://github.com/${REPO}/releases/download/deps-v${version}/${name}.tar.zst"
  info "Downloading ${name}.tar.zst..."
  curl -fL --progress-bar -o "$workdir/${name}.tar.zst" "$url" \
    || err "Download failed: $url
Check that release deps-v${version} provides the '${variant}' variant."
  curl -fsSL -o "$workdir/${name}.tar.zst.sha256" "${url}.sha256" \
    || err "Checksum download failed: ${url}.sha256"
  (cd "$workdir" && sha256_verify "${name}.tar.zst.sha256") \
    || err "Checksum verification FAILED for ${name}.tar.zst -- corrupted download?"
  info "Checksum verified."
  tarball="$workdir/${name}.tar.zst"
else
  [[ -f $tarball ]] || err "No such file: $tarball"
  info "Installing from local tarball $tarball"
fi

# ------------------------------------------------------------------- unpack --
info "Unpacking..."
tar --zstd -xf "$tarball" -C "$workdir" \
  || zstd -dc "$tarball" | tar -xf - -C "$workdir"
srcdir="$(find "$workdir" -maxdepth 1 -type d -name 'diffrg-deps-*' | head -1)"
[[ -n $srcdir && -d $srcdir/bundled ]] || err "Tarball does not look like a DiFfRG dependency bundle."

manifest="$srcdir/bundled/BUNDLE_MANIFEST.json"
[[ -f $manifest ]] || err "Bundle has no BUNDLE_MANIFEST.json."
if [[ ${os_name} == Linux ]]; then
  glibc_floor="$(grep -oE '"glibc_floor": *"[0-9.]+"' "$manifest" | grep -oE '[0-9.]+' || echo '')"
  if [[ -n $glibc_floor ]]; then
    glibc_have="$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo '')"
    if [[ -n $glibc_have && "$(printf '%s\n%s\n' "$glibc_floor" "$glibc_have" | sort -V | head -1)" != "$glibc_floor" ]]; then
      err "This system's glibc ($glibc_have) is older than the bundle requires ($glibc_floor).
Use the source installer instead: install.sh"
    fi
  fi
else
  min_macos="$(grep -oE '"min_macos": *"[0-9.]+"' "$manifest" | grep -oE '[0-9.]+' || echo '')"
  if [[ -n $min_macos ]]; then
    macos_have="$(sw_vers -productVersion 2>/dev/null || echo '')"
    if [[ -n $macos_have && "$(printf '%s\n%s\n' "$min_macos" "$macos_have" | sort -V | head -1)" != "$min_macos" ]]; then
      err "This macOS ($macos_have) is older than the bundle requires ($min_macos).
Use the source installer instead: install.sh"
    fi
  fi
fi

# ------------------------------------------------------------------ install --
if [[ -e "$prefix/bundled" ]]; then
  if [[ $force -eq 1 ]]; then
    info "Removing existing $prefix/bundled (--force)..."
    rm -rf "$prefix/bundled"
  else
    err "$prefix/bundled already exists. Re-run with --force to replace it,
or choose another prefix with --prefix."
  fi
fi
mkdir -p "$prefix"
mv "$srcdir/bundled" "$prefix/bundled"
info "Installed bundle to $prefix/bundled"

# -------------------------------------------------------------------- fixup --
# The bundle was built with prefix /opt/diffrg; rewrite every text-file
# occurrence to the actual install prefix. (Shared-library RUNPATHs are already
# $ORIGIN-relative -- no binary patching needed.)
info "Relocating bundle to $prefix..."
# Process substitution, not a pipe: grep exits 1 when nothing needs rewriting,
# which under pipefail would abort the install halfway through. (No mapfile:
# macOS ships bash 3.2.)
while IFS= read -r f; do
  [[ -n $f ]] || continue
  sed_inplace "s#${BUILD_PREFIX}/bundled#${prefix}/bundled#g; s#${BUILD_PREFIX}#${prefix}#g" "$f"
done < <(grep -rIl --exclude='*.log' "$BUILD_PREFIX" "$prefix/bundled" 2>/dev/null || true)

# On macOS the dylibs carry the build prefix in their install names and
# cross-references; rewrite those too and ad-hoc re-sign (arm64 macOS refuses
# to load a modified, unsigned binary).
if [[ ${os_name} == Darwin ]]; then
  find "$prefix/bundled/lib" -name '*.dylib' -type f | while IFS= read -r dylib; do
    install_name_tool -id "${dylib}" "${dylib}" 2>/dev/null || true
    otool -L "$dylib" | awk 'NR>1{print $1}' | grep "^${BUILD_PREFIX}" |
      while IFS= read -r dep; do
        install_name_tool -change "$dep" "${dep/#${BUILD_PREFIX}/${prefix}}" "$dylib"
      done || true
    codesign --force -s - "$dylib" 2>/dev/null \
      || warn "could not re-sign $dylib -- it may fail to load"
  done
fi

# deal.II records the compilers of the build container; point the records at
# this machine's toolchain instead (DiFfRG's own build ignores them, but
# deal.II-style consumer projects and tooling read them).
dealii_config="$prefix/bundled/lib/cmake/deal.II/deal.IIConfig.cmake"
if [[ -f $dealii_config ]]; then
  for pair in "CXX:c++" "C:cc" "Fortran:gfortran"; do
    lang="${pair%%:*}"
    comp="$(command -v "${pair##*:}" || true)"
    [[ -n $comp ]] && sed -i -E "s#^set\(DEAL_II_${lang}_COMPILER \"[^\"]*\"\)#set(DEAL_II_${lang}_COMPILER \"${comp}\")#" "$dealii_config"
  done
fi

cat > "$prefix/bundled/INSTALL_RECEIPT.json" <<EOF
{
  "installed_to": "${prefix}",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "${tarball##*/}"
}
EOF

# -------------------------------------------------------------- host checks --
info "Checking host prerequisites for building DiFfRG against the bundle..."
missing=()

cxx_probe="$workdir/probe.cpp"
echo 'int main() { return 0; }' > "$cxx_probe"
if ! command -v c++ >/dev/null || ! c++ -std=c++20 -o "$workdir/probe" "$cxx_probe" 2>/dev/null; then
  missing+=("a C++20 compiler (GCC >= 12: package 'gcc-c++'/'g++' or 'gcc-toolset-14' on EL9; Xcode CLT on macOS)")
fi
if [[ ${os_name} == Linux ]]; then
  ldconfig -p 2>/dev/null | grep -qE 'libopenblas|liblapack' \
    || missing+=("BLAS/LAPACK (package 'libopenblas-dev' / 'openblas-devel')")
  { command -v pkg-config >/dev/null && pkg-config --exists gsl; } || [[ -e /usr/include/gsl/gsl_math.h ]] \
    || missing+=("GSL headers (package 'libgsl-dev' / 'gsl-devel')")
  [[ -e /usr/include/zlib.h ]] \
    || missing+=("zlib headers (package 'zlib1g-dev' / 'zlib-devel')")
else
  # macOS: BLAS/LAPACK come from the always-present Accelerate framework and
  # zlib from the SDK; only GSL and gfortran come from Homebrew.
  { command -v pkg-config >/dev/null && pkg-config --exists gsl; } || [[ -e /opt/homebrew/include/gsl/gsl_math.h ]] \
    || missing+=("GSL (brew install gsl)")
fi
command -v gfortran >/dev/null \
  || missing+=("gfortran (package 'gfortran' / 'gcc-gfortran'; 'brew install gcc' on macOS)")

if [[ ${#missing[@]} -gt 0 ]]; then
  warn "The bundle is installed, but building DiFfRG will additionally need:"
  for m in "${missing[@]}"; do echo "  - $m" >&2; done
fi

# -------------------------------------------------------------------- verify --
info "Verifying the installed bundle..."
cmake -DBUNDLED_DIR="$prefix/bundled" -P "$prefix/bundled/share/DiFfRG/verify_install.cmake" \
  || err "Bundle verification failed -- see output above."

# ------------------------------------------------------------- next steps --
if [[ $build_library -eq 1 ]]; then
  [[ -f DiFfRG/CMakeLists.txt ]] \
    || err "--build-library requires running from a DiFfRG checkout (DiFfRG/CMakeLists.txt not found)."
  threads="${THREADS:-6}"
  info "Building the DiFfRG library against the bundle (${threads} threads)..."
  cmake -S DiFfRG -B "$prefix/library-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUNDLED_DIR="$prefix/bundled" \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DDiFfRG_DOCUMENTATION=OFF
  cmake --build "$prefix/library-build" -j "$threads"
  cmake --install "$prefix/library-build"
  info "DiFfRG installed to $prefix"
else
  echo
  info "Done. Build the DiFfRG library against the bundle with:"
  echo "    cmake -S <DiFfRG-checkout>/DiFfRG -B build \\"
  echo "        -DCMAKE_BUILD_TYPE=Release \\"
  echo "        -DBUNDLED_DIR=$prefix/bundled \\"
  echo "        -DCMAKE_INSTALL_PREFIX=$prefix"
  echo "    cmake --build build -j 6 && cmake --install build"
  if [[ ${os_name} == Linux ]]; then
    echo
    echo "  The default -march=native is safe on this machine (its CPU is a superset"
    echo "  of the bundle's x86-64-v3 baseline)."
  fi
fi
