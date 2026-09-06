#!/usr/bin/env bash
# Post-process a freshly built dependency bundle into a relocatable state and
# audit it. Runs inside the release builder container (see the Dockerfiles in
# this directory); every check is a hard failure -- a bundle that does not pass
# is not shipped.
#
# Usage: postprocess-bundle.sh <bundle-dir> <repo-src> <bundle-version> <variant> <march> <glibc-floor>
#   bundle-dir     the installed dependency tree (e.g. /opt/diffrg/bundled)
#   repo-src       the DiFfRG repository checkout (for verify_install.cmake, git SHA)
#   bundle-version e.g. 1.0.0
#   variant        e.g. linux-x86_64-v3-cpu
#   march          the -march value the bundle was compiled for (e.g. x86-64-v3)
#   glibc-floor    maximum allowed GLIBC_* symbol version (e.g. 2.34)
set -euo pipefail

BUNDLE="$1"
SRC="$2"
BUNDLE_VERSION="$3"
VARIANT="$4"
MARCH="$5"
GLIBC_FLOOR="$6"

# libstdc++ ceilings of the el9 baseline (gcc 11). gcc-toolset links newer
# symbols from its static libstdc++_nonshared.a, so nothing above these may
# leak into the shared objects' undefined symbols.
GLIBCXX_CEIL="3.4.29"
CXXABI_CEIL="1.3.13"

HERE="$(cd "$(dirname "$0")" && pwd)"

fail() { echo "postprocess: FAIL: $*" >&2; exit 1; }
note() { echo "postprocess: $*"; }

[ -d "$BUNDLE/lib" ] || fail "$BUNDLE/lib does not exist"

# The superbuild pins CMAKE_INSTALL_LIBDIR=lib, so lib64/ should not exist;
# every object pass still covers it defensively in case a dependency ever
# ignores the pin again.
LIBDIRS=("$BUNDLE/lib")
[ -d "$BUNDLE/lib64" ] && LIBDIRS+=("$BUNDLE/lib64")

# ---------------------------------------------------------------- 1. verify --
# Ship the standalone dependency checker inside the bundle so the installer
# can run it without a DiFfRG checkout.
mkdir -p "$BUNDLE/share/DiFfRG"
cp "$SRC/DiFfRG/cmake/verify_install.cmake" "$BUNDLE/share/DiFfRG/verify_install.cmake"

# ------------------------------------------- 2. system lib paths -> -l flags --
# deal.II exports the build machine's system libraries as absolute paths in its
# CMake config; those differ per distro (lib64 vs multiarch), so rewrite them to
# plain -l flags the consumer's linker resolves from default directories.
[ -f "$BUNDLE/lib/cmake/deal.II/deal.IITargets.cmake" ] \
  || fail "deal.IITargets.cmake not found -- bundle layout changed?"
for f in "$BUNDLE"/lib/cmake/deal.II/deal.IITargets.cmake \
         "$BUNDLE"/lib/cmake/deal.II/deal.IIConfig.cmake; do
  [ -f "$f" ] || continue
  sed -i -E 's#/usr/(lib64|lib)(/[A-Za-z0-9_.-]+-linux-gnu[A-Za-z0-9_.-]*)?/lib([A-Za-z0-9_+.-]+)\.(so|a)#-l\3#g' "$f"
done
if grep -nE '/usr/(lib64|lib)[^;"]*\.(so|a)' "$BUNDLE"/lib/cmake/deal.II/deal.II*.cmake; then
  fail "absolute system library paths survived the -l rewrite (above)"
fi

# ------------------------------------------------------------------ 3. strip --
find "${LIBDIRS[@]}" -name '*.so*' -type f -exec strip --strip-unneeded {} +
find "${LIBDIRS[@]}" -name '*.a' -type f -exec strip -g {} +

# ------------------------------------------------------------------ 4. rpath --
# Make every shared object find its siblings relative to itself, whichever of
# lib/ and lib64/ each side lives in; after this the tarball needs no patchelf
# at install time.
RPATH='$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64'
while IFS= read -r so; do
  file -b "$so" | grep -q ELF || continue
  patchelf --set-rpath "$RPATH" "$so"
  readelf -d "$so" | grep -q 'RUNPATH.*\$ORIGIN' \
    || fail "RUNPATH not set on $so"
done < <(find "${LIBDIRS[@]}" -name '*.so*' -type f)

# ------------------------------------------------------------------ 5. prune --
rm -rf "$BUNDLE/share/doc" "$BUNDLE/share/man"
find "$BUNDLE" -name '*.log' ! -name 'detailed.log' ! -name 'summary.log' -delete

# -------------------------------------------------------------- 6. ISA guard --
"$HERE/check-isa.sh" "$BUNDLE" "$MARCH"

# ------------------------------------------------------- 7. symbol versions --
# No undefined symbol may require a glibc newer than the floor or a libstdc++
# newer than the el9 baseline -- either would break the "runs on any distro
# with glibc >= floor" promise.
ver_gt() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" != "$2" ]; }
while IFS= read -r so; do
  file -b "$so" | grep -q ELF || continue
  while IFS= read -r ver; do
    case "$ver" in
      GLIBC_*)   ver_gt "${ver#GLIBC_}" "$GLIBC_FLOOR"   && fail "$so needs $ver (> $GLIBC_FLOOR)" ;;
      GLIBCXX_*) ver_gt "${ver#GLIBCXX_}" "$GLIBCXX_CEIL" && fail "$so needs $ver (> $GLIBCXX_CEIL)" ;;
      CXXABI_*)  ver_gt "${ver#CXXABI_}" "$CXXABI_CEIL"  && fail "$so needs $ver (> $CXXABI_CEIL)" ;;
    esac
  done < <(objdump -T "$so" 2>/dev/null | grep -oE '(GLIBC|GLIBCXX|CXXABI)_[0-9.]+' | sort -uV)
done < <(find "${LIBDIRS[@]}" -name '*.so*' -type f)
note "symbol versions within GLIBC<=$GLIBC_FLOOR GLIBCXX<=$GLIBCXX_CEIL CXXABI<=$CXXABI_CEIL"

# -------------------------------------------------- 8. residual path audit --
# Configuration files may reference: the canonical build prefix (rewritten by
# the installer), standard /usr paths (headers, -l search dirs), and the
# toolset compiler records in deal.IIConfig.cmake (also rewritten by the
# installer). An absolute path into the build container's scratch dirs is a
# leak that would break consumers. Scanned in config files only -- source
# headers legitimately contain substrings like "Eigen/src/".
# (the kept deal.II summary/detailed.log are provenance and exempt)
mapfile -t _cfg_files < <(find "$BUNDLE/lib/cmake" "$BUNDLE/lib/pkgconfig" \
  "$BUNDLE/lib64/cmake" "$BUNDLE/lib64/pkgconfig" "$BUNDLE/cmake" "$BUNDLE/bin" \
  "$BUNDLE/share/DiFfRG" -type f 2>/dev/null; find "$BUNDLE" -maxdepth 1 -type f ! -name '*.log')
if grep -lE '(^|["=[:space:];:])(/root|/home|/build|/src)/' "${_cfg_files[@]}" 2>/dev/null; then
  fail "configuration files reference build-time paths (above)"
fi
note "no stray build-time paths in configuration files"

# --------------------------------------------------------------- 9. manifest --
# Dependency versions straight from the installed CMake package metadata.
dep_versions() {
  local first=1
  for cv in "$BUNDLE"/lib{,64}/cmake/*/*[Cc]onfig[Vv]ersion.cmake \
            "$BUNDLE"/lib{,64}/cmake/*/*config-version.cmake; do
    [ -f "$cv" ] || continue
    local name ver
    name="$(basename "$(dirname "$cv")")"
    case "$name" in boost_*) continue ;; esac # per-component Boost dirs, top-level Boost suffices
    ver="$(grep -m1 -oE 'PACKAGE_VERSION "?[0-9][0-9.]*"?' "$cv" | grep -oE '[0-9][0-9.]*' || true)"
    [ -n "$ver" ] || continue
    [ $first -eq 1 ] || printf ',\n'
    first=0
    printf '    "%s": "%s"' "$name" "$ver"
  done
  printf '\n'
}

# The accelerator is the variant's last component (cpu, cuda12, ...); CUDA
# variants additionally rely on the host's CUDA runtime and driver.
ACCEL="${VARIANT##*-}"
CUDA_ALLOW=''
if [ "$ACCEL" != cpu ]; then
  CUDA_ALLOW='"libcudart", "libcuda", "libnvrtc", '
fi

BOOST_VER="$(grep -m1 -oE 'DiFfRG_PINNED_BOOST_VERSION "[0-9.]+"' "$BUNDLE/DiFfRG_bundled_config.cmake" | grep -oE '[0-9.]+' || echo unknown)"
# In the release container .git is dockerignored; the SHA arrives via env from
# the build driver instead.
GIT_SHA="${GIT_SHA:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)}"
DIFFRG_VERSION="$(cat "$SRC/VERSION" 2>/dev/null | tr -d '[:space:]' || echo unknown)"
COMPILER_ID="$(c++ --version | head -1)"

cat > "$BUNDLE/BUNDLE_MANIFEST.json" <<EOF
{
  "name": "diffrg-deps-${BUNDLE_VERSION}-${VARIANT}",
  "bundle_version": "${BUNDLE_VERSION}",
  "os": "linux",
  "arch": "x86_64",
  "isa": "${MARCH}",
  "accel": "${ACCEL}",
  "mpi": "none",
  "diffrg_version": "${DIFFRG_VERSION}",
  "diffrg_git_sha": "${GIT_SHA}",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "build_prefix": "/opt/diffrg",
  "glibc_floor": "${GLIBC_FLOOR}",
  "compiler": "${COMPILER_ID}",
  "builder_cxx": "$(command -v c++ || true)",
  "builder_cc": "$(command -v cc || true)",
  "builder_fc": "$(command -v gfortran || true)",
  "boost_version": "${BOOST_VER}",
  "dependency_versions": {
$(dep_versions)
  },
  "allowed_external_libs": [
    ${CUDA_ALLOW}"linux-vdso", "ld-linux-x86-64", "libc", "libm", "libpthread", "libdl",
    "librt", "libgcc_s", "libstdc++", "libgfortran", "libquadmath", "libgomp",
    "libz", "libopenblas"
  ]
}
EOF
note "manifest written to $BUNDLE/BUNDLE_MANIFEST.json"
note "OK"
