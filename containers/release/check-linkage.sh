#!/usr/bin/env bash
# Self-containment audit: every shared object in the bundle (plus any extra
# binaries given as further arguments) must resolve all its dependencies, and
# everything it resolves outside the bundle must be on the allowlist of
# ordinary host runtime libraries. Run this in an environment WITHOUT the
# dev packages of Boost/TBB/HDF5/SUNDIALS to prove the bundle carries them.
#
# Usage: check-linkage.sh <bundle-dir> [extra-binary ...]
set -euo pipefail

BUNDLE="$1"
shift || true

fail() { echo "check-linkage: FAIL: $*" >&2; exit 1; }

ALLOW='linux-vdso|ld-linux-x86-64|libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|libgcc_s\.so|libstdc\+\+\.so|libgfortran\.so|libquadmath\.so|libgomp\.so|libz\.so|libopenblas'
# CUDA variants additionally resolve the host CUDA runtime/driver; enable with
# CHECK_LINKAGE_CUDA=1 (set by the cuda Dockerfile and test driver).
if [ "${CHECK_LINKAGE_CUDA:-0}" = 1 ]; then
  ALLOW="${ALLOW}|libcudart\.so|libcuda\.so|libnvrtc\.so"
fi

libdirs=("$BUNDLE/lib")
[ -d "$BUNDLE/lib64" ] && libdirs+=("$BUNDLE/lib64")
mapfile -t objects < <(find "${libdirs[@]}" -name '*.so*' -type f; printf '%s\n' "$@")

checked=0
for obj in "${objects[@]}"; do
  [ -f "$obj" ] || continue
  file -b "$obj" | grep -q ELF || continue
  out="$(ldd "$obj" 2>/dev/null)" || fail "ldd failed on $obj"
  if grep -q 'not found' <<<"$out"; then
    echo "$out" | grep 'not found' >&2
    fail "$obj has unresolved dependencies"
  fi
  while IFS= read -r line; do
    dep="$(awk '{print $1}' <<<"$line")"
    target="$(awk '{print $3}' <<<"$line")"
    # Dependencies resolved inside the bundle are fine; everything else must
    # be an allowlisted host runtime library.
    case "$target" in "$BUNDLE"/*) continue ;; esac
    grep -qE "$ALLOW" <<<"$dep" \
      || fail "$obj depends on non-allowlisted host library: $dep -> $target"
  done < <(grep '=>' <<<"$out")
  checked=$((checked + 1))
done

[ "$checked" -gt 0 ] || fail "no ELF objects checked"
echo "check-linkage: $checked objects fully resolved, externals allowlisted"
