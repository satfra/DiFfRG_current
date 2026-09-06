# Binary dependency releases

This directory builds, validates, and publishes the pre-built dependency
bundles that `install-diffrg-deps.sh` (repo root) installs for users: a
relocatable tarball of the full superbuild output (`bundled/`) attached to a
GitHub Release under a `deps-v<X.Y.Z>` tag.

This is distinct from `containers/ci/`, which bakes the same dependency tree
into a GHCR **container image** for CI, pinned to the CI distro's system
libraries. The release bundle instead force-bundles Boost/TBB/HDF5/SUNDIALS
(`-DBUILD_*=ON`), is compiled for the portable `x86-64-v3` ISA baseline
(`-DMARCH=x86-64-v3`), is built on Rocky 9 for a glibc 2.34 floor, and is
post-processed to be relocatable to any install prefix.

## Two equivalent build paths

Everything below can run **locally** (this walkthrough) or **on demand in CI**:
the `release-deps-linux` and `release-deps-macos` workflows
(`.github/workflows/`, `workflow_dispatch`) execute these same scripts on
GitHub runners, upload the tarball + sha256 + `.tested` stamp as a workflow
artifact, and can optionally attach everything to a **draft** release
`deps-v<version>` (input `draft_release`). Publishing is always manual: either
publish the draft in the Releases UI, or download the artifact into
`containers/release/dist/` and run `publish-release.sh`. The installers only
ever see *published* releases -- the GitHub API hides drafts -- so nothing is
user-visible until that manual step.

## Release walkthrough

```bash
# 1. Build the tarball (in Docker; ~2-3h). Writes dist/diffrg-deps-<v>-linux-x86_64-v3-cpu.tar.zst
containers/release/build-release.sh -v 1.0.0

# 2. Validate on the distro matrix (Ubuntu 24.04, Debian 13, Fedora 41, Rocky 9):
#    install from the tarball, audit linkage, build the library, run the quick
#    test suite, and re-install at a second prefix (relocation test).
containers/release/test-tarball.sh -f containers/release/dist/diffrg-deps-1.0.0-linux-x86_64-v3-cpu.tar.zst

# 3. Tag deps-v1.0.0 and create the GitHub release with tarball + sha256 + manifest.
containers/release/publish-release.sh -v 1.0.0
```

## Versioning

Bundle versions are their own `deps-vX.Y.Z` counter, independent of the DiFfRG
version (one bundle serves many DiFfRG commits; the manifest inside records the
exact git SHA it was built from). Bump the patch level for a rebuild, minor for
a dependency version bump, major for a deal.II/Boost/Kokkos major change.

## What makes the tarball relocatable

Built at the canonical prefix `/opt/diffrg`, then post-processed
(`postprocess-bundle.sh`):

- every shared library gets `RUNPATH=$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64`,
  so no binary patching is needed at install time (the superbuild pins
  `CMAKE_INSTALL_LIBDIR=lib` for a flat `lib/`; the lib64 leg and the scripts'
  lib64 handling are kept defensively for unforeseen layouts);
- deal.II's exported absolute system-library paths (`/usr/lib64/libz.so`, ...)
  are rewritten to plain `-l` flags so any distro's linker resolves them;
- the pin file `DiFfRG_bundled_config.cmake` is generated
  `${CMAKE_CURRENT_LIST_DIR}`-relative by the superbuild itself when all
  dependencies are bundled;
- the installer rewrites the remaining `/opt/diffrg` text occurrences (CMake
  configs, pkg-config files) to the user's prefix, and points deal.II's recorded
  compilers at the host toolchain.

Hard audits run before a tarball is produced: an ISA guard (no AVX-512
anywhere, AVX present in the heavyweight libraries -- catches both a
`-march=native` leak and dead `-march` plumbing), symbol-version caps
(`GLIBC_ <= 2.34`, el9 `GLIBCXX`/`CXXABI` baselines), a build-path leak scan,
and a linkage audit in a bare runtime container without any of the bundled
libraries' dev packages.

## Variants

The full set is `cpu` and `cuda12` (plus the experimental macOS build). There
are deliberately **no MPI variants**, now or planned: MPI's audience is
clusters, and a cluster build must link the site's fabric- and
Slurm-integrated MPI -- that is inherently a source build (the wizard's
self-build path with `-DMPI=ON`).

- **cpu** (`linux-x86_64-v3-cpu`): the baseline, described above.
- **cuda12** (`linux-x86_64-v3-cuda12`, `build-release.sh -V ...`): CUDA-enabled
  Kokkos at the sm_80/Ampere floor -- Kokkos allows exactly one CUDA arch per
  build, and sm_80 embeds compute_80 PTX so every newer GPU runs via JIT
  (cached after first launch). Consumers need the CUDA 12 toolkit anyway (their
  apps compile device code), so the bundle resolves the host's libcudart. The
  host compiler must be **GCC 12 or >= 14** (never 13: nvcc's frontend
  miscompiles GCC 13's libstdc++ in C++20 mode -- the `iterator_traits<char*>`
  bug; both 12 and 14 are validated). On gcc-13 distros install g++-14 and run
  the installer with `CXX=g++-14`. The deal.II nvcc-wrapper shim ships in
  `bundled/bin` and self-relocates. The CPU bundles have no such constraint:
  any C++20 compiler, GCC >= 12, works.
  GPU-executed validation needs a GPU host: `test-tarball.sh -g` (uses the
  `*-cuda` test images; CI does build-only).
- **macOS arm64** (`macos-arm64-cpu.sh`): experimental, see above; no `-march`
  pinning needed (all M-series share Apple clang's arm64 baseline).
