# CI dependency images

CI for DiFfRG is split in two so that the expensive part runs rarely:

1. **Dependency image** (this directory) — bakes the bundled superbuild
   dependencies (deal.II, Kokkos, autodiff, GSL, Eigen, spdlog, rapidcsv, …) into
   the default install prefix `/root/.local/share/DiFfRG/bundled` (exposed as
   `$DiFfRG_BUNDLED_DIR`). Built **rarely**, locally, and pushed to GHCR.
2. **Library build + tests** (`.github/workflows/ci.yml`) — on every push to a main
   branch and every pull request, builds *only* the DiFfRG library against that
   pre-built tree (`-DBUNDLED_DIR=$DiFfRG_BUNDLED_DIR`) and runs `ctest`. Minutes,
   not hours, because the superbuild is already cached in the image.

This contrasts with `containers/Base/` and `containers/CUDA/`, which rebuild
*everything* from scratch and are driven by `containers/test_all.sh` /
`build-container.sh` for occasional multi-distro and GPU compatibility sweeps.

## Files

| File | Purpose |
|------|---------|
| `ubuntu24.04-deps.Dockerfile` | CPU (GPU=OFF) dependency image. Multi-stage: builds the deps-only ExternalProject targets, then ships only the installed tree + toolchain. Consumed by `ci.yml`. |
| `ubuntu24.04-cuda-deps.Dockerfile` | CUDA (GPU=ON) dependency image, for **manual** GPU testing. Not used by automated CI (hosted runners have no GPU). |
| `build-and-push.sh` | Build a dependency image, smoke-test it (full library build + ctest inside the image), and push to GHCR. |

## Registry

Images live under **`ghcr.io/satfra/diffrg-deps`**, tagged:

- `ubuntu24.04` — moving tag consumed by `ci.yml` (and `ubuntu24.04-cuda` for the CUDA image).
- `ubuntu24.04-<YYYYMMDD>` — immutable dated record of each build.

Make the GHCR package **public** (its contents are all open-source dependencies)
so `ci.yml` can pull it with no credentials. If you keep it private instead, add a
`credentials:` block to the `container:` in `ci.yml` (see the comment there).

## Refreshing the image

Refresh whenever the bundled dependencies change — i.e. edits to `dependencies/**`
or the top-level `CMakeLists.txt` (dep versions, build flags). Library-only changes
do **not** require a refresh; `ci.yml` rebuilds the library from source every run.

### Locally (primary path)

```bash
# One-time GHCR login (classic PAT with write:packages; add `repo` for the first
# push of a brand-new package):
echo "$GHCR_PAT" | docker login ghcr.io -u <github-user> --password-stdin

# Build, smoke-test, and push the CPU image (half the host cores by default):
containers/ci/build-and-push.sh

# CUDA image (manual GPU testing); ctest in the smoke test runs only if a GPU is present:
containers/ci/build-and-push.sh -c -a AMPERE80

# Build + verify without pushing:
containers/ci/build-and-push.sh -n
```

The first push of a new package is private by default — open it in the GitHub UI
(`satfra` → Packages → `diffrg-deps`) and set visibility to public.

### Via GitHub Actions (backup)

The `Build dependency image` workflow (`.github/workflows/build-ci-images.yml`) does
the same on a `workflow_dispatch` trigger. Note hosted runners are slow (2–4 cores),
so the superbuild can take ~3 h; the local script is the recommended path.

## Test-count badge (README)

The README shows a `tests N/M passing` badge (green when all pass, red otherwise).
GitHub doesn't expose the test count, so `ci.yml` publishes it to a Gist that
[shields.io](https://shields.io) renders. One-time setup:

1. **Create a public Gist** (gist.github.com) with a file named `diffrg-tests.json`
   (any placeholder content). Note its ID — the long hash in the Gist URL.
2. **Create a PAT** (classic) with **only** the `gist` scope, and add it as a repo
   secret named **`GIST_SECRET`** (Settings → Secrets and variables → Actions).
3. **Fill in the two placeholders:**
   - `.github/workflows/ci.yml` → `gistID: REPLACE_WITH_GIST_ID`
   - `README.md` badge URL → `REPLACE_WITH_GIST_OWNER` (the Gist owner's username)
     and `REPLACE_WITH_GIST_ID`.

The badge updates on every **push to `main`** (the `Update test-count badge` step,
which runs with `if: always()`). It shows `N/M passing` (green if all pass, red if
any fail), or red **`build failed`** if the library didn't compile — so a broken
build never leaves the badge stuck at a stale/placeholder value. PRs intentionally
skip the update — fork PRs can't read secrets — but PR runs still build and test.
The count comes from parsing ctest's `"<F> tests failed out of <T>"` summary line,
so it needs no extra tooling.

## Verifying the CI path locally

Reproduce exactly what `ci.yml` does, against a pulled (or locally built) image:

```bash
docker run --rm -v "$(pwd):/src" ghcr.io/satfra/diffrg-deps:ubuntu24.04 bash -c '
  cmake -S /src/DiFfRG -B /tmp/bin -DBUNDLED_DIR="$DiFfRG_BUNDLED_DIR" \
    -DCMAKE_BUILD_TYPE=Release -DDiFfRG_TEST=ON -DDiFfRG_DOCUMENTATION=OFF
  cmake --build /tmp/bin -j4 && ctest --test-dir /tmp/bin --output-on-failure'
```
