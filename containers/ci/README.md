# CI dependency images

CI for DiFfRG keeps the expensive dependency superbuild out of the regular
library test path:

1. **Dependency image** (this directory) — bakes the bundled superbuild
   dependencies (deal.II, Kokkos, autodiff, GSL, Eigen, spdlog, rapidcsv, …) into
   `/opt/diffrg/bundled` (exposed as `$DiFfRG_BUNDLED_DIR`). The stable image is
   built rarely and pushed to GHCR as a Docker/OCI image.
2. **Library build + tests** (`.github/workflows/ci.yml`) — on every push to a main
   branch and every pull request, selects a dependency image, pulls it with
   Singularity, builds *only* the DiFfRG library against the pre-built tree
   (`-DBUNDLED_DIR=$DiFfRG_BUNDLED_DIR`), and runs `ctest`. Library-only changes
   use `docker://ghcr.io/satfra/diffrg-deps:ubuntu24.04`; dependency-image
   changes first build and test a commit-specific image.
3. **Example and Wolfram probes** (`.github/workflows/ci.yml`) — additional jobs
   reuse the same dependency image through Singularity. The example job installs
   DiFfRG into `.ci/diffrg-install` and builds every current `Examples/*` CMake
   project, while allowing explicitly listed legacy examples to remain
   report-only. The Wolfram job bind-mounts the host `wolframscript`/FORM setup
   into the container and reports generator status plus generated-flow drift
   without baking licensed Wolfram files into the public image. If Wolfram
   preflight passes, the job then rebuilds the examples and compares selected
   short-run outputs against committed baselines.

This contrasts with `containers/Base/` and `containers/CUDA/`, which rebuild
*everything* from scratch and are driven by `containers/test_all.sh` /
`build-container.sh` for occasional multi-distro and GPU compatibility sweeps.

## Files

| File | Purpose |
|------|---------|
| `ubuntu24.04-deps.Dockerfile` | CPU (GPU=OFF) dependency image. Multi-stage: builds the deps-only ExternalProject targets, then ships only the installed tree + toolchain. Consumed by `ci.yml` through Singularity. |
| `ubuntu24.04-cuda-deps.Dockerfile` | CUDA (GPU=ON) dependency image, for **manual** GPU testing. Not used by automated CI (hosted runners have no GPU). |
| `build-and-push.sh` | Build a dependency image, smoke-test it through Singularity (full library build + ctest), and push to GHCR. |
| `build-examples.sh` | Runs inside the dependency image, installs DiFfRG into `.ci/diffrg-install`, and builds the current example CMake projects with per-example logs. |
| `run-wolfram-in-container.sh` | Host-side wrapper that bind-mounts available Wolfram/Form paths and runs Wolfram checks inside the dependency image. |
| `wolfram-example-checks.sh` | Runs inside the dependency image and records Wolfram preflight, generator, and generated-flow-drift results. |
| `run-example-regressions.sh` | Runs selected built examples with short CI overrides and compares text outputs against `Examples/ci_baselines/`. |
| `compare-example-baseline.py` | Normalizes and compares or updates small text baselines for example regression outputs. |

## Registry

Images live under **`ghcr.io/satfra/diffrg-deps`**, tagged:

- `ubuntu24.04` — stable moving tag consumed by `ci.yml` when dependency inputs
  did not change.
- `ci-<sha>` — commit-specific image built by `ci.yml` when dependency inputs
  change; this exact image is tested before promotion.
- `ubuntu24.04-<run_number>` — immutable record produced when CI promotes a
  tested `ci-<sha>` image on `main`.
- `ubuntu24.04-<YYYYMMDD>` — immutable dated record produced by the local
  `build-and-push.sh` script.
- `ubuntu24.04-cuda` — manual CUDA dependency image.

Make the GHCR package **public** (its contents are all open-source dependencies)
so `ci.yml` can pull it with no credentials. If you keep it private instead,
`ci.yml` passes `SINGULARITY_DOCKER_USERNAME` / `SINGULARITY_DOCKER_PASSWORD`
from the GitHub actor and token for the `singularity pull`.

## Refreshing the image

CI detects dependency-image inputs (`CMakeLists.txt`, `dependencies/**`,
`patches/**`, and `containers/ci/**`). If one of these paths changes, `ci.yml`
builds `ci-<sha>`, tests against that exact image, and promotes it to
`ubuntu24.04` after a successful push to `main`. Library-only changes do **not**
refresh the dependency image; `ci.yml` rebuilds the library from source every run.

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

The `Build dependency image` workflow (`.github/workflows/build-ci-images.yml`)
is a manual `workflow_dispatch` fallback. It builds and pushes the requested
stable tag directly, outside the main CI dependency chain. The normal automated
path is the `deps-image` job in `ci.yml`, because that makes the test job depend
on the exact image that was just built.

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

Reproduce exactly what `ci.yml` does, against the pushed image:

```bash
containers/singularity-run.sh \
  -b "$(pwd):/src" \
  -w /tmp \
  docker://ghcr.io/satfra/diffrg-deps:ubuntu24.04 \
  bash -lc '
  cmake -S /src/DiFfRG -B /tmp/bin -DBUNDLED_DIR="${DiFfRG_BUNDLED_DIR:-/opt/diffrg/bundled}" \
    -DCMAKE_BUILD_TYPE=Release -DDiFfRG_TEST=ON -DDiFfRG_DOCUMENTATION=OFF
  cmake --build /tmp/bin -j4 && ctest --test-dir /tmp/bin --output-on-failure'
```
