# Commit Review Log

This file is a manual review and commit playbook for the current worktree.

Excluded from this log:
- `wolfram_gross_neveu_diagnostic/`
- `wolfram_on_diagnostic/`
- `REVIEW_LOG.md`
- `profile.json.gz`
- `.cpm-cache/`

Important mixed files:
- `DiFfRG/include/DiFfRG/discretization/FV/assembler/KurganovTadmor.hh`
- `DiFfRG/tests/discretization/FV/regression/on_model_kt_regression.cc`
- `DiFfRG/tests/discretization/FV/regression/gross_neveu_kt_regression.cc`
- `DiFfRG/tests/model/fv_boundaries.cc`

These require `git add -p`. Do not stage those files wholesale unless the step below says it is safe.

## Step 1: Support-point-aware affine-constraint plumbing [done]

Committed as `287cc63 Refine affine constraint model API`.

Intent:
- Extend `model.affine_constraints(...)` so models can constrain the dof nearest the origin using full support-point information, not just boundary-point information.

Review:
- `DiFfRG/include/DiFfRG/model/model.hh`
- `DiFfRG/include/DiFfRG/discretization/FEM/assembler/common.hh`
- `DiFfRG/include/DiFfRG/discretization/FEM/assembler/cg.hh`
- `DiFfRG/include/DiFfRG/discretization/FEM/assembler/ldg.hh`
- `DiFfRG/include/DiFfRG/discretization/FV/assembler/KurganovTadmor.hh`
- `DiFfRG/tests/discretization/FV/regression/gross_neveu_kt_regression.cc`

Stage:
- Whole-file stage is safe for:
  - `model.hh`
  - `common.hh`
  - `cg.hh`
  - `ldg.hh`
- Partial stage required for:
  - `KurganovTadmor.hh`: only the support-dof/support-point collection plus the widened `model.affine_constraints(...)` call.
  - `gross_neveu_kt_regression.cc`: only the `affine_constraints(...)` signature update and the new support-point-based origin constraint logic.

Do not include in this step:
- Any boundary-stencil residual/Jacobian logic in `KurganovTadmor.hh`.
- The assembler strategy alias change in `gross_neveu_kt_regression.cc`.
- Any output verbosity or relative-error cleanup in `gross_neveu_kt_regression.cc`.

What to verify:
- The only semantic change is how model constraints can discover the origin dof.
- No KT flux, wave-speed, or timestepping behavior changes are mixed in.

Suggested check:
```bash
git diff --cached --stat
```

Suggested commit message:
```text
Thread support-point metadata into affine constraints
```

## Step 2: Boundary-stencil API migration and KT boilerplate test-model split [done]

Committed together with Steps 3-5 in `Switch KT boundaries to stencil API`.

Intent:
- Replace the old face-wise ghost-state boundary hooks with the new stencil-based KT boundary hook.
- Move KT-specific boilerplate models into a dedicated helper header that implements `apply_boundary_stencil(...)`.

Review:
- `DiFfRG/include/DiFfRG/model/fv_boundaries.hh`
- `DiFfRG/tests/boilerplate/models.hh`
- `DiFfRG/tests/boilerplate/kt_models.hh`
- `DiFfRG/tests/discretization/FV/assembler_jacobian_fd_tests.cc`
- `DiFfRG/tests/discretization/FV/wave_speed_tests.cc`
- `DiFfRG/tests/timestepping/explicit_euler.cc`
- `DiFfRG/tests/timestepping/implicit_euler.cc`
- `DiFfRG/tests/timestepping/sundials_ida.cc`

Stage:
- Whole-file stage is safe for all files in this step.

Do not include in this step:
- The tolerance relaxation in `tests/timestepping/sundials_ida.cc`.
- Any `fv_boundaries.cc` test rewrite.
- Any `KurganovTadmor.hh` residual/Jacobian changes.

What to verify:
- `fv_boundaries.hh` now defines stencil-based boundary policies.
- The new KT boilerplate models are the only consumers switched in test code.
- The only change in the four test files is the include path switch to `boilerplate/kt_models.hh`.

Suggested check:
```bash
git diff --cached -- DiFfRG/tests/timestepping/sundials_ida.cc
```

Suggested commit message:
```text
Switch KT test models to boundary-stencil hooks
```

## Step 3: Boundary behavior and derivative tests for the new stencil API [done]

Committed together with Step 2 in `Switch KT boundaries to stencil API`.

Intent:
- Rewrite boundary tests so they validate the new two-ghost-cell stencil semantics and AD derivative propagation.

Review:
- `DiFfRG/tests/model/fv_boundaries.cc`

Stage:
- Partial stage required.

Include in this step:
- The rewritten `FVDefaultBoundaries` behavior tests.
- The rewritten `OriginOddLinearExtrapolationBoundaries` behavior tests.
- The AD derivative tests for both boundary strategies.

Do not include in this step:
- The deeper origin mismatch diagnostic block:
  - `TEST_CASE("Origin boundary reconstruction differs from the mirrored full-domain interior face", ...)`

What to verify:
- This step validates the new public boundary-policy behavior.
- It does not yet include the analytical investigation comparing half-domain and mirrored full-domain reconstructions.

Suggested check:
```bash
git diff --cached -- DiFfRG/tests/model/fv_boundaries.cc
```

Suggested commit message:
```text
Add stencil-based FV boundary behavior tests
```

## Step 4: Stencil-only KT boundary residual assembly [done]

Committed together with Step 2 in `Switch KT boundaries to stencil API`.

Intent:
- Remove the old ghost-face residual path and reconstruct KT boundary traces from a physical five-point stencil.

Review:
- `DiFfRG/include/DiFfRG/discretization/FV/assembler/KurganovTadmor.hh`

Stage:
- Partial stage required.

Include in this step:
- `BoundaryFaceReconstruction1D`
- `compute_boundary_face_reconstruction_1d(...)`
- `build_boundary_stencil_1d(...)`
- The boundary-adjacent cell-stencil population that calls `model.apply_boundary_stencil(...)`
- The boundary-face residual reconstruction path that uses the conditioned boundary stencil

Do not include in this step:
- The support-point affine-constraint plumbing already committed in Step 1.
- The `diagnose_state(...)` hook.
- The selected-wave-speed Jacobian derivative logic.
- The boundary Jacobian autodiff rewrite from Step 5.

What to verify:
- The residual path is now stencil-only for one-dimensional KT boundaries.
- Boundary reconstruction uses physical interior cells plus boundary policy conditioning.

Suggested check:
```bash
git diff --cached -- DiFfRG/include/DiFfRG/discretization/FV/assembler/KurganovTadmor.hh
```

Suggested commit message:
```text
Use stencil-only KT boundary reconstruction in residual assembly
```

## Step 5: Boundary Jacobian autodiff rewrite [done]

Committed together with Step 2 in `Switch KT boundaries to stencil API`.

Intent:
- Make the analytic KT boundary Jacobian trace through the same conditioned stencil as the residual path.

Review:
- `DiFfRG/include/DiFfRG/discretization/FV/assembler/KurganovTadmor.hh`

Stage:
- Partial stage required.

Include in this step:
- Removal of the old ghost-gradient derivative helper path.
- `make_tagged_physical_boundary_stencil(...)`
- The boundary Jacobian reconstruction derivative logic that runs autodiff through `model.apply_boundary_stencil(...)` and `compute_boundary_face_reconstruction_1d(...)`

Do not include in this step:
- Residual-only stencil logic from Step 4.
- Selected-wave-speed derivative selection from Step 6.
- Diagnostics hook from Step 7.

What to verify:
- Residual and Jacobian now depend on the same stencil-conditioned boundary reconstruction.
- The old boundary-ghost-gradient differentiation path is fully removed from this staged diff.

Suggested check:
```bash
git diff --cached --word-diff DiFfRG/include/DiFfRG/discretization/FV/assembler/KurganovTadmor.hh
```

Suggested commit message:
```text
Trace KT boundary Jacobians through conditioned stencils
```

## Step 6: Selected-wave-speed Jacobian fix

Intent:
- Make the KT Jacobian match the actually selected max-eigenvalue branch instead of always differentiating both candidate branches symmetrically.

Review:
- `DiFfRG/include/DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh`
- `DiFfRG/include/DiFfRG/discretization/FV/assembler/KurganovTadmor.hh`
- `DiFfRG/tests/discretization/FV/regression/gross_neveu_kt_regression.cc`

Stage:
- Whole-file stage is safe for `max_eigenvalue_wave_speed.hh`.
- Partial stage required for:
  - `KurganovTadmor.hh`: only the `compute_selected_speed_derivatives(...)` use site.
  - `gross_neveu_kt_regression.cc`: only the assembler alias change from `MaxEigenvalueWaveSpeedZeroDeriv` back to the default strategy.

Do not include in this step:
- Constraint-plumbing hunks from Step 1.
- Regression cleanup hunks from Step 8.

What to verify:
- The Jacobian branch-selection fix is isolated from unrelated test-harness changes.

Suggested check:
```bash
git diff --cached -- DiFfRG/include/DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh DiFfRG/tests/discretization/FV/regression/gross_neveu_kt_regression.cc
```

Suggested commit message:
```text
Match KT Jacobians to the selected wave-speed branch
```

## Step 7: Generic assembler and SUNDIALS diagnostics hook

Intent:
- Add a generic assembler diagnostic callback and invoke it when SUNDIALS IDA gets stuck.

Review:
- `DiFfRG/include/DiFfRG/discretization/common/abstract_assembler.hh`
- `DiFfRG/include/DiFfRG/discretization/FV/assembler/KurganovTadmor.hh`
- `DiFfRG/src/DiFfRG/timestepping/sundials_ida.cc`

Stage:
- Whole-file stage is safe for:
  - `abstract_assembler.hh`
  - `sundials_ida.cc`
- Partial stage required for:
  - `KurganovTadmor.hh`: only `diagnose_state(...)` and the `begin_diagnostics()` call site.

Do not include in this step:
- O(N)-specific diagnostic methods inside `on_model_kt_regression.cc`.
- Boundary residual/Jacobian changes from Steps 4 and 5.

What to verify:
- This step only provides the generic hook and runtime invocation.
- The generic hook is useful even without the O(N) regression harness.

Suggested check:
```bash
git diff --cached -- DiFfRG/src/DiFfRG/timestepping/sundials_ida.cc
```

Suggested commit message:
```text
Add assembler diagnostics for stuck SUNDIALS IDA solves
```

## Step 8: KT SUNDIALS tolerance relaxation

Intent:
- Relax the two KT-specific IDA test tolerances from `5e-7` to `2e-5`.

Review:
- `DiFfRG/tests/timestepping/sundials_ida.cc`

Stage:
- Partial stage required.

Include in this step:
- Only the two tolerance changes in:
  - `test_sundials_ida_traveling_wave_kt`
  - `test_sundials_ida_two_component_burgers_kt`

Do not include in this step:
- The include switch to `boilerplate/kt_models.hh`, which belongs to Step 2.

What to verify:
- Exactly two numeric changes are staged.

Suggested check:
```bash
git diff --cached -- DiFfRG/tests/timestepping/sundials_ida.cc
```

Suggested commit message:
```text
Relax KT SUNDIALS IDA regression tolerances
```

## Step 9: Gross-Neveu regression harness cleanup

Intent:
- Clean up the Gross-Neveu KT regression harness without changing the broader KT implementation.

Review:
- `DiFfRG/tests/discretization/FV/regression/gross_neveu_kt_regression.cc`

Stage:
- Partial stage required.

Include in this step:
- Zero-safe relative-error computation.
- Lower output verbosity.
- Removal of the stray debug print.

Do not include in this step:
- Support-point-based `affine_constraints(...)` adaptation from Step 1.
- Assembler alias change from Step 6.

What to verify:
- The staged diff is purely harness cleanup and reporting hygiene.

Suggested check:
```bash
git diff --cached -- DiFfRG/tests/discretization/FV/regression/gross_neveu_kt_regression.cc
```

Suggested commit message:
```text
Clean up Gross-Neveu KT regression comparisons
```

## Step 10: O(N) KT regression import

Intent:
- Add the base O(N) KT regression harness and its reference fixtures.

Review:
- `DiFfRG/tests/discretization/FV/regression/CMakeLists.txt`
- `DiFfRG/tests/discretization/FV/regression/on_model_kt_regression.cc`
- `DiFfRG/tests/discretization/FV/regression/data/2108_02504/sc_i_on_1_10_100_n_800_xmax_10_lambda_1e6_tir_60_rg_flow.json`
- `DiFfRG/tests/discretization/FV/regression/data/2108_02504/sc_i_on_3_n_800_xmax_10_lambda_1e6_tir_60_rg_flow.json`
- `DiFfRG/tests/discretization/FV/regression/data/2108_02504/sc_ii_n_on_4_n_800_xmax_10_lambda_1e12_tir_60_rg_flow.json`
- `DiFfRG/tests/discretization/FV/regression/data/2108_02504/sc_iii_on_4_n_800_xmax_10_lambda_1e12_tir_60_rg_flow.json`
- `DiFfRG/tests/discretization/FV/regression/data/2108_02504/sc_iv_on_3_n_800_xmax_10_lambda_1e8_tir_60_rg_flow.json`

Stage:
- Whole-file stage is safe for:
  - `CMakeLists.txt`
  - all five JSON fixtures
- Partial stage required for:
  - `on_model_kt_regression.cc`

Include in this step from `on_model_kt_regression.cc`:
- The original O(N) regression harness:
  - scenario descriptors
  - fixture loading
  - model definition for the baseline half-domain regression
  - mesh setup
  - snapshot comparison helpers
  - the template regression test covering the imported scenarios

Do not include in this step:
- `GridSettings`
- templated `ONKTModelBase`
- diagnostic bookkeeping methods
- symmetric full-domain comparison utilities
- `run_flow_to_time(...)`
- the three ScenarioI_ON3 diagnostic test cases added later

What to verify:
- This commit should read like “new regression suite plus data import”, not like a debugging session.

Suggested check:
```bash
git diff --cached --stat
```

Suggested commit message:
```text
Add O(N) KT regression suite and reference fixtures
```

## Step 11: O(N) diagnostic follow-up and domain-comparison investigation

Intent:
- Add the later diagnostic scaffolding used to compare half-domain and symmetric full-domain behavior and to print denominator diagnostics on stuck solves.

Review:
- `DiFfRG/tests/discretization/FV/regression/on_model_kt_regression.cc`
- `DiFfRG/tests/model/fv_boundaries.cc`

Stage:
- Partial stage required for both files.

Include in this step from `on_model_kt_regression.cc`:
- `GridSettings`
- `ONKTModelBase<BoundaryStrategy, constrain_origin>`
- `begin_diagnostics()`
- model-local `diagnose_state(...)`
- diagnostic denominator tracking
- `make_mesh_config(...)` parametrization
- `restrict_to_nonnegative_half(...)`
- `run_flow_to_time(...)`
- the three ScenarioI_ON3 diagnostic test cases:
  - symmetric default-boundary full-domain vs reference
  - half-domain origin-constrained solve reaches final time
  - half-domain vs full-domain comparison

Include in this step from `fv_boundaries.cc`:
- `TEST_CASE("Origin boundary reconstruction differs from the mirrored full-domain interior face", ...)`

What to verify:
- This is explicitly a diagnostic/investigative commit, not a pure production behavior change.
- The boundary mismatch test here should line up with the domain-comparison investigation in the O(N) regression file.

Suggested check:
```bash
git diff --cached -- DiFfRG/tests/discretization/FV/regression/on_model_kt_regression.cc DiFfRG/tests/model/fv_boundaries.cc
```

Suggested commit message:
```text
Add O(N) KT boundary diagnostics and half-vs-full-domain comparisons
```

## Final cleanup before starting the sequence

Sanity checks:
- Keep `wolfram_*`, `REVIEW_LOG.md`, `profile.json.gz`, and `.cpm-cache/` unstaged.
- Re-check `git status --short` before every commit.
- Use `git add -p` on the four mixed files named at the top of this log.

Useful commands:
```bash
git status --short
git diff
git diff --cached
git add -p
git reset -p
```
