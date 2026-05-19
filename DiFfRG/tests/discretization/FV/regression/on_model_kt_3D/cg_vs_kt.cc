// CG-vs-KT discriminator hunt.
//
// Established (in smoke.cc and Examples/ONfiniteT*): on the LSM-physical
// integrator-based PolyExp T=0.05 setup,
//   - Examples/ONfiniteT/CG reaches t=4 in ~1.7s
//   - Examples/ONfiniteT_KT/KT fails at t≈0.008
//   - Our smoke.cc reproduces both: LSM_CG passes, every LSM_*_integrator_*
//     variant on the uniform Example grid stalls at t ≈ 0.012 - 0.015 (ρ)
//     or t ≈ 0.0016 - 0.002 (σ).
//
// Differences between the Example's CG and KT configurations:
//   (a) Discretisation:  CG fe_order=4  vs  KT FV (fe_order=0) with MinMod
//   (b) Numerical flux:  LLFFlux (CG)   vs  KT central + wave-speed
//   (c) Boundary:        FlowBoundaries vs  FVDefaultBoundaries / OriginOdd
//   (d) Initial grid:    adaptive (fine near ρ=0)  vs  uniform
//   (e) AMR:             on  vs  off
//   (f) IDA abs_tol:     1e-13  vs  1e-7
//
// (e) and (f) are not controlled in our test harness (no AMR; abs_tol via
// JSON, default 1e-7). This file isolates (d) and partial (a) by:
//   1. Running CG on the EXAMPLE KT uniform grid with default tolerance —
//      does CG still pass? Tests robustness of CG to grid choice.
//   2. Running CG on the same grid with the CG-Example tight tolerance —
//      does tightening abs_tol matter?
//   3. Running KT on the adaptive non-uniform CG-style grid — does KT
//      pass when given the same grid CG uses?
//   4. Running KT on a 4×-finer uniform grid (600 cells × [0, 0.015]) —
//      does uniform refinement help?
//
// References:
//   - arXiv:1903.09503 (Grossi & Wink, LDG predecessor): the IR flat-region
//     wave speed grows exponentially fast; explicit/CFL-style stepping
//     becomes infeasible. They circumvent this with SUNDIALS BDF (implicit).
//     We already use IDA (implicit, BDF-class); so the failure isn't pure
//     CFL — it's stiffness of the IDA-Jacobian, dominated by KT's λ_max
//     term that CG doesn't have. The tests below probe whether grid choice
//     can compensate.
//   - arXiv:2412.16053 (Zorbach et al., 2D-field-space KT): demonstrates
//     KT for multi-invariant problems but doesn't directly address the
//     IR-flat-region failure on the kind of test we run here.

#include "kt_3D_setup.hh"
#include "kt_3D_runners.hh"
#include "kt_3D_kernels.hh"  // for the diagnostic includes

// Limiter variants beyond MinMod (default) — for the kinks-off tests below.
#include <DiFfRG/discretization/FV/limiter/central_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/van_albada_limiter.hh>

#include <catch2/catch_all.hpp>

namespace
{
  using namespace on_kt_3D;
}

// ============================================================================
// Test 1: CG on KT-Example's uniform ρ-grid with KT's default abs_tol=1e-7.
// Hypothesis: CG should still reach t=4 (it's not relying on tight tol).
// If yes → tolerance isn't the discriminator.
// If no  → tolerance matters more than we thought.
// ============================================================================

TEST_CASE("CG-vs-KT: CG on KT-Example uniform grid + default tolerance",
          "[cg-vs-kt][cg-uniform-loose-tol]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_cg<LSM_CG>(default_rho_grid(), final_time, /*threads=*/1, /*fe_order=*/4,
                              /*abs_tol=*/1.0e-7, /*rel_tol=*/1.0e-7);
}

// ============================================================================
// Test 2: CG on KT-Example's uniform ρ-grid with the CG-Example's tight
// tolerance abs_tol=1e-13 (rel_tol=1e-7).
// Hypothesis: should also reach t=4. If yes, the CG advantage transfers to
// any grid.
// ============================================================================

TEST_CASE("CG-vs-KT: CG on KT-Example uniform grid + tight tolerance",
          "[cg-vs-kt][cg-uniform-tight-tol]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_cg<LSM_CG>(default_rho_grid(), final_time, /*threads=*/1, /*fe_order=*/4,
                              /*abs_tol=*/1.0e-13, /*rel_tol=*/1.0e-7);
}

// ============================================================================
// Test 3: KT on the CG-Example's adaptive non-uniform grid (fine near ρ=0,
// coarser farther out — same as `Examples/ONfiniteT/parameter.json`'s
// `0:1e-4:5e-3, 5e-3:5e-4:6e-3, 6e-3:1e-3:1.5e-2`).
// Hypothesis: if KT also passes here → grid adaptivity (not discretisation)
// is the discriminator. If KT still fails → KT discretisation is the
// intrinsic bottleneck on this physics.
// EXPECTED OUTCOME: KT likely still fails. Marked [!shouldfail] for now;
// flip the tag if it actually passes.
// ============================================================================

TEST_CASE("CG-vs-KT: KT on CG-Example adaptive non-uniform grid",
          "[cg-vs-kt][kt-adaptive-grid][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_kt_adaptive<LSM_rho_integrator_PolyExp>(example_cg_subranges(), final_time);
}

// ============================================================================
// Test 4: KT on a 4× uniformly-finer ρ-grid (600 cells × [0, 0.015]).
// Hypothesis: if more cells help → discretisation resolution is the
// bottleneck. If not → KT struggles regardless of uniform refinement.
// EXPECTED OUTCOME: based on Phase H, refinement is mesh-stable in the
// failure regime, so this should fail similarly to the 150-cell smoke.
// Marked [!shouldfail].
// ============================================================================

TEST_CASE("CG-vs-KT: KT on 4×-finer uniform ρ-grid (600 cells × [0, 0.015])",
          "[cg-vs-kt][kt-uniform-4x-finer][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  GridSettings fine_grid{600, 0.0, max_rho};
  run_flow_to_time<LSM_rho_integrator_PolyExp>(fine_grid, final_time);
}

// ============================================================================
// Test 5 (bonus): CG on the adaptive non-uniform grid — reproduces what the
// CG Example does exactly (modulo AMR). Should pass to t=4. If CG passes
// here AND on Test 1/2 (uniform), then CG is robust to grid choice; if CG
// only passes on adaptive but fails on uniform, then grid adaptivity is
// also needed for CG.
// ============================================================================

TEST_CASE("CG-vs-KT: CG on CG-Example adaptive non-uniform grid",
          "[cg-vs-kt][cg-adaptive-grid]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_cg_adaptive<LSM_CG>(example_cg_subranges(), final_time);
}

// ============================================================================
// Tests 6-7: BOTH kinks removed (limiter + wave-speed).
//
// Default file-level Reconstructor is MinMod, which is C^0 but NOT C^1 in u
// (sign-change AND magnitude-tie kinks per cell-face). MinMod + ZeroDeriv
// removes the WAVE-SPEED max-kink from J but NOT the limiter kink in the
// reconstruction u_plus = u_center + 0.5·MinMod(slope_L, slope_R)·dx.
// To remove the limiter kink too, swap to a smoother limiter:
//   VanAlbada: C^1 TVD modulo the (rarely-crossed) sign-change boundary
//   Central:   C^∞ but NOT TVD (diagnostic baseline)
// Both already implemented in DiFfRG/include/DiFfRG/discretization/FV/limiter/.
//
// If VanAlbada+ZeroDeriv (both kinks removed) passes → limiter is the
// bottleneck and we've found the fix.
// If it still fails → the bottleneck is elsewhere in the KT discretisation
// (the central+wave-speed flux structure itself, the boundary stencil, the
// piecewise-linear reconstruction in cells, etc.).
// ============================================================================

TEST_CASE("CG-vs-KT: KT-ρ uniform grid, VanAlbada limiter + ZeroDeriv (both kinks off)",
          "[cg-vs-kt][kt-vanalbada-zerod][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  // Custom limiter + ZeroDeriv requires a one-off run with explicit Assembler.
  // We can't go through run_flow_to_time (which uses the file-level MinMod).
  using ReconstructorVA = def::TVDReconstructor<dim, def::VanAlbadaLimiter, double>;
  using AssemblerVA = FV::KurganovTadmor::Assembler<Discretization, LSM_rho_integrator_PolyExp, ReconstructorVA,
                                                    FV::KurganovTadmor::MaxEigenvalueWaveSpeedZeroDeriv>;

  const auto grid = default_rho_grid();
  const JSONValue json = make_json(/*threads=*/1);
  LSM_rho_integrator_PolyExp model(json, grid);
  Mesh mesh(make_mesh_config(grid));
  Discretization discretization(mesh, json);
  AssemblerVA assembler(discretization, model, json);

  kt_regression::TemporaryDirectory tmp_dir("on_kt_3D");
  DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), "on_kt_3D", "output", json);
  auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
  ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  time_stepper.run(&state, 0.0, final_time);
}

TEST_CASE("CG-vs-KT: KT-ρ uniform grid, Central limiter + ZeroDeriv (both kinks off, NOT TVD)",
          "[cg-vs-kt][kt-central-zerod][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  using ReconstructorC = def::TVDReconstructor<dim, def::CentralLimiter, double>;
  using AssemblerC = FV::KurganovTadmor::Assembler<Discretization, LSM_rho_integrator_PolyExp, ReconstructorC,
                                                   FV::KurganovTadmor::MaxEigenvalueWaveSpeedZeroDeriv>;

  const auto grid = default_rho_grid();
  const JSONValue json = make_json(/*threads=*/1);
  LSM_rho_integrator_PolyExp model(json, grid);
  Mesh mesh(make_mesh_config(grid));
  Discretization discretization(mesh, json);
  AssemblerC assembler(discretization, model, json);

  kt_regression::TemporaryDirectory tmp_dir("on_kt_3D");
  DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), "on_kt_3D", "output", json);
  auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
  ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  time_stepper.run(&state, 0.0, final_time);
}
