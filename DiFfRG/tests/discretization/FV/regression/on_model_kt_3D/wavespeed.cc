// Wave-speed-strategy comparison: sweep the three KT wave-speed strategies
// across all four integrator-based models (σ-PolyExp, σ-Litim, ρ-PolyExp,
// ρ-Litim). Prior data (Phases F/H/I in /IDA_failure_diagnosis.md) showed
// that on ρ-coords all three strategies fail at the same t to 5+ sig figs;
// kept here as a regression on that property.
//
// Tags:
//   [ws-zerod][!shouldfail]    MaxEigenvalueWaveSpeedZeroDeriv (Real<1>, H = 0)
//   [ws-ad][!shouldfail]       MaxEigenvalueWaveSpeed          (Real<2> AD for H)
//   [ws-fd][!shouldfail]       MaxEigenvalueWaveSpeedFD        (Real<1> + central-FD H)

#include "kt_3D_runners.hh"

#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed_fd.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed_zero_deriv.hh>

#include <catch2/catch_all.hpp>

namespace
{
  using namespace on_kt_3D;
  using ZeroDeriv = FV::KurganovTadmor::MaxEigenvalueWaveSpeedZeroDeriv;
  using FullAD = FV::KurganovTadmor::MaxEigenvalueWaveSpeed;
  using FD = FV::KurganovTadmor::MaxEigenvalueWaveSpeedFD;
}

// --- σ-coords, PolyExp T=0.05 -----------------------------------------------

TEST_CASE("wavespeed σ-PolyExp ZeroDeriv", "[wavespeed][sigma][polyexp][ws-zerod][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_sigma_integrator_PolyExp, ZeroDeriv>(default_sigma_grid(), final_time);
}

TEST_CASE("wavespeed σ-PolyExp AD (Real<2> H)", "[wavespeed][sigma][polyexp][ws-ad][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_sigma_integrator_PolyExp, FullAD>(default_sigma_grid(), final_time);
}

TEST_CASE("wavespeed σ-PolyExp FD (Real<1> + central-FD H)", "[wavespeed][sigma][polyexp][ws-fd][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_sigma_integrator_PolyExp, FD>(default_sigma_grid(), final_time);
}

// --- σ-coords, Litim T=0 ----------------------------------------------------

TEST_CASE("wavespeed σ-Litim ZeroDeriv", "[wavespeed][sigma][litim][ws-zerod][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_sigma_integrator_Litim, ZeroDeriv>(default_sigma_grid(), final_time);
}

TEST_CASE("wavespeed σ-Litim AD (Real<2> H)", "[wavespeed][sigma][litim][ws-ad][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_sigma_integrator_Litim, FullAD>(default_sigma_grid(), final_time);
}

TEST_CASE("wavespeed σ-Litim FD (Real<1> + central-FD H)", "[wavespeed][sigma][litim][ws-fd][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_sigma_integrator_Litim, FD>(default_sigma_grid(), final_time);
}

// --- ρ-coords, PolyExp T=0.05 -----------------------------------------------

TEST_CASE("wavespeed ρ-PolyExp ZeroDeriv", "[wavespeed][rho][polyexp][ws-zerod][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_rho_integrator_PolyExp, ZeroDeriv>(default_rho_grid(), final_time);
}

TEST_CASE("wavespeed ρ-PolyExp AD (Real<2> H)", "[wavespeed][rho][polyexp][ws-ad][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_rho_integrator_PolyExp, FullAD>(default_rho_grid(), final_time);
}

TEST_CASE("wavespeed ρ-PolyExp FD (Real<1> + central-FD H)", "[wavespeed][rho][polyexp][ws-fd][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_rho_integrator_PolyExp, FD>(default_rho_grid(), final_time);
}

// --- ρ-coords, Litim T=0 ----------------------------------------------------

TEST_CASE("wavespeed ρ-Litim ZeroDeriv", "[wavespeed][rho][litim][ws-zerod][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_rho_integrator_Litim, ZeroDeriv>(default_rho_grid(), final_time);
}

TEST_CASE("wavespeed ρ-Litim AD (Real<2> H)", "[wavespeed][rho][litim][ws-ad][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_rho_integrator_Litim, FullAD>(default_rho_grid(), final_time);
}

TEST_CASE("wavespeed ρ-Litim FD (Real<1> + central-FD H)", "[wavespeed][rho][litim][ws-fd][!shouldfail]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_rho_integrator_Litim, FD>(default_rho_grid(), final_time);
}
