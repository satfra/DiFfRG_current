// Positive regressions for the single production KT wave-speed strategy.
// MaxEigenvalueWaveSpeed evaluates the maximum spectral radius and obtains
// its state and gradient derivatives from the AD flux derivatives.

#include "kt_3D_runners.hh"

#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>

#include <catch2/catch_all.hpp>

namespace
{
  using namespace on_kt_3D;
  using WaveSpeed = FV::KurganovTadmor::MaxEigenvalueWaveSpeed;
}

TEST_CASE("wavespeed sigma-PolyExp full AD", "[wavespeed][sigma][polyexp]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_sigma_integrator_PolyExp, WaveSpeed>(default_sigma_grid(), final_time);
}

TEST_CASE("wavespeed sigma-Litim full AD", "[wavespeed][sigma][litim]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_sigma_integrator_Litim, WaveSpeed>(default_sigma_grid(), final_time);
}

TEST_CASE("wavespeed rho-PolyExp full AD", "[wavespeed][rho][polyexp]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_rho_integrator_PolyExp, WaveSpeed>(default_rho_grid(), final_time);
}

TEST_CASE("wavespeed rho-Litim full AD", "[wavespeed][rho][litim]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_ws<LSM_rho_integrator_Litim, WaveSpeed>(default_rho_grid(), final_time);
}
