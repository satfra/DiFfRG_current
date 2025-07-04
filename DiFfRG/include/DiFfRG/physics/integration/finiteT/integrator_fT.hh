
#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  template <typename NT, typename KERNEL, typename ExecutionSpace>
  using Integrator_fT = QuadratureIntegrator_fT<0, NT, KERNEL, ExecutionSpace>;
} // namespace DiFfRG
