#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

#include "./kernel.hh"

class V_integrator
{
public:
  V_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::JSONValue &json);

  using Regulator = DiFfRG::PolynomialExpRegulator<>;

  DiFfRG::Integrator_p2<3, double, V_kernel<Regulator>, DiFfRG::TBB_exec> integrator;

  DiFfRG::Integrator_p2<3, autodiff::real, V_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD;

  void get(double &dest, const double &k, const double &N, const double &T, const double &m2Pi, const double &m2Sigma);

  void get(autodiff::real &dest, const double &k, const double &N, const double &T, const autodiff::real &m2Pi,
           const autodiff::real &m2Sigma);

private:
  DiFfRG::QuadratureProvider &quadrature_provider;
};