#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

#include "./kernel.hh"

class V_integrator
{
public:
  V_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::JSONValue &json);

  template <typename NT = double> void get(NT &dest, auto &&...t)
  {
    static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>,
                  "Unknown type requested of V_integrator::get");
    if constexpr (std::is_same_v<NT, double>)
      get_CT(dest, std::forward<decltype(t)>(t)...);
    else if constexpr (std::is_same_v<NT, autodiff::real>)
      get_AD(dest, std::forward<decltype(t)>(t)...);
  }

  using Regulator = DiFfRG::PolynomialExpRegulator<>;

  DiFfRG::Integrator_p2<3, double, V_kernel<Regulator>, DiFfRG::TBB_exec> integrator;

  DiFfRG::Integrator_p2<3, autodiff::real, V_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD;

private:
  void get_CT(double &dest, const double &k, const double &N, const double &T, const double &m2Pi,
              const double &m2Sigma);

  void get_AD(autodiff::real &dest, const double &k, const double &N, const double &T, const autodiff::real &m2Pi,
              const autodiff::real &m2Sigma);

  DiFfRG::QuadratureProvider &quadrature_provider;
};