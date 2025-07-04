#include "./flows.hh"

ONFiniteTFlows::ONFiniteTFlows(const DiFfRG::JSONValue &json) : V(quadrature_provider, json) {}
void ONFiniteTFlows::set_k(const double k)
{
  if constexpr (DiFfRG::has_set_k<decltype(V.integrator)>) V.integrator.set_k(k);
  if constexpr (DiFfRG::has_integrator_AD<decltype(V)>)
    if constexpr (DiFfRG::has_set_k<decltype(V.integrator_AD)>) V.integrator_AD.set_k(k);
  if constexpr (DiFfRG::has_set_T<decltype(V.integrator)>) V.integrator.set_T(k);
  if constexpr (DiFfRG::has_integrator_AD<decltype(V)>)
    if constexpr (DiFfRG::has_set_T<decltype(V.integrator_AD)>) V.integrator_AD.set_T(k);
}