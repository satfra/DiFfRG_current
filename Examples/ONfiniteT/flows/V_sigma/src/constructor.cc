#include "../kernel.hh"

#include "../V_sigma.hh"

V_sigma_integrator::V_sigma_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::JSONValue& json) : integrator(quadrature_provider, json), integrator_AD(quadrature_provider, json), integrator_AD2(quadrature_provider, json), quadrature_provider(quadrature_provider)
{}