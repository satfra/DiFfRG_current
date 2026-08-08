#include "../kernel.hh"

#include "../V_pion.hh"

V_pion_integrator::V_pion_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& json) : integrator(quadrature_provider, json), integrator_AD(quadrature_provider, json), integrator_AD2(quadrature_provider, json), quadrature_provider(quadrature_provider)
{}