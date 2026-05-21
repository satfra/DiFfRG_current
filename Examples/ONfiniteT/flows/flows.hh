#pragma once

#include "DiFfRG/common/utils.hh"
#include "DiFfRG/physics/integration.hh"
#include "./V/V.hh"

class ONFiniteTFlows
{
public:   ONFiniteTFlows(const DiFfRG::JSONValue& json)
;

 void set_k(const double k)
;

 void set_T(const double T)
;

 void set_typical_E(const double E)
;

 void set_x_extent(const double x_extent)
;

DiFfRG::QuadratureProvider quadrature_provider;

V_integrator V;
};