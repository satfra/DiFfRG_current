#include "./flows.hh"

ONFiniteTFlows::ONFiniteTFlows(const DiFfRG::JSONValue &json) : V(quadrature_provider, json) {}
void ONFiniteTFlows::set_k(const double k) { DiFfRG::all_set_k(V, k); }
void ONFiniteTFlows::set_T(const double T) { DiFfRG::all_set_T(V, T); }
void ONFiniteTFlows::set_x_extent(const double x_extent) { DiFfRG::all_set_x_extent(V, x_extent); }