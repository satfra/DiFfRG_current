#include "./flows.hh"

ONFiniteTFlows::ONFiniteTFlows(const DiFfRG::JSONValue &json) : V(quadrature_provider, json) {}
void ONFiniteTFlows::set_k(const double k) { DiFfRG::all_set_k(V, k); }
void ONFiniteTFlows::set_T(const double T) { DiFfRG::all_set_T(V, T); }