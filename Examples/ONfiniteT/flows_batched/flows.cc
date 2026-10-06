#include "./flows.hh"

ONFiniteTBatchedFlows::ONFiniteTBatchedFlows(const DiFfRG::ConfigTree& config) : quadrature_provider(config), V(quadrature_provider, config), V_GPU(quadrature_provider, config), V_GPU_f(quadrature_provider, config)
{
}
void ONFiniteTBatchedFlows::set_k(const double k)
{
  DiFfRG::all_set_k(V, k);DiFfRG::all_set_k(V_GPU, k);DiFfRG::all_set_k(V_GPU_f, k);
}
void ONFiniteTBatchedFlows::set_T(const double T)
{
  DiFfRG::all_set_T(V, T);DiFfRG::all_set_T(V_GPU, T);DiFfRG::all_set_T(V_GPU_f, T);
}
void ONFiniteTBatchedFlows::set_typical_E(const double E)
{
  DiFfRG::all_set_typical_E(V, E);DiFfRG::all_set_typical_E(V_GPU, E);DiFfRG::all_set_typical_E(V_GPU_f, E);
}
void ONFiniteTBatchedFlows::set_x_extent(const double x_extent)
{
  DiFfRG::all_set_x_extent(V, x_extent);DiFfRG::all_set_x_extent(V_GPU, x_extent);DiFfRG::all_set_x_extent(V_GPU_f, x_extent);
}