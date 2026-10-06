#include "../kernel.hh"

#include "../V_GPU_f.hh"

void V_GPU_f_integrator::map_points(double* dest, const size_t n, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& m2Pi, const DiFfRG::PointArg<double>& m2Sigma)
{
  integrator.map_points(dest, n,  k, N, T, m2Pi, m2Sigma);
}