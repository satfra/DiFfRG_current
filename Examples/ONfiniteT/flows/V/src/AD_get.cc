#include "../kernel.hh"

#include "../V.hh"

 void V_integrator::get(autodiff::real& dest, const double& k, const double& Nf, const double& T, const autodiff::real& m2Pi, const autodiff::real& m2Sigma)
{
integrator_AD.get(dest,  k, Nf, T, m2Pi, m2Sigma);
}
