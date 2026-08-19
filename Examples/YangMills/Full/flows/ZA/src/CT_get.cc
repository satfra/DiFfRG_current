#include "../kernel.hh"

#include "../ZA.hh"

void ZA_integrator::get(double& dest, const double& p, const double& k, const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates>& ZA3, const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates>& ZAcbc, const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>>& ZA4SP, const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates>& ZA4tadpole, const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>>& dtZc, const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>>& Zc, const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>>& dtZA, const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>>& ZA)
{
  integrator.get(dest, p,  k, ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA);
}