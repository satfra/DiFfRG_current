#include "../kernel.hh"

#include "../ZA4.hh"

void ZA4_integrator::get(double &dest, const double &p, const double &k,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA3,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZccbA,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
{
  integrator.get(dest, p, k, ZA3, ZccbA, ZA4, dtZc, Zc, dtZA, ZA);
}