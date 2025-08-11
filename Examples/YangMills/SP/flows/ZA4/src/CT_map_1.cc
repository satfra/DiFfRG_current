#include "../kernel.hh"

#include "../ZA4.hh"

DiFfRG::GPU_exec
ZA4_integrator::map(double *dest, const LogarithmicCoordinates1D<double> &coordinates, const double &k,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA3,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZAcbc,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
{
  return integrator.map(dest, coordinates, k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);
}