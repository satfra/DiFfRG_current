#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/utils.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const auto &l1, const auto &cos1, const auto &p, const double &k,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA3,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZccbA,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      return (-4. * (-1. + powr<2>(cos1)) * powr<-1>(1. + powr<6>(k)) *
                  (8. * powr<2>(l1) * p + powr<2>(cos1) * powr<2>(l1) * p + 3. * powr<3>(p) +
                   6. * (powr<2>(l1) + powr<2>(p)) * cos1 * l1) *
                  powr<-4>(l1) *
                  (dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) * (1. + powr<6>(k)) *
                       RB(powr<2>(k), powr<2>(l1)) +
                   RBdot(powr<2>(k), powr<2>(l1)) * (1. + powr<6>(k)) *
                       ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                   -50. *
                       (ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                        -1. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667))) *
                       powr<6>(k) * RB(powr<2>(k), powr<2>(l1))) *
                  p * powr<-2>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)) * powr<-2>(ZA(l1)) *
                  powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))) *
                  powr<2>(ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cos1 * l1 * p + powr<2>(p)))) +
              -12. * (-1. + powr<2>(cos1)) * powr<-2>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)) *
                  (RBdot(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                   powr<-1>(1. + powr<6>(k)) *
                       ((1. + powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                        50. *
                            (-1. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                             ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667))) *
                            powr<6>(k)) *
                       RB(powr<2>(k), powr<2>(l1))) *
                  powr<-2>(ZA(l1)) * powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))) *
                  powr<2>(ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cos1 * l1 * p + powr<2>(p)))) +
              powr<-4>(l1) * (-7. + powr<2>(cos1)) * powr<-2>(ZA(l1)) *
                  (RBdot(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                   powr<-1>(1. + powr<6>(k)) *
                       ((1. + powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                        50. *
                            (-1. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                             ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667))) *
                            powr<6>(k)) *
                       RB(powr<2>(k), powr<2>(l1))) *
                  ZA4(0.7071067811865475 * sqrt(powr<2>(l1) + powr<2>(p))) +
              2. * (-1. + powr<2>(cos1)) * powr<-2>(l1) *
                  (-1. * RBdot(powr<2>(k), powr<2>(l1)) * Zc(k) +
                   -1. * (dtZc(k) + (-1. * Zc(k) + Zc(1.02 * k)) * 50.) * RB(powr<2>(k), powr<2>(l1))) *
                  powr<-1>(powr<2>(l1) + -2. * cos1 * l1 * p + powr<2>(p)) * powr<-2>(Zc(l1)) *
                  powr<-1>(Zc(sqrt(powr<2>(l1) + -2. * cos1 * l1 * p + powr<2>(p)))) *
                  powr<2>(ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + -1. * cos1 * l1 * p + powr<2>(p))))) *
             powr<-2>(p);
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto
    constant(const auto &p, const double &k,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA3,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZccbA,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      return 0.;
    }

  private:
    static KOKKOS_FORCEINLINE_FUNCTION auto RB(const auto &k2, const auto &p2) { return Regulator::RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RF(const auto &k2, const auto &p2) { return Regulator::RF(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RBdot(const auto &k2, const auto &p2) { return Regulator::RBdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RFdot(const auto &k2, const auto &p2) { return Regulator::RFdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RB(const auto &k2, const auto &p2) { return Regulator::dq2RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RF(const auto &k2, const auto &p2) { return Regulator::dq2RF(k2, p2); }
  };
} // namespace DiFfRG
using DiFfRG::ZA_kernel;