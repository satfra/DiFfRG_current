#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/utils.hh"

namespace DiFfRG
{
  template <typename _Regulator> class Zc_kernel
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
      return 3. * (-1. + powr<2>(cos1)) * powr<-1>(1. + powr<6>(k)) *
             (powr<2>(l1) + 2. * cos1 * l1 * p +
              powr<2>(p) * (1. + powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                  RB(powr<2>(k), powr<2>(l1)) * ZA(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) *
                  powr<2>(Zc(l1)) +
              RBdot(powr<2>(k), powr<2>(l1)) * (1. + powr<6>(k)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                  (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)) *
                  ZA(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * powr<2>(Zc(l1)) +
              -1. * powr<2>(l1) * powr<2>(ZA(l1)) * Zc(k) * Zc(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) +
              -1. *
                  (50. * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)) * powr<6>(k) *
                       ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                       ZA(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * powr<2>(Zc(l1)) +
                   -50. * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)) * powr<6>(k) *
                       ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)) *
                       ZA(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * powr<2>(Zc(l1)) +
                   powr<2>(l1) * (1. + powr<6>(k)) * powr<2>(ZA(l1)) * (dtZc(k) + (-1. * Zc(k) + Zc(1.02 * k)) * 50.) *
                       Zc(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))) *
                  RB(powr<2>(k), powr<2>(l1))) *
             powr<-4>(l1) * powr<-2>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)) * powr<-2>(ZA(l1)) *
             powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))) * powr<-2>(Zc(l1)) *
             powr<-1>(Zc(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))) *
             powr<2>(ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + cos1 * l1 * p + powr<2>(p))));
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
using DiFfRG::Zc_kernel;