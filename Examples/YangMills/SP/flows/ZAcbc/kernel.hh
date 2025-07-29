#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/utils.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZAcbc_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const auto &l1, const auto &cos1, const auto &cos2, const auto &p, const double &k,
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
      const double cosp1q = cos1;
      const double cosp2q = (-1. * cos1 + sqrt(3. + -3. * powr<2>(cos1)) * cos2) * 0.5;
      const double cosp3q = (-1. * cos1 + -1. * sqrt(3. + -3. * powr<2>(cos1)) * cos2) * 0.5;
      return 0.5 * (cosp1q + 2. * cosp2q) * powr<-2>(l1) *
                 ((-1. + 2. * cosp1q * cosp2q + 2. * powr<2>(cosp2q)) * l1 + (2. * cosp1q + cosp2q) * p) * p *
                 (dtZc(k) * RB(powr<2>(k), powr<2>(l1)) + RBdot(powr<2>(k), powr<2>(l1)) * Zc(k) +
                  -50. * (Zc(k) + -1. * Zc(1.02 * k)) * RB(powr<2>(k), powr<2>(l1))) *
                 powr<-1>(powr<2>(l1) + -2. * cosp1q * l1 * p + powr<2>(p)) *
                 powr<-2>(powr<2>(l1) + 2. * cosp2q * l1 * p + powr<2>(p)) *
                 powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * cosp2q * l1 * p + powr<2>(p)))) * powr<-2>(Zc(l1)) *
                 powr<-1>(Zc(sqrt(powr<2>(l1) + -2. * cosp1q * l1 * p + powr<2>(p)))) *
                 ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + -1. * cosp1q * l1 * p + powr<2>(p))) *
                 ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + cosp2q * l1 * p + powr<2>(p))) *
                 ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + l1 * (-1. * cosp1q + cosp2q) * p + 1.5 * powr<2>(p))) +
             0.5 *
                 (powr<-1>(1. + powr<6>(k)) *
                      (-4. * cosp2q * powr<3>(l1) + 8. * powr<3>(cosp2q) * powr<3>(l1) + 6. * powr<2>(l1) * p +
                       -4. * powr<2>(cosp2q) * powr<2>(l1) * p +
                       2. * (2. * cosp2q * l1 + -7. * p) * powr<3>(cosp1q) * l1 * p + 2. * cosp2q * l1 * powr<2>(p) +
                       3. * powr<3>(p) +
                       2. *
                           (2. * cosp2q * powr<3>(l1) + -6. * powr<2>(l1) * p + 6. * powr<2>(cosp2q) * powr<2>(l1) * p +
                            -9. * cosp2q * l1 * powr<2>(p) + -3. * powr<3>(p)) *
                           powr<2>(cosp1q) +
                       (2. * (-1. + 6. * powr<2>(cosp2q)) * powr<3>(l1) +
                        2. * (-7. + 4. * powr<2>(cosp2q)) * cosp2q * powr<2>(l1) * p +
                        l1 * (7. + -4. * powr<2>(cosp2q)) * powr<2>(p) + -6. * cosp2q * powr<3>(p)) *
                           cosp1q) *
                      powr<-2>(ZA(l1)) *
                      (-1. * (1. + powr<6>(k)) * powr<2>(l1) * dtZc(k) *
                           RB(powr<2>(k), powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) * ZA(l1) +
                       -1. * powr<2>(l1) *
                           RBdot(powr<2>(k), powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) * ZA(l1) *
                           Zc(k) +
                       -1. * powr<6>(k) * powr<2>(l1) *
                           RBdot(powr<2>(k), powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) * ZA(l1) *
                           Zc(k) +
                       50. * (1. + powr<6>(k)) * powr<2>(l1) * (Zc(k) + -1. * Zc(1.02 * k)) *
                           RB(powr<2>(k), powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) * ZA(l1) +
                       powr<2>(l1) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) * RB(powr<2>(k), powr<2>(l1)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       powr<6>(k) * powr<2>(l1) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           RB(powr<2>(k), powr<2>(l1)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       2. * cosp1q * l1 * p * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           RB(powr<2>(k), powr<2>(l1)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       2. * cosp2q * l1 * p * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           RB(powr<2>(k), powr<2>(l1)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       2. * cosp1q * powr<6>(k) * l1 * p * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           RB(powr<2>(k), powr<2>(l1)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       2. * cosp2q * powr<6>(k) * l1 * p * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           RB(powr<2>(k), powr<2>(l1)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       powr<2>(p) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) * RB(powr<2>(k), powr<2>(l1)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       powr<6>(k) * powr<2>(p) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           RB(powr<2>(k), powr<2>(l1)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       -50. * powr<6>(k) * powr<2>(l1) * RB(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       -100. * cosp1q * powr<6>(k) * l1 * p * RB(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       -100. * cosp2q * powr<6>(k) * l1 * p * RB(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       -50. * powr<6>(k) * powr<2>(p) * RB(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       powr<2>(l1) * RBdot(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       powr<6>(k) * powr<2>(l1) * RBdot(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       2. * cosp1q * l1 * p * RBdot(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       2. * cosp2q * l1 * p * RBdot(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       2. * cosp1q * powr<6>(k) * l1 * p * RBdot(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       2. * cosp2q * powr<6>(k) * l1 * p * RBdot(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       powr<2>(p) * RBdot(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       powr<6>(k) * powr<2>(p) * RBdot(powr<2>(k), powr<2>(l1)) *
                           ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       50. * powr<6>(k) * powr<2>(l1) * RB(powr<2>(k), powr<2>(l1)) *
                           ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       100. * cosp1q * powr<6>(k) * l1 * p * RB(powr<2>(k), powr<2>(l1)) *
                           ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       100. * cosp2q * powr<6>(k) * l1 * p * RB(powr<2>(k), powr<2>(l1)) *
                           ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) +
                       50. * powr<6>(k) * powr<2>(p) * RB(powr<2>(k), powr<2>(l1)) *
                           ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)) *
                           Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)))) *
                      powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * cosp1q * l1 * p + powr<2>(p)))) *
                      ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cosp1q * l1 * p + powr<2>(p))) *
                      powr<-2>(Zc(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)))) +
                  powr<2>(l1) * (cosp1q + 2. * cosp2q) * powr<2>(l1) + 2. * cosp1q * l1 * p +
                  powr<2>(p) *
                      ((-1. + 2. * cosp1q * cosp2q + 2. * powr<2>(cosp2q)) * l1 + (-1. * cosp1q + cosp2q) * p) *
                      powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)))) *
                      (dtZc(k) * RB(powr<2>(k), powr<2>(l1)) + RBdot(powr<2>(k), powr<2>(l1)) * Zc(k) +
                       -50. * (Zc(k) + -1. * Zc(1.02 * k)) * RB(powr<2>(k), powr<2>(l1))) *
                      powr<-2>(Zc(l1)) * powr<-1>(Zc(sqrt(powr<2>(l1) + 2. * cosp1q * l1 * p + powr<2>(p)))) *
                      ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + cosp1q * l1 * p + powr<2>(p)))) *
                 powr<-4>(l1) * p * powr<-2>(powr<2>(l1) + 2. * cosp1q * l1 * p + powr<2>(p)) *
                 powr<-2>(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) *
                 ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + l1 * (cosp1q + cosp2q) * p + powr<2>(p))) *
                 ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + l1 * (2. * cosp1q + cosp2q) * p + 1.5 * powr<2>(p))) +
             0.25 *
                 (dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) * (1. + powr<6>(k)) * RB(powr<2>(k), powr<2>(l1)) +
                  RBdot(powr<2>(k), powr<2>(l1)) * (1. + powr<6>(k)) *
                      ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                  -50. *
                      (ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                       -1. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667))) *
                      powr<6>(k) * RB(powr<2>(k), powr<2>(l1))) *
                 powr<-1>(1. + powr<6>(k)) *
                 (-2. *
                      (-4. * cosp2q * powr<3>(l1) + 8. * powr<3>(cosp2q) * powr<3>(l1) + -6. * powr<2>(l1) * p +
                       4. * powr<3>(cosp1q) * cosp2q * powr<2>(l1) * p + 4. * powr<2>(cosp2q) * powr<2>(l1) * p +
                       2. * cosp2q * l1 * powr<2>(p) + -3. * powr<3>(p) +
                       2. *
                           (2. * cosp2q * powr<2>(l1) + l1 * p + 6. * powr<2>(cosp2q) * l1 * p +
                            -5. * cosp2q * powr<2>(p)) *
                           powr<2>(cosp1q) * l1 +
                       (2. * (-1. + 6. * powr<2>(cosp2q)) * powr<3>(l1) +
                        2. * (-3. + 4. * powr<2>(cosp2q)) * cosp2q * powr<2>(l1) * p +
                        l1 * (-5. + 4. * powr<2>(cosp2q)) * powr<2>(p) + -6. * cosp2q * powr<3>(p)) *
                           cosp1q) *
                      powr<-2>(powr<2>(l1) + 2. * cosp1q * l1 * p + powr<2>(p)) *
                      powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * cosp1q * l1 * p + powr<2>(p)))) *
                      ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cosp1q * l1 * p + powr<2>(p))) *
                      ZccbA(0.816496580927726 *
                            sqrt(powr<2>(l1) + l1 * (cosp1q + -1. * cosp2q) * p + 1.5 * powr<2>(p))) +
                  -1. * (-1. + 2. * cosp1q * cosp2q + 2. * powr<2>(cosp2q)) *
                      powr<-1>(powr<2>(l1) + -2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) *
                      (2. * cosp1q * l1 + 4. * cosp2q * l1 + -3. * p) *
                      powr<-1>(Zc(sqrt(powr<2>(l1) + -2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)))) *
                      ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + -1. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) *
                      ZccbA(0.5773502691896258 *
                            sqrt(2. * powr<2>(l1) + -2. * (cosp1q + 2. * cosp2q) * l1 * p + 3. * powr<2>(p)))) *
                 powr<-4>(l1) * p * powr<-1>(powr<2>(l1) + -2. * cosp2q * l1 * p + powr<2>(p)) * powr<-2>(ZA(l1)) *
                 powr<-1>(Zc(sqrt(powr<2>(l1) + -2. * cosp2q * l1 * p + powr<2>(p)))) *
                 ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + -1. * cosp2q * l1 * p + powr<2>(p)));
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
using DiFfRG::ZAcbc_kernel;