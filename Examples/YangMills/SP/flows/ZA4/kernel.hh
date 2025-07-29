#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/utils.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA4_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const auto &l1, const auto &cos1, const auto &cos2, const auto &phi, const auto &p, const double &k,
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
      return 0.0909090909090909 *
             (-1. *
                  (dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) * (1. + powr<6>(k)) *
                       RB(powr<2>(k), powr<2>(l1)) +
                   RBdot(powr<2>(k), powr<2>(l1)) * (1. + powr<6>(k)) *
                       ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                   -50. *
                       (ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                        -1. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667))) *
                       powr<6>(k) * RB(powr<2>(k), powr<2>(l1))) *
                  powr<-1>(1. + powr<6>(k)) *
                  (-1. *
                       (-16. * powr<5>(cosp2q) * powr<5>(l1) * powr<2>(p) +
                        32. * powr<6>(cosp1q) * powr<4>(l1) * powr<3>(p) +
                        16. * (13. * powr<2>(l1) + 6. * cosp2q * l1 * p + 28. * powr<2>(p)) * powr<5>(cosp1q) *
                            powr<3>(l1) * powr<2>(p) +
                        -48. * (2. * powr<6>(l1) * p + 5. * powr<4>(l1) * powr<3>(p)) * powr<4>(cosp2q) +
                        2. *
                            (96. * powr<4>(l1) + 260. * cosp2q * powr<3>(l1) * p +
                             2. * (311. + 10. * powr<2>(cosp2q)) * powr<2>(l1) * powr<2>(p) +
                             560. * cosp2q * l1 * powr<3>(p) + 627. * powr<4>(p)) *
                            powr<4>(cosp1q) * powr<2>(l1) * p +
                        -8. * (6. * powr<7>(l1) + 25. * powr<5>(l1) * powr<2>(p) + 16. * powr<3>(l1) * powr<4>(p)) *
                            powr<3>(cosp2q) +
                        8. * (3. * powr<6>(l1) * p + 7. * powr<4>(l1) * powr<3>(p) + -9. * powr<2>(l1) * powr<5>(p)) *
                            powr<2>(cosp2q) +
                        4. *
                            (12. * powr<6>(l1) + 96. * cosp2q * powr<5>(l1) * p +
                             2. * (97. + 2. * powr<2>(cosp2q)) * powr<4>(l1) * powr<2>(p) +
                             2. * (311. + -10. * powr<2>(cosp2q)) * cosp2q * powr<3>(l1) * powr<3>(p) +
                             8. * (60. + 17. * powr<2>(cosp2q)) * powr<2>(l1) * powr<4>(p) +
                             627. * cosp2q * l1 * powr<5>(p) + 231. * powr<6>(p)) *
                            powr<3>(cosp1q) * l1 +
                        -1. *
                            (144. * powr<5>(l1) * powr<2>(p) + 482. * powr<3>(l1) * powr<4>(p) +
                             231. * l1 * powr<6>(p)) *
                            cosp2q +
                        (24. * powr<6>(l1) * p + 124. * powr<4>(l1) * powr<3>(p) + 126. * powr<2>(l1) * powr<5>(p) +
                         33. * powr<7>(p)) *
                            -3. +
                        (132. * powr<6>(l1) * p + 326. * powr<4>(l1) * powr<3>(p) +
                         -72. * powr<4>(cosp2q) * powr<4>(l1) * powr<3>(p) + 153. * powr<2>(l1) * powr<5>(p) +
                         198. * powr<7>(p) +
                         -16. * (31. * powr<5>(l1) * powr<2>(p) + 19. * powr<3>(l1) * powr<4>(p)) * powr<3>(cosp2q) +
                         (-144. * powr<6>(l1) * p + 436. * powr<4>(l1) * powr<3>(p) +
                          1302. * powr<2>(l1) * powr<5>(p)) *
                             powr<2>(cosp2q) +
                         6. *
                             (12. * powr<7>(l1) + 194. * powr<5>(l1) * powr<2>(p) + 480. * powr<3>(l1) * powr<4>(p) +
                              231. * l1 * powr<6>(p)) *
                             cosp2q) *
                            powr<2>(cosp1q) +
                        -1. *
                            (288. * powr<5>(l1) * powr<2>(p) + 16. * powr<5>(cosp2q) * powr<4>(l1) * powr<3>(p) +
                             964. * powr<3>(l1) * powr<4>(p) + 462. * l1 * powr<6>(p) +
                             8. * (29. * powr<5>(l1) * powr<2>(p) + 22. * powr<3>(l1) * powr<4>(p)) * powr<4>(cosp2q) +
                             8. *
                                 (42. * powr<6>(l1) * p + 101. * powr<4>(l1) * powr<3>(p) +
                                  -6. * powr<2>(l1) * powr<5>(p)) *
                                 powr<3>(cosp2q) +
                             2. *
                                 (36. * powr<7>(l1) + 6. * powr<5>(l1) * powr<2>(p) + -352. * powr<3>(l1) * powr<4>(p) +
                                  -231. * l1 * powr<6>(p)) *
                                 powr<2>(cosp2q) +
                             -1. *
                                 (132. * powr<6>(l1) * p + 326. * powr<4>(l1) * powr<3>(p) +
                                  153. * powr<2>(l1) * powr<5>(p) + 198. * powr<7>(p)) *
                                 cosp2q) *
                            cosp1q) *
                       ZA3(0.816496580927726 * sqrt(powr<2>(l1) + l1 * (cosp1q + cosp2q) * p + powr<2>(p))) *
                       ZA3(0.816496580927726 * sqrt(powr<2>(l1) + l1 * (2. * cosp1q + cosp2q) * p + 1.5 * powr<2>(p))) +
                   3. *
                       ((-54. + 53. * powr<2>(cosp1q) + -4. * cosp1q * cosp2q + -4. * powr<2>(cosp2q)) * powr<2>(l1) +
                        cosp1q * (-54. + 53. * powr<2>(cosp1q) + -4. * cosp1q * cosp2q + -4. * powr<2>(cosp2q)) * l1 *
                            p +
                        33. * (-1. + powr<2>(cosp1q)) * powr<2>(p)) *
                       p * powr<2>(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) *
                       ZA(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) *
                       ZA4(0.5 * sqrt(2. * powr<2>(l1) + 2. * cosp1q * l1 * p + 3. * powr<2>(p)))) *
                  powr<-2>(powr<2>(l1) + 2. * cosp1q * l1 * p + powr<2>(p)) *
                  powr<-2>(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) * powr<-2>(ZA(l1)) *
                  powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * cosp1q * l1 * p + powr<2>(p)))) *
                  powr<-1>(ZA(sqrt(powr<2>(l1) + 2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)))) *
                  ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cosp1q * l1 * p + powr<2>(p))) +
              powr<2>(l1) *
                  (4. * powr<3>(cosp1q) * l1 + -4. * powr<3>(cosp2q) * l1 +
                   (6. * cosp2q * l1 + -11. * p) * powr<2>(cosp1q) + 6. * p + -2. * powr<2>(cosp2q) * p +
                   -1. * (6. * cosp2q * l1 + 11. * p) * cosp1q * cosp2q) *
                  powr<-1>(powr<2>(l1) + -2. * cosp1q * l1 * p + powr<2>(p)) *
                  (dtZc(k) * RB(powr<2>(k), powr<2>(l1)) + RBdot(powr<2>(k), powr<2>(l1)) * Zc(k) +
                   -50. * (Zc(k) + -1. * Zc(1.02 * k)) * RB(powr<2>(k), powr<2>(l1))) *
                  powr<-1>(powr<2>(l1) + -2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)) * powr<-2>(Zc(l1)) *
                  powr<-1>(Zc(sqrt(powr<2>(l1) + -2. * cosp1q * l1 * p + powr<2>(p)))) *
                  powr<-1>(Zc(sqrt(powr<2>(l1) + -2. * (cosp1q + cosp2q) * l1 * p + powr<2>(p)))) *
                  ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + -1. * cosp1q * l1 * p + powr<2>(p))) *
                  ZccbA(0.816496580927726 * sqrt(powr<2>(l1) + -1. * (cosp1q + cosp2q) * l1 * p + powr<2>(p))) *
                  ZccbA(0.5773502691896258 *
                        sqrt(2. * powr<2>(l1) + -2. * (2. * cosp1q + cosp2q) * l1 * p + 3. * powr<2>(p)))) *
             powr<-4>(l1) * powr<-1>(p);
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
using DiFfRG::ZA4_kernel;