#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/utils.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA3_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const auto &l1, const auto &cos1, const auto &cos2, const auto &p, const double &k,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA3,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZAcbc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const double cosl1p1 = cos1;
      const double cosl1p2 = ((-1.) * (cos1) + (sqrt(3. + (-3.) * (powr<2>(cos1)))) * (cos2)) * (0.5);
      const double cosl1p3 = ((-1.) * (cos1) + (-1.) * ((sqrt(3. + (-3.) * (powr<2>(cos1)))) * (cos2))) * (0.5);
      const auto _repl1 = RB(powr<2>(k), powr<2>(l1));
      const auto _repl2 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _repl3 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl4 = Zc(k);
      return (0.0909090909090909) *
                 ((_repl2) * (1. + powr<6>(k)) * (_repl3) +
                  (_repl1) * (1. + powr<6>(k)) * (dtZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                  (-50.) * (_repl3 + (-1.) * (ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667))))) *
                      ((_repl1) * (powr<6>(k)))) *
                 ((powr<-1>(1. + powr<6>(k))) *
                  ((powr<-1>(p)) *
                       ((-3.) *
                            ((24.) * (powr<6>(l1)) + (124.) * ((powr<4>(l1)) * (powr<2>(p))) +
                             (126.) * ((powr<2>(l1)) * (powr<4>(p))) + (33.) * (powr<6>(p))) *
                            (p) +
                        (cosl1p2) *
                            ((144.) * ((powr<4>(l1)) * (powr<2>(p))) +
                             (16.) * ((powr<4>(cosl1p2)) * ((powr<4>(l1)) * (powr<2>(p)))) +
                             (482.) * ((powr<2>(l1)) * (powr<4>(p))) + (231.) * (powr<6>(p)) +
                             (-48.) * ((2.) * ((powr<5>(l1)) * (p)) + (5.) * ((powr<3>(l1)) * (powr<3>(p)))) *
                                 (powr<3>(cosl1p2)) +
                             (8.) *
                                 ((6.) * (powr<6>(l1)) + (25.) * ((powr<4>(l1)) * (powr<2>(p))) +
                                  (16.) * ((powr<2>(l1)) * (powr<4>(p)))) *
                                 (powr<2>(cosl1p2)) +
                             (8.) *
                                 ((3.) * ((powr<5>(l1)) * (p)) + (7.) * ((powr<3>(l1)) * (powr<3>(p))) +
                                  (-9.) * ((l1) * (powr<5>(p)))) *
                                 (cosl1p2)) *
                            (l1) +
                        ((288.) * ((powr<5>(l1)) * (powr<2>(p))) +
                         (32.) * ((powr<5>(cosl1p1)) * ((powr<4>(l1)) * (powr<3>(p)))) +
                         (-16.) * ((powr<5>(cosl1p2)) * ((powr<4>(l1)) * (powr<3>(p)))) +
                         (964.) * ((powr<3>(l1)) * (powr<4>(p))) + (462.) * ((l1) * (powr<6>(p))) +
                         (-16.) * ((13.) * (powr<2>(l1)) + (-6.) * ((cosl1p2) * ((l1) * (p))) + (28.) * (powr<2>(p))) *
                             ((powr<4>(cosl1p1)) * ((powr<3>(l1)) * (powr<2>(p)))) +
                         (2.) *
                             ((96.) * (powr<4>(l1)) + (-260.) * ((cosl1p2) * ((powr<3>(l1)) * (p))) +
                              (2.) * (311. + (10.) * (powr<2>(cosl1p2))) * ((powr<2>(l1)) * (powr<2>(p))) +
                              (-560.) * ((cosl1p2) * ((l1) * (powr<3>(p)))) + (627.) * (powr<4>(p))) *
                             ((powr<3>(cosl1p1)) * ((powr<2>(l1)) * (p))) +
                         (8.) * ((29.) * ((powr<5>(l1)) * (powr<2>(p))) + (22.) * ((powr<3>(l1)) * (powr<4>(p)))) *
                             (powr<4>(cosl1p2)) +
                         (-8.) *
                             ((42.) * ((powr<6>(l1)) * (p)) + (101.) * ((powr<4>(l1)) * (powr<3>(p))) +
                              (-6.) * ((powr<2>(l1)) * (powr<5>(p)))) *
                             (powr<3>(cosl1p2)) +
                         (-4.) *
                             ((12.) * (powr<6>(l1)) + (-96.) * ((cosl1p2) * ((powr<5>(l1)) * (p))) +
                              (2.) * (97. + (2.) * (powr<2>(cosl1p2))) * ((powr<4>(l1)) * (powr<2>(p))) +
                              (2.) * (-311. + (10.) * (powr<2>(cosl1p2))) *
                                  ((cosl1p2) * ((powr<3>(l1)) * (powr<3>(p)))) +
                              (8.) * (60. + (17.) * (powr<2>(cosl1p2))) * ((powr<2>(l1)) * (powr<4>(p))) +
                              (-627.) * ((cosl1p2) * ((l1) * (powr<5>(p)))) + (231.) * (powr<6>(p))) *
                             ((powr<2>(cosl1p1)) * (l1)) +
                         (2.) *
                             ((36.) * (powr<7>(l1)) + (6.) * ((powr<5>(l1)) * (powr<2>(p))) +
                              (-352.) * ((powr<3>(l1)) * (powr<4>(p))) + (-231.) * ((l1) * (powr<6>(p)))) *
                             (powr<2>(cosl1p2)) +
                         ((132.) * ((powr<6>(l1)) * (p)) + (326.) * ((powr<4>(l1)) * (powr<3>(p))) +
                          (153.) * ((powr<2>(l1)) * (powr<5>(p))) + (198.) * (powr<7>(p))) *
                             (cosl1p2) +
                         ((132.) * ((powr<6>(l1)) * (p)) + (326.) * ((powr<4>(l1)) * (powr<3>(p))) +
                          (-72.) * ((powr<4>(cosl1p2)) * ((powr<4>(l1)) * (powr<3>(p)))) +
                          (153.) * ((powr<2>(l1)) * (powr<5>(p))) + (198.) * (powr<7>(p)) +
                          (16.) * ((31.) * ((powr<5>(l1)) * (powr<2>(p))) + (19.) * ((powr<3>(l1)) * (powr<4>(p)))) *
                              (powr<3>(cosl1p2)) +
                          ((-144.) * ((powr<6>(l1)) * (p)) + (436.) * ((powr<4>(l1)) * (powr<3>(p))) +
                           (1302.) * ((powr<2>(l1)) * (powr<5>(p)))) *
                              (powr<2>(cosl1p2)) +
                          (-6.) *
                              ((12.) * (powr<7>(l1)) + (194.) * ((powr<5>(l1)) * (powr<2>(p))) +
                               (480.) * ((powr<3>(l1)) * (powr<4>(p))) + (231.) * ((l1) * (powr<6>(p)))) *
                              (cosl1p2)) *
                             (cosl1p1)) *
                            (cosl1p1)) *
                       ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                        ((powr<-1>((_repl3) *
                                       (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) +
                                   (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                       (ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
                         ((ZA3((0.816496580927726) *
                               (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))))) *
                          (ZA3((0.5773502691896258) *
                               (sqrt((2.) * (powr<2>(l1)) + (-2.) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) +
                                     (3.) * (powr<2>(p))))))))) +
                   (-3.) *
                       ((-54. + (53.) * (powr<2>(cosl1p1)) + (110.) * ((cosl1p1) * (cosl1p2)) +
                         (53.) * (powr<2>(cosl1p2))) *
                            (powr<2>(l1)) +
                        (-1.) *
                            ((-54.) * (cosl1p1) + (53.) * (powr<3>(cosl1p1)) + (-54.) * (cosl1p2) +
                             (163.) * ((powr<2>(cosl1p1)) * (cosl1p2)) + (163.) * ((cosl1p1) * (powr<2>(cosl1p2))) +
                             (53.) * (powr<3>(cosl1p2))) *
                            ((l1) * (p)) +
                        (33.) * (-1. + powr<2>(cosl1p1) + (2.) * ((cosl1p1) * (cosl1p2)) + powr<2>(cosl1p2)) *
                            (powr<2>(p))) *
                       (ZA4((0.5) * (sqrt((2.) * (powr<2>(l1)) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                          (3.) * (powr<2>(p))))))) *
                  ((powr<-1>(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                   ((powr<-2>((_repl1) * (_repl3) + (powr<2>(l1)) * (ZA(l1)))) *
                    ((powr<-1>((_repl3) * (RB(powr<2>(k),
                                              powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) +
                               (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                   (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)))))) *
                     (ZA3((0.816496580927726) *
                          (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))))))))) +
             (-0.0909090909090909) *
                 ((4.) * ((powr<3>(cosl1p1)) * (l1)) + (-4.) * ((powr<3>(cosl1p2)) * (l1)) + (-6.) * (p) +
                  (2.) * ((powr<2>(cosl1p2)) * (p)) +
                  (cosl1p1) * ((-6.) * ((cosl1p2) * (l1)) + (11.) * (p)) * (cosl1p2) +
                  ((6.) * ((cosl1p2) * (l1)) + (11.) * (p)) * (powr<2>(cosl1p1))) *
                 ((powr<2>(l1)) *
                  ((_repl2) * (_repl4) + (_repl1) * (dtZc(k)) +
                   (-50.) * (_repl4 + (-1.) * (Zc((1.02) * (k)))) * (_repl1)) *
                  ((powr<-1>(p)) *
                   ((ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (cosl1p1) * ((l1) * (p)) + powr<2>(p))))) *
                    ((ZAcbc((0.816496580927726) *
                            (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + cosl1p2) * (p) + powr<2>(p))))) *
                     ((ZAcbc((0.816496580927726) *
                             (sqrt(powr<2>(l1) + (l1) * ((2.) * (cosl1p1) + cosl1p2) * (p) + (1.5) * (powr<2>(p)))))) *
                      ((powr<-2>((_repl1) * (_repl4) + (powr<2>(l1)) * (Zc(l1)))) *
                       ((powr<-1>((_repl4) *
                                      (RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) +
                                  (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                      (Zc(sqrt(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
                        (powr<-1>((_repl4) * (RB(powr<2>(k), powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                                                 powr<2>(p))) +
                                  (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                      (Zc(sqrt(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                               powr<2>(p)))))))))))));
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto
    constant(const auto &p, const double &k,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA3,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZAcbc,
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
using DiFfRG::ZA3_kernel;