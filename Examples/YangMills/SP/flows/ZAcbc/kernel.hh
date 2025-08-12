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
      const auto _repl1 = RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p));
      const auto _repl2 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl3 = ZA(sqrt(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl4 = RB(powr<2>(k), powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p));
      const auto _repl5 = ZA(sqrt(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
      const auto _repl6 = ZA3((0.816496580927726) *
                              (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + (2.) * (cosl1p2)) * (p) + (1.5) * (powr<2>(p)))));
      const auto _repl7 = ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (cosl1p2) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl8 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + cosl1p2) * (p) + powr<2>(p))));
      const auto _repl9 = dtZc(k);
      const auto _repl10 = RB(powr<2>(k), powr<2>(l1));
      const auto _repl11 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _repl12 = Zc(k);
      const auto _repl13 = Zc((1.02) * (k));
      const auto _repl14 = Zc(l1);
      const auto _repl15 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl16 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _repl17 = ZA(l1);
      const auto _repl18 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))));
      const auto _repl19 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl20 = ZAcbc(
          (0.5773502691896258) *
          (sqrt((2.) * (powr<2>(l1)) + (-2.) * (cosl1p1 + (2.) * (cosl1p2)) * ((l1) * (p)) + (3.) * (powr<2>(p)))));
      const auto _repl21 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p));
      const auto _repl22 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl23 = RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p));
      const auto _repl24 = Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
      const auto _repl25 = RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl26 = ZA(sqrt(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl27 = ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (cosl1p1) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl28 =
          ZAcbc((0.816496580927726) *
                (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + (-1.) * (cosl1p2)) * (p) + (1.5) * (powr<2>(p)))));
      const auto _repl29 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl30 = ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl31 =
          ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))));
      const auto _repl32 = ZAcbc(
          (0.5773502691896258) *
          (sqrt((2.) * (powr<2>(l1)) + (-2.) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) + (3.) * (powr<2>(p)))));
      const auto _repl33 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))));
      const auto _repl34 =
          ZAcbc((0.816496580927726) *
                (sqrt(powr<2>(l1) + (l1) * ((-1.) * (cosl1p1) + cosl1p2) * (p) + (1.5) * (powr<2>(p)))));
      const auto _repl35 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl36 = ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (cosl1p1) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl37 = ZAcbc(
          (0.816496580927726) * (sqrt(powr<2>(l1) + (l1) * ((2.) * (cosl1p1) + cosl1p2) * (p) + (1.5) * (powr<2>(p)))));
      const auto _repl38 = Zc(sqrt(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
      return (3.) *
                 ((-0.25) *
                      ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                       (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                      ((_repl18) *
                       ((_repl19) *
                        ((_repl20) *
                         ((powr<-1>(1. + powr<6>(k))) *
                          ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                           ((powr<-1>((_repl12) * (_repl21) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl22))) *
                            (powr<-1>((_repl12) * (_repl23) +
                                      (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl24))))))))) +
                  (-1.5) *
                      ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                       (_repl10) * (_repl9)) *
                      ((_repl6) *
                       ((_repl7) *
                        ((_repl8) *
                         ((powr<4>(l1)) *
                          ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                           ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                            ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                             ((powr<-1>((_repl1) * (_repl2) +
                                        (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                              (powr<-1>((_repl2) * (_repl4) +
                                        (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl5))))))))))) +
                  (-1.) *
                      ((_repl18) *
                           ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                            (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                           ((_repl27) *
                            ((_repl28) *
                             ((powr<-1>(1. + powr<6>(k))) *
                              ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                               ((powr<-1>((_repl2) * (_repl25) +
                                          (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                                (powr<-1>((_repl12) * (_repl21) +
                                          (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                              (_repl22)))))))) +
                       (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                           ((_repl31) *
                            ((_repl32) *
                             ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                              ((powr<-1>((_repl2) * (_repl29) +
                                         (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                               (powr<-1>((_repl12) * (_repl23) +
                                         (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                             (_repl24))))))) +
                       (-50.) * ((-1.) * (_repl16) + _repl2) *
                           ((_repl10) *
                            ((_repl19) *
                             ((_repl31) *
                              ((_repl32) *
                               ((powr<6>(k)) *
                                ((powr<-1>(1. + powr<6>(k))) *
                                 ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                  ((powr<-1>((_repl2) * (_repl29) +
                                             (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                 (_repl30))) *
                                   (powr<-1>((_repl12) * (_repl23) +
                                             (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                 (_repl24)))))))))))) *
                      ((powr<2>(l1)) * (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))) +
                  ((-0.5) *
                       ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                        (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                       ((_repl18) *
                        ((_repl27) *
                         ((_repl28) *
                          ((powr<-1>(1. + powr<6>(k))) *
                           ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                            ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                             ((powr<-1>((_repl2) * (_repl25) +
                                        (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                              (powr<-1>((_repl12) * (_repl21) +
                                        (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                            (_repl22)))))))))) +
                   (-0.5) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                       ((_repl19) *
                        ((_repl31) *
                         ((_repl32) *
                          ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                           ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                            ((powr<-1>((_repl2) * (_repl29) +
                                       (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                             (powr<-1>((_repl12) * (_repl23) +
                                       (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                           (_repl24))))))))) +
                   (25.) * ((-1.) * (_repl16) + _repl2) *
                       ((_repl10) *
                        ((_repl19) *
                         ((_repl31) *
                          ((_repl32) *
                           ((powr<6>(k)) *
                            ((powr<-1>(1. + powr<6>(k))) *
                             ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                              ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                               ((powr<-1>((_repl2) * (_repl29) +
                                          (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                              (_repl30))) *
                                (powr<-1>((_repl12) * (_repl23) +
                                          (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl24)))))))))))) +
                   (-1.) *
                       ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                        (_repl10) * (_repl9)) *
                       ((_repl6) *
                        ((_repl7) *
                         ((_repl8) *
                          ((powr<2>(l1)) *
                           ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                            ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                             ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                              ((powr<-1>((_repl1) * (_repl2) +
                                         (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                               (powr<-1>((_repl2) * (_repl4) +
                                         (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                             (_repl5)))))))))))) *
                      (powr<2>(p))) *
                 (powr<2>(p)) +
             (((0.5) *
                   ((-50.) * (_repl12 + (-1.) * (_repl13)) *
                        ((_repl10) *
                         ((_repl33) *
                          ((_repl34) *
                           (powr<-1>((_repl12) * (_repl29) +
                                     (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35)))))) +
                    (_repl33) * ((_repl11) * (_repl12) + (_repl10) * (_repl9)) *
                        ((_repl34) *
                         (powr<-1>((_repl12) * (_repl29) +
                                   (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35)))) +
                    (-2.) *
                        ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                         (_repl10) * (_repl9)) *
                        ((_repl6) *
                         ((_repl8) *
                          ((powr<2>(l1)) *
                           ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                            (powr<-1>((_repl2) * (_repl4) +
                                      (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl5)))))))) *
                   ((_repl7) *
                    ((powr<3>(l1)) *
                     ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                      ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                       (powr<-1>((_repl1) * (_repl2) +
                                 (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))))))) +
               ((-4.5) *
                    ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                     (_repl10) * (_repl9)) *
                    ((_repl6) *
                     ((_repl7) *
                      ((_repl8) *
                       ((powr<3>(l1)) *
                        ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                          ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                           ((powr<-1>((_repl1) * (_repl2) +
                                      (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                            (powr<-1>((_repl2) * (_repl4) +
                                      (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl5))))))))))) +
                (0.5) *
                    ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                     (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                    ((powr<-1>(1. + powr<6>(k))) *
                     ((-5.) *
                          ((_repl18) *
                           ((_repl27) *
                            ((_repl28) *
                             ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                              ((powr<-1>((_repl2) * (_repl25) +
                                         (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                               (powr<-1>((_repl12) * (_repl21) +
                                         (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                             (_repl22)))))))) +
                      (7.) *
                          ((_repl19) *
                           ((_repl31) *
                            ((_repl32) *
                             ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                              ((powr<-1>((_repl2) * (_repl29) +
                                         (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                               (powr<-1>((_repl12) * (_repl23) +
                                         (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                             (_repl24))))))))) *
                     ((l1) * (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))))) *
                   (powr<2>(p)) +
               ((0.5) *
                    ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                     (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                    ((_repl18) *
                     ((_repl19) *
                      ((_repl20) *
                       ((powr<-1>(1. + powr<6>(k))) *
                        ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                         ((powr<-1>((_repl12) * (_repl21) +
                                    (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl22))) *
                          (powr<-1>((_repl12) * (_repl23) +
                                    (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                        (_repl24))))))))) +
                ((-1.) *
                     ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                      (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                     ((_repl18) *
                      ((_repl27) *
                       ((_repl28) *
                        ((powr<-1>(1. + powr<6>(k))) *
                         ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                          ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                           ((powr<-1>((_repl2) * (_repl25) +
                                      (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                            (powr<-1>((_repl12) * (_repl21) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                          (_repl22)))))))))) +
                 (-1.) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                     ((_repl19) *
                      ((_repl31) *
                       ((_repl32) *
                        ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                          ((powr<-1>((_repl2) * (_repl29) +
                                     (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                           (powr<-1>((_repl12) * (_repl23) +
                                     (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                         (_repl24))))))))) +
                 (50.) * ((-1.) * (_repl16) + _repl2) *
                     ((_repl10) *
                      ((_repl19) *
                       ((_repl31) *
                        ((_repl32) *
                         ((powr<6>(k)) *
                          ((powr<-1>(1. + powr<6>(k))) *
                           ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                            ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                             ((powr<-1>((_repl2) * (_repl29) +
                                        (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                              (powr<-1>((_repl12) * (_repl23) +
                                        (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl24)))))))))))) +
                 (0.5) *
                     ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                      (_repl10) * (_repl9)) *
                     ((_repl36) *
                      ((_repl37) *
                       ((_repl8) *
                        ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                          ((powr<-1>((_repl12) * (_repl25) +
                                     (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl38))) *
                           (powr<-1>((_repl2) * (_repl4) +
                                     (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                         (_repl5)))))))))) *
                    (powr<2>(l1))) *
                   (l1)) *
                  (p) +
              ((cosl1p1) *
                   ((-7.) *
                        ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                         (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                        ((_repl19) *
                         ((_repl31) *
                          ((_repl32) *
                           ((powr<-1>(1. + powr<6>(k))) *
                            ((l1) *
                             ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                              ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                               ((powr<-1>((_repl2) * (_repl29) +
                                          (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                              (_repl30))) *
                                (powr<-1>((_repl12) * (_repl23) +
                                          (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl24))))))))))) +
                    (2.) *
                        ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                         (_repl10) * (_repl9)) *
                        ((_repl6) *
                         ((_repl7) *
                          ((_repl8) *
                           ((powr<3>(l1)) *
                            ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                             ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                              ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                               ((powr<-1>((_repl1) * (_repl2) +
                                          (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                                (powr<-1>((_repl2) * (_repl4) +
                                          (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl5)))))))))))) *
                   (powr<3>(p)) +
               (((3.) *
                     ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                      (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                     ((_repl19) *
                      ((_repl31) *
                       ((_repl32) *
                        ((powr<-1>(1. + powr<6>(k))) *
                         ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                          ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                           ((powr<-1>((_repl2) * (_repl29) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                            (powr<-1>((_repl12) * (_repl23) +
                                      (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl24)))))))))) +
                 (_repl6) *
                     ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                      (_repl10) * (_repl9)) *
                     ((_repl7) *
                      ((_repl8) *
                       ((powr<2>(l1)) *
                        ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                          ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                           ((powr<-1>((_repl1) * (_repl2) +
                                      (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                            (powr<-1>((_repl2) * (_repl4) +
                                      (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl5))))))))))) *
                    (powr<2>(p)) +
                ((_repl18) *
                     ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                      (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                     ((_repl27) *
                      ((_repl28) *
                       ((powr<-1>(1. + powr<6>(k))) *
                        ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                          ((powr<-1>((_repl2) * (_repl25) +
                                     (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                           (powr<-1>((_repl12) * (_repl21) +
                                     (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                         (_repl22))))))))) +
                 (6.) *
                     ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                      (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                     ((_repl19) *
                      ((_repl31) *
                       ((_repl32) *
                        ((powr<-1>(1. + powr<6>(k))) *
                         ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                          ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                           ((powr<-1>((_repl2) * (_repl29) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                            (powr<-1>((_repl12) * (_repl23) +
                                      (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl24)))))))))) +
                 (0.5) *
                     ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                      (_repl10) * (_repl9)) *
                     ((_repl36) *
                      ((_repl37) *
                       ((_repl8) *
                        ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                          ((powr<-1>((_repl12) * (_repl25) +
                                     (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl38))) *
                           (powr<-1>((_repl2) * (_repl4) +
                                     (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                         (_repl5))))))))) +
                 (_repl7) *
                     ((50.) * (_repl12 + (-1.) * (_repl13)) *
                          ((_repl10) *
                           ((_repl33) *
                            ((_repl34) * (powr<-1>((_repl12) * (_repl29) +
                                                   (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                       (_repl35)))))) +
                      (-1.) * ((_repl11) * (_repl12) + (_repl10) * (_repl9)) *
                          ((_repl33) *
                           ((_repl34) *
                            (powr<-1>((_repl12) * (_repl29) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35))))) +
                      (3.) *
                          ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                           (_repl10) * (_repl9)) *
                          ((_repl6) *
                           ((_repl8) *
                            ((powr<2>(l1)) *
                             ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                              (powr<-1>((_repl2) * (_repl4) +
                                        (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl5)))))))) *
                     ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                      ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                       (powr<-1>((_repl1) * (_repl2) +
                                 (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3)))))) *
                    (powr<2>(l1))) *
                   (powr<2>(p))) *
                  (cosl1p1)) *
                 (cosl1p1) +
             (((-1.) *
                   ((50.) * (_repl12 + (-1.) * (_repl13)) *
                        ((_repl10) *
                         ((_repl33) *
                          ((_repl34) *
                           (powr<-1>((_repl12) * (_repl29) +
                                     (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35)))))) +
                    (-1.) * ((_repl11) * (_repl12) + (_repl10) * (_repl9)) *
                        ((_repl33) *
                         ((_repl34) *
                          (powr<-1>((_repl12) * (_repl29) +
                                    (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35))))) +
                    (2.) *
                        ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                         (_repl10) * (_repl9)) *
                        ((_repl6) *
                         ((_repl8) *
                          ((powr<2>(l1)) *
                           ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                            (powr<-1>((_repl2) * (_repl4) +
                                      (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl5)))))))) *
                   ((_repl7) *
                    ((powr<3>(l1)) *
                     ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                      ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                       (powr<-1>((_repl1) * (_repl2) +
                                 (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))))))) +
               ((-9.) *
                    ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                     (_repl10) * (_repl9)) *
                    ((_repl6) *
                     ((_repl7) *
                      ((_repl8) *
                       ((powr<3>(l1)) *
                        ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                          ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                           ((powr<-1>((_repl1) * (_repl2) +
                                      (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                            (powr<-1>((_repl2) * (_repl4) +
                                      (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl5))))))))))) +
                (l1) *
                    ((_repl18) *
                         ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                          (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                         ((_repl27) *
                          ((_repl28) *
                           ((powr<-1>(1. + powr<6>(k))) *
                            ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                             ((powr<-1>((_repl2) * (_repl25) +
                                        (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                              (powr<-1>((_repl12) * (_repl21) +
                                        (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                            (_repl22)))))))) +
                     (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                         ((_repl31) *
                          ((_repl32) *
                           ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                            ((powr<-1>((_repl2) * (_repl29) +
                                       (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                             (powr<-1>((_repl12) * (_repl23) +
                                       (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                           (_repl24))))))) +
                     (-50.) * ((-1.) * (_repl16) + _repl2) *
                         ((_repl10) *
                          ((_repl19) *
                           ((_repl31) *
                            ((_repl32) *
                             ((powr<6>(k)) *
                              ((powr<-1>(1. + powr<6>(k))) *
                               ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                ((powr<-1>((_repl2) * (_repl29) +
                                           (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                               (_repl30))) *
                                 (powr<-1>((_repl12) * (_repl23) +
                                           (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                               (_repl24)))))))))))) *
                    (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))) *
                   (powr<2>(p)) +
               (-1.) *
                   ((-1.) *
                        ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                         (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                        ((_repl18) *
                         ((_repl19) *
                          ((_repl20) *
                           ((powr<-1>(1. + powr<6>(k))) *
                            ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                             ((powr<-1>((_repl12) * (_repl21) +
                                        (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl22))) *
                              (powr<-1>((_repl12) * (_repl23) +
                                        (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl24))))))))) +
                    ((-1.) *
                         ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                          (_repl10) * (_repl9)) *
                         ((_repl36) *
                          ((_repl37) *
                           ((_repl8) *
                            ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                             ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                              ((powr<-1>((_repl12) * (_repl25) +
                                         (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl38))) *
                               (powr<-1>((_repl2) * (_repl4) +
                                         (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                             (_repl5))))))))) +
                     (2.) *
                         ((_repl18) *
                              ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                               (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                              ((_repl27) *
                               ((_repl28) *
                                ((powr<-1>(1. + powr<6>(k))) *
                                 ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                  ((powr<-1>((_repl2) * (_repl25) +
                                             (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                 (_repl26))) *
                                   (powr<-1>((_repl12) * (_repl21) +
                                             (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                 (_repl22)))))))) +
                          (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                              ((_repl31) *
                               ((_repl32) *
                                ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                 ((powr<-1>((_repl2) * (_repl29) +
                                            (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl30))) *
                                  (powr<-1>((_repl12) * (_repl23) +
                                            (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl24))))))) +
                          (-50.) * ((-1.) * (_repl16) + _repl2) *
                              ((_repl10) *
                               ((_repl19) *
                                ((_repl31) *
                                 ((_repl32) *
                                  ((powr<6>(k)) *
                                   ((powr<-1>(1. + powr<6>(k))) *
                                    ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                     ((powr<-1>((_repl2) * (_repl29) +
                                                (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                    (_repl30))) *
                                      (powr<-1>(
                                          (_repl12) * (_repl23) +
                                          (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl24)))))))))))) *
                         (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))) *
                        (powr<2>(l1))) *
                   (l1)) *
                  (p) +
              (((1.5) *
                    ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                     (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                    ((_repl18) *
                     ((_repl19) *
                      ((_repl20) *
                       ((powr<-1>(1. + powr<6>(k))) *
                        ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                         ((powr<-1>((_repl12) * (_repl21) +
                                    (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl22))) *
                          (powr<-1>((_repl12) * (_repl23) +
                                    (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                        (_repl24))))))))) +
                ((-3.) *
                     ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                      (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                     ((_repl18) *
                      ((_repl27) *
                       ((_repl28) *
                        ((powr<-1>(1. + powr<6>(k))) *
                         ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                          ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                           ((powr<-1>((_repl2) * (_repl25) +
                                      (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                            (powr<-1>((_repl12) * (_repl21) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                          (_repl22)))))))))) +
                 (7.) *
                     ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                      (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                     ((_repl19) *
                      ((_repl31) *
                       ((_repl32) *
                        ((powr<-1>(1. + powr<6>(k))) *
                         ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                          ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                           ((powr<-1>((_repl2) * (_repl29) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                            (powr<-1>((_repl12) * (_repl23) +
                                      (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl24)))))))))) +
                 (0.5) *
                     ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                      (_repl10) * (_repl9)) *
                     ((_repl36) *
                      ((_repl37) *
                       ((_repl8) *
                        ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                          ((powr<-1>((_repl12) * (_repl25) +
                                     (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl38))) *
                           (powr<-1>((_repl2) * (_repl4) +
                                     (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                         (_repl5))))))))) +
                 (0.5) *
                     ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                      (_repl10) * (_repl9)) *
                     ((_repl7) *
                      ((-5.) * (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                           ((_repl2) * ((_repl33) * ((_repl34) * (_repl4)))) +
                       (-5.) *
                           ((_repl33) *
                            ((_repl34) * ((_repl5) * (powr<2>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                                              powr<2>(p)))))) +
                       (6.) *
                           ((_repl12) * (_repl29) +
                            (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35)) *
                           ((_repl6) * ((_repl8) * (powr<2>(l1))))) *
                      ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                       ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                        ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                         ((powr<-1>((_repl12) * (_repl29) +
                                    (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35))) *
                          ((powr<-1>((_repl1) * (_repl2) +
                                     (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                           (powr<-1>((_repl2) * (_repl4) +
                                     (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                         (_repl5)))))))))) *
                    (powr<2>(l1)) +
                ((5.5) *
                     ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                      (_repl10) * (_repl9)) *
                     ((_repl6) *
                      ((_repl7) *
                       ((_repl8) *
                        ((powr<2>(l1)) *
                         ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                          ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                           ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                            ((powr<-1>((_repl1) * (_repl2) +
                                       (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                             (powr<-1>((_repl2) * (_repl4) +
                                       (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                           (_repl5))))))))))) +
                 (3.) *
                     ((-1.) *
                          ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                           (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                          ((_repl18) *
                           ((_repl27) *
                            ((_repl28) *
                             ((powr<-1>(1. + powr<6>(k))) *
                              ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                               ((powr<-1>((_repl2) * (_repl25) +
                                          (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                                (powr<-1>((_repl12) * (_repl21) +
                                          (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                              (_repl22))))))))) +
                      (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                          ((_repl31) *
                           ((_repl32) *
                            ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                             ((powr<-1>((_repl2) * (_repl29) +
                                        (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                              (powr<-1>((_repl12) * (_repl23) +
                                        (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl24))))))) +
                      (-50.) * ((-1.) * (_repl16) + _repl2) *
                          ((_repl10) *
                           ((_repl19) *
                            ((_repl31) *
                             ((_repl32) *
                              ((powr<6>(k)) *
                               ((powr<-1>(1. + powr<6>(k))) *
                                ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                 ((powr<-1>((_repl2) * (_repl29) +
                                            (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl30))) *
                                  (powr<-1>((_repl12) * (_repl23) +
                                            (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl24)))))))))))) *
                     (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))) *
                    (powr<2>(p))) *
                   (powr<2>(p)) +
               ((2.) *
                    ((_repl6) *
                         ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                          (_repl10) * (_repl9)) *
                         ((_repl7) *
                          ((_repl8) *
                           ((powr<4>(l1)) *
                            ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                             ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                              ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                               ((powr<-1>((_repl1) * (_repl2) +
                                          (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                                (powr<-1>((_repl2) * (_repl4) +
                                          (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl5)))))))))) +
                     (-1.) *
                         ((-1.) *
                              ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                               (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                              ((_repl18) *
                               ((_repl27) *
                                ((_repl28) *
                                 ((powr<-1>(1. + powr<6>(k))) *
                                  ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                   ((powr<-1>((_repl2) * (_repl25) +
                                              (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl26))) *
                                    (powr<-1>((_repl12) * (_repl21) +
                                              (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl22))))))))) +
                          (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                              ((_repl31) *
                               ((_repl32) *
                                ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                 ((powr<-1>((_repl2) * (_repl29) +
                                            (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl30))) *
                                  (powr<-1>((_repl12) * (_repl23) +
                                            (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl24))))))) +
                          (-50.) * ((-1.) * (_repl16) + _repl2) *
                              ((_repl10) *
                               ((_repl19) *
                                ((_repl31) *
                                 ((_repl32) *
                                  ((powr<6>(k)) *
                                   ((powr<-1>(1. + powr<6>(k))) *
                                    ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                     ((powr<-1>((_repl2) * (_repl29) +
                                                (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                    (_repl30))) *
                                      (powr<-1>(
                                          (_repl12) * (_repl23) +
                                          (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl24)))))))))))) *
                         ((powr<2>(l1)) * (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))))) *
                    ((cosl1p1) * (powr<2>(p))) +
                ((_repl7) *
                     ((50.) * (_repl12 + (-1.) * (_repl13)) *
                          ((_repl10) *
                           ((_repl33) *
                            ((_repl34) * (powr<-1>((_repl12) * (_repl29) +
                                                   (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                       (_repl35)))))) +
                      (-1.) * ((_repl11) * (_repl12) + (_repl10) * (_repl9)) *
                          ((_repl33) *
                           ((_repl34) *
                            (powr<-1>((_repl12) * (_repl29) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35))))) +
                      (2.) *
                          ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                           (_repl10) * (_repl9)) *
                          ((_repl6) *
                           ((_repl8) *
                            ((powr<2>(l1)) *
                             ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                              (powr<-1>((_repl2) * (_repl4) +
                                        (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl5)))))))) *
                     ((powr<3>(l1)) *
                      ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                       ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                        (powr<-1>((_repl1) * (_repl2) +
                                  (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3)))))) +
                 ((12.) *
                      ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                       (_repl10) * (_repl9)) *
                      ((_repl6) *
                       ((_repl7) *
                        ((_repl8) *
                         ((powr<3>(l1)) *
                          ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                           ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                            ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                             ((powr<-1>((_repl1) * (_repl2) +
                                        (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                              (powr<-1>((_repl2) * (_repl4) +
                                        (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl5))))))))))) +
                  (powr<-1>(1. + powr<6>(k))) *
                      ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                       (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                      ((l1) *
                       ((-5.) *
                            ((_repl18) *
                             ((_repl27) *
                              ((_repl28) * ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                            ((powr<-1>((_repl2) * (_repl25) +
                                                       (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                           (_repl26))) *
                                             (powr<-1>((_repl12) * (_repl21) +
                                                       (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                           (_repl22)))))))) +
                        (-9.) * ((_repl19) *
                                 ((_repl31) *
                                  ((_repl32) *
                                   ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                    ((powr<-1>((_repl2) * (_repl29) +
                                               (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                   (_repl30))) *
                                     (powr<-1>((_repl12) * (_repl23) +
                                               (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                   (_repl24))))))))) *
                       (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))))) *
                     (powr<2>(p)) +
                 ((-1.) *
                      ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                       (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                      ((_repl18) *
                       ((_repl19) *
                        ((_repl20) *
                         ((powr<-1>(1. + powr<6>(k))) *
                          ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                           ((powr<-1>((_repl12) * (_repl21) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl22))) *
                            (powr<-1>((_repl12) * (_repl23) +
                                      (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                          (_repl24))))))))) +
                  ((-1.) *
                       ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                        (_repl10) * (_repl9)) *
                       ((_repl36) *
                        ((_repl37) *
                         ((_repl8) *
                          ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                           ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                            ((powr<-1>((_repl12) * (_repl25) +
                                       (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl38))) *
                             (powr<-1>((_repl2) * (_repl4) +
                                       (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                           (_repl5))))))))) +
                   (2.) *
                       ((_repl18) *
                            ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                             (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                            ((_repl27) *
                             ((_repl28) * ((powr<-1>(1. + powr<6>(k))) *
                                           ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                            ((powr<-1>((_repl2) * (_repl25) +
                                                       (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                           (_repl26))) *
                                             (powr<-1>((_repl12) * (_repl21) +
                                                       (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                           (_repl22)))))))) +
                        (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                            ((_repl31) *
                             ((_repl32) *
                              ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                               ((powr<-1>((_repl2) * (_repl29) +
                                          (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                              (_repl30))) *
                                (powr<-1>((_repl12) * (_repl23) +
                                          (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl24))))))) +
                        (-50.) * ((-1.) * (_repl16) + _repl2) *
                            ((_repl10) *
                             ((_repl19) *
                              ((_repl31) *
                               ((_repl32) *
                                ((powr<6>(k)) *
                                 ((powr<-1>(1. + powr<6>(k))) *
                                  ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                   ((powr<-1>((_repl2) * (_repl29) +
                                              (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl30))) *
                                    (powr<-1>((_repl12) * (_repl23) +
                                              (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                  (_repl24)))))))))))) *
                       (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))) *
                      (powr<2>(l1))) *
                     (l1)) *
                    (p)) *
                   (cosl1p1)) *
                  (cosl1p1) +
              (((1.5) *
                    ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                     (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                    ((_repl18) *
                     ((_repl19) *
                      ((_repl20) *
                       ((powr<-1>(1. + powr<6>(k))) *
                        ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                         ((powr<-1>((_repl12) * (_repl21) +
                                    (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl22))) *
                          (powr<-1>((_repl12) * (_repl23) +
                                    (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                        (_repl24))))))))) +
                ((-1.) *
                     ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                      (_repl10) * (_repl9)) *
                     ((_repl36) *
                      ((_repl37) *
                       ((_repl8) *
                        ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                         ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                          ((powr<-1>((_repl12) * (_repl25) +
                                     (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl38))) *
                           (powr<-1>((_repl2) * (_repl4) +
                                     (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                         (_repl5))))))))) +
                 (5.5) *
                     ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                      (_repl10) * (_repl9)) *
                     ((_repl6) *
                      ((_repl7) *
                       ((_repl8) *
                        ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                         ((powr<2>(p)) *
                          ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                           ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                            ((powr<-1>((_repl1) * (_repl2) +
                                       (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                             (powr<-1>((_repl2) * (_repl4) +
                                       (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                           (_repl5))))))))))) +
                 (2.) *
                     ((_repl18) *
                          ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                           (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                          ((_repl27) *
                           ((_repl28) *
                            ((powr<-1>(1. + powr<6>(k))) *
                             ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                              ((powr<-1>((_repl2) * (_repl25) +
                                         (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl26))) *
                               (powr<-1>((_repl12) * (_repl21) +
                                         (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                             (_repl22)))))))) +
                      (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                          ((_repl31) *
                           ((_repl32) *
                            ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                             ((powr<-1>((_repl2) * (_repl29) +
                                        (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl30))) *
                              (powr<-1>((_repl12) * (_repl23) +
                                        (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl24))))))) +
                      (-50.) * ((-1.) * (_repl16) + _repl2) *
                          ((_repl10) *
                           ((_repl19) *
                            ((_repl31) *
                             ((_repl32) *
                              ((powr<6>(k)) *
                               ((powr<-1>(1. + powr<6>(k))) *
                                ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                 ((powr<-1>((_repl2) * (_repl29) +
                                            (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl30))) *
                                  (powr<-1>((_repl12) * (_repl23) +
                                            (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl24)))))))))))) *
                     (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) +
                 (_repl7) *
                     ((50.) * (_repl12 + (-1.) * (_repl13)) *
                          ((_repl10) *
                           ((_repl33) *
                            ((_repl34) * (powr<-1>((_repl12) * (_repl29) +
                                                   (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                       (_repl35)))))) +
                      (-1.) * ((_repl11) * (_repl12) + (_repl10) * (_repl9)) *
                          ((_repl33) *
                           ((_repl34) *
                            (powr<-1>((_repl12) * (_repl29) +
                                      (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35))))) +
                      (3.) *
                          ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                           (_repl10) * (_repl9)) *
                          ((_repl6) *
                           ((_repl8) *
                            ((powr<2>(l1)) *
                             ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                              (powr<-1>((_repl2) * (_repl4) +
                                        (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl5)))))))) *
                     ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                      ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                       (powr<-1>((_repl1) * (_repl2) +
                                 (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3)))))) *
                    (powr<2>(l1))) *
                   (powr<2>(p)) +
               (2.) *
                   ((4.) *
                        ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                         (_repl10) * (_repl9)) *
                        ((_repl6) *
                         ((_repl7) *
                          ((_repl8) *
                           ((cosl1p2) *
                            ((powr<4>(l1)) *
                             ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                              ((powr<2>(p)) *
                               ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                                ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                                 ((powr<-1>((_repl1) * (_repl2) +
                                            (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl3))) *
                                  (powr<-1>((_repl2) * (_repl4) +
                                            (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl5))))))))))))) +
                    (2.) *
                        ((4.) *
                             ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                              (_repl10) * (_repl9)) *
                             ((_repl6) *
                              ((_repl7) *
                               ((_repl8) *
                                ((powr<4>(l1)) *
                                 ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                                  ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                                   ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                                    ((powr<-1>((_repl1) * (_repl2) +
                                               (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                   (_repl3))) *
                                     (powr<-1>((_repl2) * (_repl4) +
                                               (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                   (_repl5))))))))))) +
                         (-1.) *
                             ((-1.) *
                                  ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                                   (_repl10) * (1. + powr<6>(k)) * (_repl15) +
                                   (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                                  ((_repl18) *
                                   ((_repl27) *
                                    ((_repl28) *
                                     ((powr<-1>(1. + powr<6>(k))) *
                                      ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                       ((powr<-1>((_repl2) * (_repl25) +
                                                  (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                      (_repl26))) *
                                        (powr<-1>((_repl12) * (_repl21) +
                                                  (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                      (_repl22))))))))) +
                              (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                                  ((_repl31) *
                                   ((_repl32) *
                                    ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                     ((powr<-1>((_repl2) * (_repl29) +
                                                (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                    (_repl30))) *
                                      (powr<-1>(
                                          (_repl12) * (_repl23) +
                                          (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl24))))))) +
                              (-50.) * ((-1.) * (_repl16) + _repl2) *
                                  ((_repl10) *
                                   ((_repl19) *
                                    ((_repl31) *
                                     ((_repl32) *
                                      ((powr<6>(k)) *
                                       ((powr<-1>(1. + powr<6>(k))) *
                                        ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                         ((powr<-1>((_repl2) * (_repl29) +
                                                    (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                        (_repl30))) *
                                          (powr<-1>(
                                              (_repl12) * (_repl23) +
                                              (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                  (_repl24)))))))))))) *
                             ((powr<2>(l1)) * (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))))) *
                        ((cosl1p1) * (powr<2>(p))) +
                    ((8.) *
                         ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                          (_repl10) * (_repl9)) *
                         ((_repl6) *
                          ((_repl7) *
                           ((_repl8) *
                            ((powr<3>(l1)) *
                             ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                              ((powr<2>(p)) *
                               ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                                ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                                 ((powr<-1>((_repl1) * (_repl2) +
                                            (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl3))) *
                                  (powr<-1>((_repl2) * (_repl4) +
                                            (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl5)))))))))))) +
                     (_repl7) *
                         ((50.) * (_repl12 + (-1.) * (_repl13)) *
                              ((_repl10) *
                               ((_repl33) *
                                ((_repl34) * (powr<-1>((_repl12) * (_repl29) +
                                                       (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                           (_repl35)))))) +
                          (-1.) * ((_repl11) * (_repl12) + (_repl10) * (_repl9)) *
                              ((_repl33) *
                               ((_repl34) * (powr<-1>((_repl12) * (_repl29) +
                                                      (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                          (_repl35))))) +
                          (2.) *
                              ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                               (_repl10) * (_repl9)) *
                              ((_repl6) *
                               ((_repl8) *
                                ((powr<2>(l1)) *
                                 ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                                  (powr<-1>((_repl2) * (_repl4) +
                                            (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl5)))))))) *
                         ((powr<3>(l1)) *
                          ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                           ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                            (powr<-1>((_repl1) * (_repl2) +
                                      (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3)))))) +
                     ((-1.) *
                          ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                           (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                          ((_repl18) *
                           ((_repl19) *
                            ((_repl20) *
                             ((powr<-1>(1. + powr<6>(k))) *
                              ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                               ((powr<-1>((_repl12) * (_repl21) +
                                          (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                              (_repl22))) *
                                (powr<-1>((_repl12) * (_repl23) +
                                          (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl24))))))))) +
                      ((-1.) *
                           ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                            (_repl10) * (_repl9)) *
                           ((_repl36) *
                            ((_repl37) *
                             ((_repl8) *
                              ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                               ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                                ((powr<-1>((_repl12) * (_repl25) +
                                           (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                               (_repl38))) *
                                 (powr<-1>((_repl2) * (_repl4) +
                                           (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                               (_repl5))))))))) +
                       (2.) *
                           ((_repl18) *
                                ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                                 (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                                ((_repl27) *
                                 ((_repl28) *
                                  ((powr<-1>(1. + powr<6>(k))) *
                                   ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                    ((powr<-1>((_repl2) * (_repl25) +
                                               (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                   (_repl26))) *
                                     (powr<-1>((_repl12) * (_repl21) +
                                               (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                   (_repl22)))))))) +
                            (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                                ((_repl31) *
                                 ((_repl32) *
                                  ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                   ((powr<-1>((_repl2) * (_repl29) +
                                              (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl30))) *
                                    (powr<-1>((_repl12) * (_repl23) +
                                              (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                  (_repl24))))))) +
                            (-50.) * ((-1.) * (_repl16) + _repl2) *
                                ((_repl10) *
                                 ((_repl19) *
                                  ((_repl31) *
                                   ((_repl32) *
                                    ((powr<6>(k)) *
                                     ((powr<-1>(1. + powr<6>(k))) *
                                      ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                       ((powr<-1>((_repl2) * (_repl29) +
                                                  (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                      (_repl30))) *
                                        (powr<-1>(
                                            (_repl12) * (_repl23) +
                                            (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl24)))))))))))) *
                           (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))) *
                          (powr<2>(l1))) *
                         (l1)) *
                        (p)) *
                   (cosl1p2) +
               ((2.) *
                    ((5.) *
                         ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                          (_repl10) * (_repl9)) *
                         ((_repl6) *
                          ((_repl7) *
                           ((_repl8) *
                            ((powr<4>(l1)) *
                             ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                              ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                               ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                                ((powr<-1>((_repl1) * (_repl2) +
                                           (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3))) *
                                 (powr<-1>((_repl2) * (_repl4) +
                                           (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                               (_repl5))))))))))) +
                     (-3.) *
                         ((-1.) *
                              ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                               (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                              ((_repl18) *
                               ((_repl27) *
                                ((_repl28) *
                                 ((powr<-1>(1. + powr<6>(k))) *
                                  ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                   ((powr<-1>((_repl2) * (_repl25) +
                                              (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl26))) *
                                    (powr<-1>((_repl12) * (_repl21) +
                                              (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl22))))))))) +
                          (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                              ((_repl31) *
                               ((_repl32) *
                                ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                 ((powr<-1>((_repl2) * (_repl29) +
                                            (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl30))) *
                                  (powr<-1>((_repl12) * (_repl23) +
                                            (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl24))))))) +
                          (-50.) * ((-1.) * (_repl16) + _repl2) *
                              ((_repl10) *
                               ((_repl19) *
                                ((_repl31) *
                                 ((_repl32) *
                                  ((powr<6>(k)) *
                                   ((powr<-1>(1. + powr<6>(k))) *
                                    ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                     ((powr<-1>((_repl2) * (_repl29) +
                                                (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                    (_repl30))) *
                                      (powr<-1>(
                                          (_repl12) * (_repl23) +
                                          (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                              (_repl24)))))))))))) *
                         ((powr<2>(l1)) * (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))))) *
                    ((cosl1p1) * (powr<2>(p))) +
                ((2.) *
                     ((12.) *
                          ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                           (_repl10) * (_repl9)) *
                          ((_repl6) *
                           ((_repl7) *
                            ((_repl8) *
                             ((powr<3>(l1)) *
                              ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                               ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                                ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                                 ((powr<-1>((_repl1) * (_repl2) +
                                            (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl3))) *
                                  (powr<-1>((_repl2) * (_repl4) +
                                            (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                (_repl5))))))))))) +
                      (l1) *
                          ((_repl18) *
                               ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                                (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                               ((_repl27) *
                                ((_repl28) *
                                 ((powr<-1>(1. + powr<6>(k))) *
                                  ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                   ((powr<-1>((_repl2) * (_repl25) +
                                              (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl26))) *
                                    (powr<-1>((_repl12) * (_repl21) +
                                              (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl22)))))))) +
                           (-1.) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                               ((_repl19) *
                                ((_repl31) *
                                 ((_repl32) *
                                  ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                   ((powr<-1>((_repl2) * (_repl29) +
                                              (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                  (_repl30))) *
                                    (powr<-1>((_repl12) * (_repl23) +
                                              (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                  (_repl24)))))))) +
                           (50.) * ((-1.) * (_repl16) + _repl2) *
                               ((_repl10) *
                                ((_repl19) *
                                 ((_repl31) *
                                  ((_repl32) *
                                   ((powr<6>(k)) *
                                    ((powr<-1>(1. + powr<6>(k))) *
                                     ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                      ((powr<-1>((_repl2) * (_repl29) +
                                                 (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                     (_repl30))) *
                                       (powr<-1>(
                                           (_repl12) * (_repl23) +
                                           (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                               (_repl24)))))))))))) *
                          (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))) *
                     (powr<2>(p)) +
                 ((_repl7) *
                      ((50.) * (_repl12 + (-1.) * (_repl13)) *
                           ((_repl10) *
                            ((_repl33) *
                             ((_repl34) * (powr<-1>((_repl12) * (_repl29) +
                                                    (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                        (_repl35)))))) +
                       (-1.) * ((_repl11) * (_repl12) + (_repl10) * (_repl9)) *
                           ((_repl33) *
                            ((_repl34) *
                             (powr<-1>((_repl12) * (_repl29) +
                                       (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl35))))) +
                       (2.) *
                           ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                            (_repl10) * (_repl9)) *
                           ((_repl6) *
                            ((_repl8) *
                             ((powr<2>(l1)) *
                              ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                               (powr<-1>((_repl2) * (_repl4) +
                                         (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                             (_repl5)))))))) *
                      ((powr<3>(l1)) *
                       ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                        ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                         (powr<-1>((_repl1) * (_repl2) +
                                   (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl3)))))) +
                  ((-1.) *
                       ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                        (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                       ((_repl18) *
                        ((_repl19) *
                         ((_repl20) *
                          ((powr<-1>(1. + powr<6>(k))) *
                           ((powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1)))) *
                            ((powr<-1>((_repl12) * (_repl21) +
                                       (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) * (_repl22))) *
                             (powr<-1>((_repl12) * (_repl23) +
                                       (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                           (_repl24))))))))) +
                   ((-1.) *
                        ((_repl11) * (_repl12) + (-50.) * (_repl12 + (-1.) * (_repl13)) * (_repl10) +
                         (_repl10) * (_repl9)) *
                        ((_repl36) *
                         ((_repl37) *
                          ((_repl8) *
                           ((powr<-2>((_repl10) * (_repl12) + (_repl14) * (powr<2>(l1)))) *
                            ((powr<-1>(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                             ((powr<-1>((_repl12) * (_repl25) +
                                        (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) * (_repl38))) *
                              (powr<-1>((_repl2) * (_repl4) +
                                        (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                            (_repl5))))))))) +
                    (2.) *
                        ((_repl18) *
                             ((-50.) * ((-1.) * (_repl16) + _repl2) * ((_repl10) * (powr<6>(k))) +
                              (_repl10) * (1. + powr<6>(k)) * (_repl15) + (_repl11) * (1. + powr<6>(k)) * (_repl2)) *
                             ((_repl27) *
                              ((_repl28) *
                               ((powr<-1>(1. + powr<6>(k))) *
                                ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                 ((powr<-1>((_repl2) * (_repl25) +
                                            (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl26))) *
                                  (powr<-1>((_repl12) * (_repl21) +
                                            (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                                                (_repl22)))))))) +
                         (_repl19) * ((_repl10) * (_repl15) + (_repl11) * (_repl2)) *
                             ((_repl31) *
                              ((_repl32) *
                               ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                ((powr<-1>((_repl2) * (_repl29) +
                                           (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                               (_repl30))) *
                                 (powr<-1>((_repl12) * (_repl23) +
                                           (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                               (_repl24))))))) +
                         (-50.) * ((-1.) * (_repl16) + _repl2) *
                             ((_repl10) *
                              ((_repl19) *
                               ((_repl31) *
                                ((_repl32) *
                                 ((powr<6>(k)) *
                                  ((powr<-1>(1. + powr<6>(k))) *
                                   ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                                    ((powr<-1>((_repl2) * (_repl29) +
                                               (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                                                   (_repl30))) *
                                     (powr<-1>((_repl12) * (_repl23) +
                                               (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                                                   (_repl24)))))))))))) *
                        (powr<-2>((_repl10) * (_repl2) + (_repl17) * (powr<2>(l1))))) *
                       (powr<2>(l1))) *
                      (l1)) *
                     (3.)) *
                    (p)) *
                   (cosl1p1)) *
                  (cosl1p2)) *
                 (cosl1p2);
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
using DiFfRG::ZAcbc_kernel;