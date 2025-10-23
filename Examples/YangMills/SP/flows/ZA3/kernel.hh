#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA3_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const double &l1, const double &cos1, const double &cos2, const double &p, const double &k,
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
      const auto _repl1 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl2 = RB(powr<2>(k), powr<2>(l1));
      const auto _repl3 = Zc(k);
      const auto _repl4 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _repl5 = RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p));
      const auto _repl6 = ZA(l1);
      const auto _repl7 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _repl8 =
          ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl9 = ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
      const auto _repl10 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl11 = powr<2>(l1);
      const auto _repl12 = powr<2>(p);
      const auto _repl13 = powr<6>(k);
      const auto _repl14 = powr<2>(cosl1p2);
      const auto _repl15 = powr<2>(k);
      const auto _repl16 = powr<3>(p);
      const auto _repl17 = powr<4>(l1);
      const auto _repl18 = powr<3>(l1);
      const auto _repl19 = powr<5>(l1);
      const auto _repl20 = powr<4>(p);
      const auto _repl21 = powr<6>(l1);
      const auto _repl22 = powr<5>(p);
      const auto _repl23 = powr<2>(cosl1p1);
      const auto _repl24 = powr<4>(cosl1p2);
      const auto _repl25 = powr<6>(p);
      const auto _repl26 = powr<7>(l1);
      const auto _repl27 = powr<3>(cosl1p2);
      const auto _repl28 = powr<3>(cosl1p1);
      const auto _repl29 = powr<7>(p);
      const auto _repl30 = powr<-2>((_repl1) * (_repl2) + (_repl11) * (_repl6));
      return (18.) *
                 ((_repl10) * (1. + (1.) * (_repl13)) * (_repl2) + (_repl1) * (1. + _repl13) * (_repl4) +
                  (_repl13) * ((-50.) * (_repl1) + (50.) * (_repl7)) * (_repl2)) *
                 ((powr<-1>(1. + _repl13)) *
                  ((-1.878787878787879) * ((_repl16) * (_repl17)) + (-1.909090909090909) * ((_repl11) * (_repl22)) +
                   ((1.01010101010101) * ((_repl12) * (_repl19)) + (0.6464646464646465) * ((_repl18) * (_repl20)) +
                    (0.2424242424242424) * (_repl26)) *
                       (_repl27) +
                   (-0.4999999999999998) * (_repl29) +
                   (0.1616161616161616) * ((_repl16) * ((_repl17) * (powr<6>(cosl1p1)))) +
                   (0.0808080808080808) * ((_repl12) * ((_repl19) * (powr<5>(cosl1p2)))) +
                   ((0.7272727272727272) * ((_repl12) * (_repl19)) + (2.434343434343434) * ((_repl18) * (_repl20)) +
                    (-3.588599675556062e-17) * (_repl26) + (1.166666666666666) * ((_repl25) * (l1))) *
                       (cosl1p2) +
                   (-0.3636363636363637) * ((_repl21) * (p)) +
                   ((-1.212121212121212) * ((_repl16) * (_repl17)) + (-0.4848484848484849) * ((_repl21) * (p))) *
                       (_repl24) +
                   ((0.2828282828282828) * ((_repl16) * (_repl17)) + (-0.3636363636363637) * ((_repl11) * (_repl22)) +
                    (0.1212121212121213) * ((_repl21) * (p))) *
                       (_repl14) +
                   (_repl11) *
                       ((_repl11) * (6.282828282828283 + (0.2020202020202019) * (_repl14)) * (_repl12) +
                        (0.96969696969697) * (_repl17) + (6.333333333333333) * (_repl20) +
                        (-5.656565656565657) * ((_repl16) * ((cosl1p2) * (l1))) +
                        (-2.626262626262626) * ((_repl18) * ((cosl1p2) * (p)))) *
                       ((powr<4>(cosl1p1)) * (p)) +
                   (_repl28) *
                       ((_repl12) * (-3.919191919191919 + (-0.0808080808080808) * (_repl14)) * (_repl17) +
                        (_repl11) * (-9.6969696969697 + (-2.747474747474747) * (_repl14)) * (_repl20) +
                        (-0.2424242424242424) * (_repl21) + (-4.666666666666667) * (_repl25) +
                        (_repl16) * (12.56565656565657 + (-0.4040404040404041) * (_repl14)) * ((_repl18) * (cosl1p2)) +
                        (12.66666666666667) * ((_repl22) * ((cosl1p2) * (l1))) +
                        (1.93939393939394) * ((_repl19) * ((cosl1p2) * (p)))) *
                       (l1) +
                   ((_repl18) *
                        (4.868686868686869 + (-3.555555555555555) * (_repl14) + (0.888888888888889) * (_repl24)) *
                        (_repl20) +
                    (_repl12) *
                        (1.454545454545455 + (0.0606060606060606) * (_repl14) + (1.171717171717172) * (_repl24)) *
                        (_repl19) +
                    (3.588599675556062e-17 + (0.3636363636363636) * (_repl14)) * (_repl26) +
                    (_repl11) * (0.7727272727272733 + (0.2424242424242424) * (_repl14)) * ((_repl22) * (cosl1p2)) +
                    (_repl16) *
                        (1.646464646464646 + (-4.08080808080808) * (_repl14) + (-0.0808080808080808) * (_repl24)) *
                        ((_repl17) * (cosl1p2)) +
                    (1.) * ((_repl29) * (cosl1p2)) +
                    (_repl25) * (2.333333333333333 + (-2.333333333333333) * (_repl14)) * (l1) +
                    (_repl21) * (0.6666666666666666 + (-1.696969696969697) * (_repl14)) * ((cosl1p2) * (p))) *
                       (cosl1p1) +
                   (_repl12) *
                       ((-1.050505050505051) * (_repl11) + (-2.262626262626263) * (_repl12) +
                        (0.4848484848484849) * ((cosl1p2) * ((l1) * (p)))) *
                       ((_repl18) * (powr<5>(cosl1p1))) +
                   ((1.646464646464646) * ((_repl16) * (_repl17)) + (0.7727272727272733) * ((_repl11) * (_repl22)) +
                    (-0.3636363636363636) * ((_repl16) * ((_repl17) * (_repl24))) +
                    ((2.505050505050505) * ((_repl12) * (_repl19)) + (1.535353535353535) * ((_repl18) * (_repl20))) *
                        (_repl27) +
                    (1.) * (_repl29) +
                    ((-5.878787878787879) * ((_repl12) * (_repl19)) + (-14.54545454545455) * ((_repl18) * (_repl20)) +
                     (-0.3636363636363636) * (_repl26) + (-7.) * ((_repl25) * (l1))) *
                        (cosl1p2) +
                    (0.6666666666666665) * ((_repl21) * (p)) +
                    ((2.202020202020202) * ((_repl16) * (_repl17)) + (6.575757575757576) * ((_repl11) * (_repl22)) +
                     (-0.7272727272727272) * ((_repl21) * (p))) *
                        (_repl14)) *
                       (_repl23)) *
                  ((_repl30) *
                   ((_repl8) *
                    ((powr<-1>(p)) *
                     ((powr<-1>(_repl11 + _repl12 + (-2.) * ((cosl1p1) * ((l1) * (p))))) *
                      ((powr<-1>(_repl11 + _repl12 + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                                 (-2.) * ((cosl1p2) * ((l1) * (p))))) *
                       ((powr<-1>((_repl1) * (_repl5) + (_repl11 + _repl12 + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                                                         (-2.) * ((cosl1p2) * ((l1) * (p)))) *
                                                            (_repl9))) *
                        ((powr<-1>((_repl1) * (RB(_repl15, _repl11 + _repl12 + (-2.) * ((cosl1p1) * ((l1) * (p))))) +
                                   (_repl11 + _repl12 + (-2.) * ((cosl1p1) * ((l1) * (p)))) *
                                       (ZA(sqrt(_repl11 + _repl12 + (-2.) * ((cosl1p1) * ((l1) * (p)))))))) *
                         ((ZA3((0.816496580927726) * (sqrt(_repl11 + _repl12 + (-1.) * ((cosl1p1) * ((l1) * (p))))))) *
                          (ZA3(sqrt((0.6666666666666666) * (_repl11) + _repl12 +
                                    (-0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)))))))))))))) +
             (-0.1363636363636364) *
                 ((_repl1) * (_repl4) +
                  (_repl10 + (50.) * ((-1.) * (_repl1) + _repl7) * ((_repl13) * (powr<-1>(1. + _repl13)))) * (_repl2)) *
                 ((_repl30) *
                  ((-66. + (66.) * (_repl14) + (66.) * (_repl23) + (132.) * ((cosl1p1) * (cosl1p2))) * (_repl12) +
                   (-108. + (106.) * (_repl14) + (106.) * (_repl23) + (220.) * ((cosl1p1) * (cosl1p2))) * (_repl11) +
                   (l1) *
                       ((-106.) * (_repl27) + (-106.) * (_repl28) + (108. + (-326.) * (_repl14)) * (cosl1p1) +
                        (108.) * (cosl1p2) + (-326.) * ((_repl23) * (cosl1p2))) *
                       (p)) *
                  ((_repl8) *
                   ((powr<-1>((1.) * (_repl11) + (1.) * (_repl12) + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                              (-2.) * ((cosl1p2) * ((l1) * (p))))) *
                    ((powr<-1>((_repl1) * (_repl5) +
                               (_repl11 + _repl12 + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p))) * (_repl9))) *
                     (ZA4((0.5) * (sqrt((2.) * (_repl11) + (3.) * (_repl12) +
                                        (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)))))))))) +
             (-0.3636363636363637) *
                 ((-1.) * ((_repl27) * (l1)) + (1.) * ((_repl28) * (l1)) + (-1.5) * (p) + (0.5) * ((_repl14) * (p)) +
                  (cosl1p1) * ((-1.5) * ((cosl1p2) * (l1)) + (2.75) * (p)) * (cosl1p2) +
                  ((1.5) * ((cosl1p2) * (l1)) + (2.75) * (p)) * (_repl23)) *
                 ((_repl11) *
                  ((_repl3) * (_repl4) + (_repl2) * (dtZc(k)) +
                   ((-50.) * (_repl3) + (50.) * (Zc((1.02) * (k)))) * (_repl2)) *
                  ((powr<-1>(p)) *
                   ((ZAcbc((0.816496580927726) * (sqrt(_repl11 + _repl12 + (cosl1p1) * ((l1) * (p)))))) *
                    ((ZAcbc((0.816496580927726) * (sqrt(_repl11 + _repl12 + (l1) * (cosl1p1 + cosl1p2) * (p))))) *
                     ((ZAcbc(sqrt((0.6666666666666666) * (_repl11) + _repl12 +
                                  (0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p))))) *
                      ((powr<-2>((_repl2) * (_repl3) + (_repl11) * (Zc(l1)))) *
                       ((powr<-1>((_repl3) * (RB(_repl15, _repl11 + _repl12 + (2.) * ((cosl1p1) * ((l1) * (p))))) +
                                  (_repl11 + _repl12 + (2.) * ((cosl1p1) * ((l1) * (p)))) *
                                      (Zc(sqrt(_repl11 + _repl12 + (2.) * ((cosl1p1) * ((l1) * (p)))))))) *
                        (powr<-1>(
                            (_repl3) * (RB(_repl15, _repl11 + _repl12 + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)))) +
                            (_repl11 + _repl12 + (2.) * ((cosl1p1) * ((l1) * (p))) +
                             (2.) * ((cosl1p2) * ((l1) * (p)))) *
                                (Zc(sqrt(_repl11 + _repl12 + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p))))))))))))));
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto
    constant(const double &p, const double &k,
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