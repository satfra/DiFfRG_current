#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZAcbc_kernel
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
      const auto _repl2 = Zc(k);
      const auto _repl3 = RB(powr<2>(k), powr<2>(l1));
      const auto _repl4 = RB(powr<2>(k), powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p));
      const auto _repl5 = ZA(l1);
      const auto _repl6 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _repl7 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl8 = Zc((1.02) * (k));
      const auto _repl9 = dtZc(k);
      const auto _repl10 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _repl11 = Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
      const auto _repl12 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl13 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl14 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl15 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p));
      const auto _repl16 = ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (cosl1p1) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl17 = ZA(sqrt(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl18 = RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl19 = Zc(sqrt(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
      const auto _repl20 = RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p));
      const auto _repl21 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))));
      const auto _repl22 = ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (cosl1p2) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl23 = RBdot(powr<2>(k), powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p));
      const auto _repl24 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + cosl1p2) * (p) + powr<2>(p))));
      const auto _repl25 = RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p));
      const auto _repl26 = powr<2>(l1);
      const auto _repl27 = powr<2>(p);
      const auto _repl28 = powr<2>(k);
      const auto _repl29 = powr<6>(k);
      const auto _repl30 = powr<2>(cosl1p2);
      const auto _repl31 = powr<3>(l1);
      const auto _repl32 = powr<3>(p);
      const auto _repl33 = powr<4>(l1);
      const auto _repl34 = powr<5>(l1);
      const auto _repl35 = powr<3>(cosl1p2);
      const auto _repl36 = powr<4>(p);
      const auto _repl37 = powr<2>(cosl1p1);
      const auto _repl38 = powr<5>(p);
      const auto _repl39 = powr<-2>((_repl1) * (_repl3) + (_repl26) * (_repl5));
      const auto _repl40 = powr<3>(cosl1p1);
      const auto _repl41 = powr<-2>(
          (_repl2) * (_repl4) + (_repl26 + _repl27 + (l1) * ((2.) * (cosl1p1) + (2.) * (cosl1p2)) * (p)) * (_repl19));
      const auto _repl42 = powr<4>(cosl1p1);
      return (-3.) *
                 ((_repl29) * ((-50.) * (_repl1) + (50.) * (_repl10)) * (_repl3) +
                  (_repl1) * (1. + _repl29) * (_repl6) + (_repl3) * (1. + (1.) * (_repl29)) * (_repl7)) *
                 ((_repl16) *
                  ((1.5) * ((_repl26) * (_repl32)) +
                   ((-1.333333333333333) * ((_repl27) * (_repl31)) + (-1.333333333333333) * (_repl34)) * (_repl35) +
                   (0.5) * (_repl38) + (-1.333333333333333) * ((_repl27) * ((_repl31) * ((_repl42) * (cosl1p2)))) +
                   ((0.3333333333333333) * ((_repl27) * (_repl31)) + (0.6666666666666666) * (_repl34) +
                    (-0.3333333333333333) * ((_repl36) * (l1))) *
                       (cosl1p2) +
                   (1.) * ((_repl33) * (p)) +
                   ((-0.6666666666666666) * ((_repl26) * (_repl32)) + (-0.6666666666666666) * ((_repl33) * (p))) *
                       (_repl30) +
                   ((_repl27) * (3.166666666666667 + (-4.) * (_repl30)) * (_repl31) +
                    (0.3333333333333333 + (-2.) * (_repl30)) * (_repl34) +
                    (_repl26) * (1.333333333333333 + (-1.333333333333333) * (_repl30)) * ((_repl32) * (cosl1p2)) +
                    (1.) * ((_repl38) * (cosl1p2)) +
                    (_repl36) * (1.833333333333333 + (-0.6666666666666666) * (_repl30)) * (l1) +
                    (_repl33) * (2.333333333333333 + (-4.) * (_repl30)) * ((cosl1p2) * (p))) *
                       (cosl1p1) +
                   (_repl26) *
                       ((-2.) * ((_repl26) * (cosl1p2)) + (2.666666666666667) * ((_repl27) * (cosl1p2)) +
                        (-0.6666666666666666) * ((l1) * (p)) + (-4.) * ((_repl30) * ((l1) * (p)))) *
                       ((_repl40) * (p)) +
                   (_repl37) *
                       ((-2.666666666666667) * ((_repl26) * ((_repl27) * (_repl35))) +
                        ((3.) * ((_repl26) * (_repl27)) + (-0.6666666666666666) * (_repl33) +
                         (3.666666666666667) * (_repl36)) *
                            (cosl1p2) +
                        (1.333333333333333) * ((_repl32) * (l1)) + (0.3333333333333333) * ((_repl31) * (p)) +
                        ((-3.333333333333333) * ((_repl32) * (l1)) + (-6.) * ((_repl31) * (p))) * (_repl30)) *
                       (l1)) *
                  ((_repl21) *
                   ((powr<-1>(1. + _repl29)) *
                    ((_repl39) *
                     ((p) *
                      ((powr<-2>(_repl26 + _repl27 + (2.) * ((cosl1p1) * ((l1) * (p))))) *
                       ((powr<-1>((_repl1) * (_repl18) +
                                  (_repl26 + _repl27 + (2.) * ((cosl1p1) * ((l1) * (p)))) * (_repl17))) *
                        ((powr<-1>((_repl15) * (_repl2) +
                                   (_repl26 + _repl27 + (-2.) * ((cosl1p2) * ((l1) * (p)))) * (_repl13))) *
                         (ZAcbc(sqrt((0.6666666666666666) * (_repl26) + _repl27 +
                                     (0.6666666666666666) * (cosl1p1 + (-1.) * (cosl1p2)) * ((l1) * (p))))))))))))) +
             (3.) *
                 ((_repl29) * ((-50.) * (_repl1) + (50.) * (_repl10)) * (_repl3) +
                  (_repl1) * (1. + _repl29) * (_repl6) + (_repl3) * (1. + (1.) * (_repl29)) * (_repl7)) *
                 ((_repl12) *
                  ((-0.5) * (_repl32) + (1.333333333333333) * ((_repl31) * (_repl35)) +
                   (-0.6666666666666666) * ((_repl31) * (cosl1p2)) +
                   (0.3333333333333333) * ((_repl27) * ((cosl1p2) * (l1))) + (-1.) * ((_repl26) * (p)) +
                   (0.6666666666666666) * ((_repl26) * ((_repl30) * (p))) +
                   (_repl40) * ((-0.6666666666666666) * ((cosl1p2) * (l1)) + (-2.333333333333333) * (p)) *
                       ((l1) * (p)) +
                   ((1.) * (_repl32) + (0.6666666666666666) * ((_repl31) * (cosl1p2)) +
                    (-3.) * ((_repl27) * ((cosl1p2) * (l1))) + (2.) * ((_repl26) * (p)) +
                    (-2.) * ((_repl26) * ((_repl30) * (p)))) *
                       (_repl37) +
                   ((-0.3333333333333333 + (2.) * (_repl30)) * (_repl31) + (1.) * ((_repl32) * (cosl1p2)) +
                    (_repl27) * (1.166666666666667 + (-0.6666666666666666) * (_repl30)) * (l1) +
                    (_repl26) * (2.333333333333333 + (-1.333333333333333) * (_repl30)) * ((cosl1p2) * (p))) *
                       (cosl1p1)) *
                  ((powr<-1>(1. + _repl29)) *
                   ((_repl39) *
                    ((p) *
                     ((powr<-1>((1.) * (_repl26) + (1.) * (_repl27) + (-2.) * ((cosl1p1) * ((l1) * (p))))) *
                      ((powr<-1>((_repl2) * (_repl20) + (_repl26 + _repl27 + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                                                         (-2.) * ((cosl1p2) * ((l1) * (p)))) *
                                                            (_repl11))) *
                       ((powr<-1>((_repl1) * (_repl14) +
                                  (_repl26 + _repl27 + (-2.) * ((cosl1p1) * ((l1) * (p)))) *
                                      (ZA(sqrt(_repl26 + _repl27 + (-2.) * ((cosl1p1) * ((l1) * (p)))))))) *
                        ((ZA3((0.816496580927726) * (sqrt(_repl26 + _repl27 + (-1.) * ((cosl1p1) * ((l1) * (p))))))) *
                         (ZAcbc(sqrt((0.6666666666666666) * (_repl26) + _repl27 +
                                     (-0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p))))))))))))) +
             (3.) * ((_repl2) * (_repl23) + ((-50.) * (_repl2) + (50.) * (_repl8)) * (_repl4) + (_repl4) * (_repl9)) *
                 ((_repl16) *
                  ((-1.5) * ((_repl26) * (_repl32)) +
                   ((-1.333333333333333) * ((_repl27) * (_repl31)) + (-1.333333333333333) * (_repl34)) * (_repl35) +
                   (-0.5) * (_repl38) +
                   ((0.3333333333333333) * ((_repl27) * (_repl31)) + (0.6666666666666666) * (_repl34) +
                    (-0.3333333333333333) * ((_repl36) * (l1))) *
                       (cosl1p2) +
                   (-1.) * ((_repl33) * (p)) +
                   (_repl26) * ((-1.333333333333333) * ((cosl1p2) * (l1)) + (4.666666666666667) * (p)) *
                       ((_repl27) * (_repl42)) +
                   (_repl40) *
                       ((4.333333333333333) * (_repl32) + (-2.) * ((_repl31) * (cosl1p2)) +
                        (5.333333333333333) * ((_repl27) * ((cosl1p2) * (l1))) +
                        (6.333333333333333) * ((_repl26) * (p)) + (-4.) * ((_repl26) * ((_repl30) * (p)))) *
                       ((l1) * (p)) +
                   ((0.6666666666666666) * ((_repl26) * (_repl32)) + (0.6666666666666666) * ((_repl33) * (p))) *
                       (_repl30) +
                   ((-2.833333333333334) * ((_repl27) * (_repl31)) +
                    (0.3333333333333333 + (-2.) * (_repl30)) * (_repl34) +
                    (_repl26) * (2.666666666666667 + (-1.333333333333333) * (_repl30)) * ((_repl32) * (cosl1p2)) +
                    (1.) * ((_repl38) * (cosl1p2)) +
                    (_repl36) * (-2.166666666666667 + (0.6666666666666666) * (_repl30)) * (l1) +
                    (_repl33) * (3.666666666666667 + (-4.) * (_repl30)) * ((cosl1p2) * (p))) *
                       (cosl1p1) +
                   ((0.6666666666666666) * ((_repl26) * (_repl32)) +
                    (-2.666666666666667) * ((_repl27) * ((_repl31) * (_repl35))) + (1.) * (_repl38) +
                    ((7.) * ((_repl27) * (_repl31)) + (-0.6666666666666666) * (_repl34) + (5.) * ((_repl36) * (l1))) *
                        (cosl1p2) +
                    (2.666666666666667) * ((_repl33) * (p)) +
                    ((-0.6666666666666666) * ((_repl26) * (_repl32)) + (-6.) * ((_repl33) * (p))) * (_repl30)) *
                       (_repl37)) *
                  ((_repl24) *
                   ((_repl41) *
                    ((powr<-1>((_repl1) * (_repl3) + (_repl26) * (_repl5))) *
                     ((p) * ((powr<-2>((1.) * (_repl26) + (1.) * (_repl27) + (2.) * ((cosl1p1) * ((l1) * (p))))) *
                             ((powr<-1>((_repl1) * (_repl18) +
                                        (_repl26 + _repl27 + (2.) * ((cosl1p1) * ((l1) * (p)))) * (_repl17))) *
                              (ZAcbc(sqrt((0.6666666666666666) * (_repl26) + _repl27 +
                                          (0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)))))))))))) +
             (-2.) *
                 ((_repl29) * ((-50.) * (_repl1) + (50.) * (_repl10)) * (_repl3) +
                  (_repl1) * (1. + _repl29) * (_repl6) + (_repl3) * (1. + (1.) * (_repl29)) * (_repl7)) *
                 ((_repl12) *
                  ((1.) * ((_repl35) * (l1)) + (cosl1p1) * (-0.25 + (1.5) * (_repl30)) * (l1) +
                   (-0.5) * ((cosl1p2) * (l1)) + (0.5) * ((_repl37) * ((cosl1p2) * (l1))) + (0.375) * (p) +
                   (-0.75) * ((_repl30) * (p)) + (-0.75) * ((cosl1p1) * ((cosl1p2) * (p)))) *
                  ((_repl21) *
                   ((powr<-1>(1. + _repl29)) *
                    ((_repl39) *
                     ((p) *
                      ((powr<-1>((_repl15) * (_repl2) +
                                 (_repl26 + _repl27 + (-2.) * ((cosl1p2) * ((l1) * (p)))) * (_repl13))) *
                       ((powr<-1>((_repl2) * (_repl20) + (_repl26 + _repl27 + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                                                          (-2.) * ((cosl1p2) * ((l1) * (p)))) *
                                                             (_repl11))) *
                        (ZAcbc(sqrt((0.6666666666666666) * (_repl26) + _repl27 +
                                    (-0.6666666666666666) * (cosl1p1 + (2.) * (cosl1p2)) * ((l1) * (p)))))))))))) +
             (-1.) * ((_repl2) * (_repl6) + ((-50.) * (_repl2) + (50.) * (_repl8)) * (_repl3) + (_repl3) * (_repl9)) *
                 ((_repl22) *
                  (((1.) * (_repl32) + (1.) * ((_repl31) * (cosl1p2)) + (3.) * ((_repl27) * ((cosl1p2) * (l1))) +
                    (1.) * ((_repl26) * (p)) + (2.) * ((_repl26) * ((_repl30) * (p)))) *
                       (_repl37) +
                   ((-1. + (2.) * (_repl30)) * (_repl31) + (1.) * ((_repl32) * (cosl1p2)) +
                    (_repl27) * (-1. + (4.) * (_repl30)) * (l1) +
                    (_repl26) * (-1. + (4.) * (_repl30)) * ((cosl1p2) * (p))) *
                       (cosl1p2) +
                   ((-0.5 + (3.) * (_repl30)) * (_repl31) + (2.5) * ((_repl32) * (cosl1p2)) +
                    (_repl27) * (-0.5 + (8.) * (_repl30)) * (l1) +
                    (_repl26) * (1.5 + (6.) * (_repl30)) * ((cosl1p2) * (p))) *
                       (cosl1p1)) *
                  ((_repl26) *
                   ((p) *
                    ((powr<-2>((1.) * (_repl26) + (1.) * (_repl27) + (2.) * ((cosl1p2) * ((l1) * (p))))) *
                     ((powr<-1>((_repl1) * (_repl25) +
                                (_repl26 + _repl27 + (2.) * ((cosl1p2) * ((l1) * (p)))) *
                                    (ZA(sqrt(_repl26 + _repl27 + (2.) * ((cosl1p2) * ((l1) * (p)))))))) *
                      ((ZAcbc((0.816496580927726) * (sqrt(_repl26 + _repl27 + (-1.) * ((cosl1p1) * ((l1) * (p))))))) *
                       ((ZAcbc(sqrt((0.6666666666666666) * (_repl26) + _repl27 +
                                    (0.6666666666666666) * ((-1.) * (cosl1p1) + cosl1p2) * ((l1) * (p))))) *
                        ((powr<-2>((_repl2) * (_repl3) + (_repl26) * (Zc(l1)))) *
                         (powr<-1>((_repl14) * (_repl2) +
                                   (_repl26 + _repl27 + (-2.) * ((cosl1p1) * ((l1) * (p)))) *
                                       (Zc(sqrt(_repl26 + _repl27 + (-2.) * ((cosl1p1) * ((l1) * (p)))))))))))))))) +
             (1.) * ((_repl2) * (_repl23) + ((-50.) * (_repl2) + (50.) * (_repl8)) * (_repl4) + (_repl4) * (_repl9)) *
                 ((_repl22) *
                  ((2.) * ((_repl35) * (l1)) + (cosl1p1) * (-0.5 + (3.) * (_repl30)) * (l1) +
                   (-1.) * ((cosl1p2) * (l1)) + (1.) * ((_repl37) * ((cosl1p2) * (l1))) + (-0.75) * (p) +
                   (1.5) * ((_repl30) * (p)) + (1.5) * ((cosl1p1) * ((cosl1p2) * (p)))) *
                  ((_repl24) *
                   ((_repl41) *
                    ((powr<-1>((_repl1) * (_repl3) + (_repl26) * (_repl5))) *
                     ((p) * ((ZAcbc(sqrt((0.6666666666666666) * (_repl26) + _repl27 +
                                         (0.6666666666666666) * (cosl1p1 + (2.) * (cosl1p2)) * ((l1) * (p))))) *
                             (powr<-1>((_repl2) * (_repl25) +
                                       (_repl26 + _repl27 + (2.) * ((cosl1p2) * ((l1) * (p)))) *
                                           (Zc(sqrt(_repl26 + _repl27 + (2.) * ((cosl1p2) * ((l1) * (p))))))))))))));
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
using DiFfRG::ZAcbc_kernel;