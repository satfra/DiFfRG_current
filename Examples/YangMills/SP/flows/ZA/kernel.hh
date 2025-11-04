#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const double &l1, const double &cos1, const double &p, const double &k,
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
      const auto _repl1 = RB(powr<2>(k), powr<2>(l1));
      const auto _repl2 = Zc(k);
      const auto _repl3 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl4 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _repl5 = dtZc(k);
      const auto _repl6 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl7 = ZA(l1);
      const auto _repl8 = Zc(l1);
      const auto _repl9 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _repl10 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl11 = Zc((1.02) * (k));
      const auto _repl12 = powr<2>(l1);
      const auto _repl13 = powr<2>(k);
      const auto _repl14 = powr<6>(k);
      const auto _repl15 = powr<2>(p);
      const auto _repl16 = powr<2>(cos1);
      const auto _repl17 = powr<-2>(p);
      const auto _repl18 = powr<-2>((_repl1) * (_repl2) + (_repl12) * (_repl8));
      const auto _repl19 = powr<-2>((_repl1) * (_repl3) + (_repl12) * (_repl7));
      return (-12.) *
                 ((_repl1) * (1. + (1.) * (_repl14)) * (_repl10) + (_repl3) * (1. + _repl14) * (_repl4) +
                  (_repl1) * ((-50.) * (_repl3) + (50.) * (_repl9)) * (_repl14)) *
                 ((powr<-1>(1. + _repl14)) *
                  ((_repl12) *
                       (-2.666666666666667 + (2.333333333333333) * (_repl16) + (0.3333333333333333) * (powr<4>(cos1))) *
                       (_repl15) +
                   (-1. + (1.) * (_repl16)) * (powr<4>(l1)) +
                   (cos1) * (2. + (-2.) * (_repl16)) * ((powr<3>(l1)) * (p)) +
                   (cos1) * (2. + (-2.) * (_repl16)) * ((l1) * (powr<3>(p))) +
                   (-1. + (1.) * (_repl16)) * (powr<4>(p))) *
                  ((_repl17) *
                   ((_repl19) * ((powr<-1>(_repl12 + _repl15 + (-2.) * ((cos1) * ((l1) * (p))))) *
                                 ((powr<-1>((_repl3) * (_repl6) +
                                            (_repl12 + _repl15 + (-2.) * ((cos1) * ((l1) * (p)))) *
                                                (ZA(sqrt(_repl12 + _repl15 + (-2.) * ((cos1) * ((l1) * (p)))))))) *
                                  (powr<2>(ZA3((0.816496580927726) *
                                               (sqrt(_repl12 + _repl15 + (-1.) * ((cos1) * ((l1) * (p))))))))))))) +
             (-1.) * (7. + (-1.) * (_repl16)) *
                 ((_repl17) *
                  ((_repl3) * (_repl4) +
                   (_repl10 + (50.) * ((-1.) * (_repl3) + _repl9) * ((_repl14) * (powr<-1>(1. + _repl14)))) *
                       (_repl1)) *
                  ((_repl19) * (ZA4((0.7071067811865475) * (sqrt(_repl12 + _repl15)))))) +
             (1.) * (-1. + _repl16) *
                 ((_repl12) *
                  (((50.) * (_repl11) + (-50.) * (_repl2)) * (_repl1) + (_repl2) * (_repl4) + (_repl1) * (_repl5)) *
                  ((_repl17) *
                   ((_repl18) * ((powr<2>(ZAcbc((0.816496580927726) *
                                                (sqrt(_repl12 + _repl15 + (-1.) * ((cos1) * ((l1) * (p)))))))) *
                                 (powr<-1>((_repl2) * (_repl6) +
                                           (_repl12 + _repl15 + (-2.) * ((cos1) * ((l1) * (p)))) *
                                               (Zc(sqrt(_repl12 + _repl15 + (-2.) * ((cos1) * ((l1) * (p)))))))))))) +
             (1.) * (-1. + _repl16) *
                 ((_repl12) *
                  (((50.) * (_repl11) + (-50.) * (_repl2)) * (_repl1) + (_repl2) * (_repl4) + (_repl1) * (_repl5)) *
                  ((_repl17) *
                   ((_repl18) *
                    ((powr<2>(ZAcbc((0.816496580927726) * (sqrt(_repl12 + _repl15 + (cos1) * ((l1) * (p))))))) *
                     (powr<-1>((_repl2) * (RB(_repl13, _repl12 + _repl15 + (2.) * ((cos1) * ((l1) * (p))))) +
                               (_repl12 + _repl15 + (2.) * ((cos1) * ((l1) * (p)))) *
                                   (Zc(sqrt(_repl12 + _repl15 + (2.) * ((cos1) * ((l1) * (p))))))))))));
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
using DiFfRG::ZA_kernel;