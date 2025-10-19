#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class Zc_kernel
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
      const auto _repl1 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl2 = Zc(k);
      const auto _repl3 = RB(powr<2>(k), powr<2>(l1));
      const auto _repl4 = RB(powr<2>(k), powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl5 = ZA(l1);
      const auto _repl6 = powr<2>(l1);
      const auto _repl7 = powr<2>(p);
      const auto _repl8 = powr<2>(k);
      const auto _repl9 = powr<6>(k);
      const auto _repl10 = powr<2>(cos1);
      return (-3.) * (-1. + _repl10) *
                 ((powr<-2>((_repl1) * (_repl3) + (_repl5) * (_repl6))) *
                  ((_repl1) * (RBdot(_repl8, _repl6)) +
                   (dtZA(pow(1. + _repl9, 0.16666666666666666667)) +
                    (50.) * ((-1.) * (_repl1) + ZA((1.02) * (pow(1. + _repl9, 0.16666666666666666667)))) *
                        ((_repl9) * (powr<-1>(1. + _repl9)))) *
                       (_repl3)) *
                  ((powr<2>(ZAcbc((0.816496580927726) * (sqrt(_repl6 + _repl7 + (-1.) * ((cos1) * ((l1) * (p)))))))) *
                   (powr<-1>((_repl2) * (RB(_repl8, _repl6 + _repl7 + (-2.) * ((cos1) * ((l1) * (p))))) +
                             (_repl6 + _repl7 + (-2.) * ((cos1) * ((l1) * (p)))) *
                                 (Zc(sqrt(_repl6 + _repl7 + (-2.) * ((cos1) * ((l1) * (p)))))))))) +
             (-3.) * (-1. + _repl10) *
                 ((powr<-1>((_repl1) * (_repl3) + (_repl5) * (_repl6))) *
                  ((_repl2) * (RBdot(_repl8, _repl6 + _repl7 + (2.) * ((cos1) * ((l1) * (p))))) +
                   ((-50.) * (_repl2) + dtZc(k) + (50.) * (Zc((1.02) * (k)))) * (_repl4)) *
                  ((powr<2>(ZAcbc((0.816496580927726) * (sqrt(_repl6 + _repl7 + (cos1) * ((l1) * (p))))))) *
                   (powr<-2>((_repl2) * (_repl4) + (_repl6 + _repl7 + (2.) * ((cos1) * ((l1) * (p)))) *
                                                       (Zc(sqrt(_repl6 + _repl7 + (2.) * ((cos1) * ((l1) * (p))))))))));
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
using DiFfRG::Zc_kernel;