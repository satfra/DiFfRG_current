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
      const auto _repl2 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _repl3 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl4 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl5 = ZA(sqrt(powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl6 = Zc(k);
      const auto _repl7 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl8 = RB(powr<2>(k), powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl9 = Zc(sqrt(powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl10 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cos1) * ((l1) * (p))) + powr<2>(p))));
      const auto _repl11 = ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (cos1) * ((l1) * (p)) + powr<2>(p))));
      return ((powr<-1>(1. + powr<6>(k))) *
                  ((_repl2) * (1. + powr<6>(k)) * (_repl3) +
                   (_repl1) * (1. + powr<6>(k)) * (dtZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                   (-50.) * (_repl3 + (-1.) * (ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667))))) *
                       ((_repl1) * (powr<6>(k)))) *
                  ((powr<-1>(powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p))) *
                   ((-4.) * (-1. + powr<2>(cos1)) *
                        (((3.) * (powr<4>(l1)) + (-6.) * ((cos1) * ((powr<3>(l1)) * (p))) +
                          (powr<2>(l1)) * (8. + powr<2>(cos1)) * (powr<2>(p)) +
                          (-6.) * ((cos1) * ((l1) * (powr<3>(p)))) + (3.) * (powr<4>(p))) *
                         (powr<2>(ZA3((0.816496580927726) *
                                      (sqrt(powr<2>(l1) + (-1.) * ((cos1) * ((l1) * (p))) + powr<2>(p))))))) +
                    (powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (-7. + powr<2>(cos1)) *
                        (((_repl3) * (_repl4) +
                          (powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl5)) *
                         (ZA4((0.7071067811865475) * (sqrt(powr<2>(l1) + powr<2>(p))))))) *
                   ((powr<-1>((_repl3) * (_repl4) +
                              (powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl5))) *
                    (powr<-2>((_repl1) * (_repl3) + (powr<2>(l1)) * (ZA(l1)))))) +
              (powr<2>(l1)) * (-1. + powr<2>(cos1)) *
                  ((powr<-1>((_repl4) * (_repl6) +
                             (powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl7))) *
                   ((powr<2>(_repl11)) * ((_repl4) * (_repl6)) + (powr<2>(_repl10)) * ((_repl6) * (_repl8)) +
                    (powr<2>(_repl11)) * ((_repl7) * (powr<2>(l1))) + (powr<2>(_repl10)) * ((_repl9) * (powr<2>(l1))) +
                    (-2.) * ((powr<2>(_repl11)) * ((_repl7) * ((cos1) * ((l1) * (p))))) +
                    (2.) * ((powr<2>(_repl10)) * ((_repl9) * ((cos1) * ((l1) * (p))))) +
                    (powr<2>(_repl11)) * ((_repl7) * (powr<2>(p))) + (powr<2>(_repl10)) * ((_repl9) * (powr<2>(p)))) *
                   ((powr<-1>((_repl6) * (_repl8) +
                              (powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl9))) *
                    ((_repl2) * (_repl6) + (_repl1) * (dtZc(k)) +
                     (-50.) * (_repl6 + (-1.) * (Zc((1.02) * (k)))) * (_repl1)) *
                    (powr<-2>((_repl1) * (_repl6) + (powr<2>(l1)) * (Zc(l1))))))) *
             (powr<-2>(p));
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
using DiFfRG::ZA_kernel;