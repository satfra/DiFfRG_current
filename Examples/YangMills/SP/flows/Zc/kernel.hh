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
      const auto _repl2 = RB(powr<2>(k), powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl3 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _repl4 = ZA(sqrt(powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      const auto _repl5 = ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (cos1) * ((l1) * (p)) + powr<2>(p))));
      const auto _repl6 = Zc(k);
      const auto _repl7 = Zc(l1);
      const auto _repl8 = ZA(l1);
      const auto _repl9 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cos1) * ((l1) * (p))) + powr<2>(p))));
      const auto _repl10 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _repl11 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      return (3.) * (-1. + powr<2>(cos1)) *
             ((-1.) *
                  ((_repl1) * ((powr<2>(_repl5)) *
                               ((powr<2>(l1)) *
                                ((powr<-2>((_repl1) * (_repl6) + (_repl7) * (powr<2>(l1)))) *
                                 ((powr<-1>(powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p))) *
                                  ((powr<-1>((_repl2) * (_repl3) +
                                             (powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl4))) *
                                   (dtZc(k)))))))) +
              ((-1.) * ((_repl3) *
                        ((powr<2>(_repl9)) *
                         ((powr<-2>((_repl1) * (_repl3) + (_repl8) * (powr<2>(l1)))) *
                          (powr<-1>((_repl10) * (_repl6) +
                                    (powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl11)))))) +
               (-1.) * ((powr<2>(_repl5)) *
                        ((_repl6) *
                         ((powr<2>(l1)) *
                          ((powr<-2>((_repl1) * (_repl6) + (_repl7) * (powr<2>(l1)))) *
                           ((powr<-1>(powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p))) *
                            (powr<-1>((_repl2) * (_repl3) +
                                      (powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl4))))))))) *
                  (RBdot(powr<2>(k), powr<2>(l1))) +
              ((-1.) *
                   ((1. + powr<6>(k)) * (dtZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                    (50.) * ((-1.) * (_repl3) + ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)))) *
                        (powr<6>(k))) *
                   ((powr<2>(_repl9)) *
                    ((powr<-1>(1. + powr<6>(k))) *
                     ((powr<-2>((_repl1) * (_repl3) + (_repl8) * (powr<2>(l1)))) *
                      (powr<-1>((_repl10) * (_repl6) +
                                (powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl11)))))) +
               (50.) * (_repl6 + (-1.) * (Zc((1.02) * (k)))) *
                   ((powr<2>(_repl5)) *
                    ((powr<2>(l1)) *
                     ((powr<-2>((_repl1) * (_repl6) + (_repl7) * (powr<2>(l1)))) *
                      ((powr<-1>(powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p))) *
                       (powr<-1>((_repl2) * (_repl3) +
                                 (powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)) * (_repl4)))))))) *
                  (_repl1));
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
using DiFfRG::Zc_kernel;