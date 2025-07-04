#pragma once

#include "DiFfRG/physics/utils.hh"

namespace DiFfRG
{
  template <typename _Regulator> class V_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const auto &l1, const double &k, const double &N, const double &T,
                                                   const auto &m2Pi, const auto &m2Sigma)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const auto _repl1 = RB(powr<2>(k), powr<2>(l1));
      const auto _repl2 = RBdot(powr<2>(k), powr<2>(l1));
      return 0.25 * (-1. + N) * _repl2 * sqrt(powr<-1>(_repl1 + powr<2>(l1) + m2Pi)) *
                 CothFiniteT(sqrt(_repl1 + powr<2>(l1) + m2Pi), T) +
             0.25 * _repl2 * sqrt(powr<-1>(_repl1 + powr<2>(l1) + m2Sigma)) *
                 CothFiniteT(sqrt(_repl1 + powr<2>(l1) + m2Sigma), T);
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double &k, const double &N, const double &T,
                                                     const auto &m2Pi, const auto &m2Sigma)
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
using DiFfRG::V_kernel;