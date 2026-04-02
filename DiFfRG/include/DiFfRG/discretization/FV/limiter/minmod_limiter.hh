#pragma once

// DiFfRG
#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>

// standard library
#include <algorithm>
#include <cmath>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief MinMod slope limiter.
     *
     * Implements the classic minmod TVD limiter:
     *   slope_limit(a, b) = 0.5 * (sgn(a) + sgn(b)) * min(|a|, |b|)
     *
     * When the two one-sided slopes have the same sign, the limiter returns the
     * smaller one (in absolute value). When they have opposite signs (indicating
     * a local extremum), it returns zero.
     *
     * Satisfies the @c HasSlopeLimiter concept so it can be passed as the
     * @c Limiter template parameter of the TVDReconstructor:
     * @code
     * using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model,
     *                       def::TVDReconstructor<def::MinModLimiter, double>>;
     * @endcode
     */
    class MinModLimiter
    {
    public:
      template <typename NumberType> static NumberType slope_limit(const NumberType &du_1, const NumberType &du_2)
      {
                // Use the algebraic identity min(a,b) = 0.5*(a+b-|a-b|) instead of std::min so that
        // forward-mode AD (autodiff::Real) gets symmetric derivatives at ties (|du_1|==|du_2|).
        // autodiff::abs returns zero derivatives when the value is exactly zero, which makes
        // the result consistent with central-difference finite differences at the kink.
        const auto a1 = abs(du_1), a2 = abs(du_2);
        const auto min_a = NumberType(0.5) * (a1 + a2 - abs(a1 - a2));
        return NumberType(0.5) * (limiter_utils::sgn(du_1) + limiter_utils::sgn(du_2)) * min_a;
      }
    };
    static_assert(HasSlopeLimiter<MinModLimiter>);

  } // namespace def
} // namespace DiFfRG