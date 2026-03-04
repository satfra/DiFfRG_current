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
     * @c Limiter template parameter of the KT assembler:
     * @code
     * using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, def::MinModLimiter>;
     * @endcode
     */
    class MinModLimiter
    {
    public:
      template <typename NumberType> static NumberType slope_limit(const NumberType &du_1, const NumberType &du_2)
      {
        using std::abs;
        using std::min;
        return 0.5 * (limiter_utils::sgn(du_1) + limiter_utils::sgn(du_2)) * min(abs(du_1), abs(du_2));
      }
    };
    static_assert(HasSlopeLimiter<MinModLimiter>);

  } // namespace def
} // namespace DiFfRG