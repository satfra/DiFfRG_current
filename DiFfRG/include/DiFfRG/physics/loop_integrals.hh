#pragma once

// standard library
#include <cmath>

// external libraries
#include <deal.II/base/quadrature_lib.h>
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  using namespace dealii;

  namespace LoopIntegrals
  {
    /**
     * @brief Performs the integral
     * \f[
     *    \int d\Omega_{d} \int_0^\infty dq f(q^2) q^{d-1}
     * \f]
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT integrate(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent, const double k)
    {
      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();

      static_assert(d > 0, "Dimension must be greater than zero");
      // Dimension of the spatial integral
      const double S_d = 2. * std::pow(M_PI, d / 2.) / std::tgammal(d / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_d                    // angular integral
                               * powr<-d>(2. * M_PI); // fourier factors

      const NT result = tbb::parallel_reduce(
          tbb::blocked_range<int>(0, x_size), NT(0),
          [&](const tbb::blocked_range<int> &r, NT running_total) -> NT {
            const double q_max = std::sqrt(x_extent) * k;
            for (int x_it = r.begin(); x_it < r.end(); x_it++) {
              const double q = x_q_p[x_it][0] * q_max;
              const double q_weight = x_q_w[x_it] * q_max;
              const double q2 = powr<2>(q);

              running_total += q_weight * prefactor * std::pow(q, d - 1) // integral over q in d dimensions
                               * fun(q2);                                // integrand
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result;
    }
  } // namespace LoopIntegrals
} // namespace DiFfRG