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
            for (int x_it = r.begin(); x_it < r.end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              running_total +=
                  x_weight * prefactor *
                  (0.5 * powr<d>(k) * std::pow(x, (d - 2) / 2.)) // integral over x = p^2 / k^2 in d dimensions
                  * fun(q2);                                     // integrand
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result;
    }

    /**
     * @brief Performs the integral
     * \f[
     *    \int d\Omega_{d-1} \int_{-1}^1 d\cos \int_0^\infty dq f(q^2, cos) q^{d-1}
     * \f]
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @param cos_quadrature Quadrature rule for the integral over \f$\cos\f$.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT angle_integrate(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent, const double k,
                       const QGauss<1> &cos_quadrature)
    {
      static_assert(d > 1, "Dimension must be greater than one, otherwise the angular integral not defined.");

      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();
      const auto &cos_q_p = cos_quadrature.get_points();
      const auto &cos_q_w = cos_quadrature.get_weights();
      const int cos_size = cos_quadrature.size();

      static_assert(d > 1, "Dimension must be greater than one");
      // Area of unit sphere in d dimensions
      const double S_d = 2. * std::pow(M_PI, d / 2.) / std::tgamma(d / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_d                    // angular integral
                               * powr<-d>(2. * M_PI); // fourier factors

      const NT result = tbb::parallel_reduce(
          tbb::blocked_range2d<int>(0, x_size, 0, cos_size), NT(0),
          [&](const tbb::blocked_range2d<int> &r, NT running_total) -> NT {
            for (int x_it = r.rows().begin(); x_it < r.rows().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              for (int cos_it = r.cols().begin(); cos_it < r.cols().end(); cos_it++) {
                const double cos = (cos_q_p[cos_it][0] - 0.5) * 2.;
                const double cos_weight = cos_q_w[cos_it] * 2.;

                running_total +=
                    x_weight * prefactor *
                    (0.5 * powr<d>(k) * std::pow(x, (d - 2) / 2.)) // integral over x = p^2 / k^2 in d dimensions
                    * 0.5 * cos_weight // integral over cos(theta), 0.5 removes the factor from the angular integral
                    * fun(q2, cos);    // integrand
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result;
    }

    /**
     * @brief Performs the integral
     * \f[
     *    \int d\Omega_{d-2} \int_{-1}^1 d\cos \int_0^{2\pi}d\phi \int_0^\infty dq f(q^2, cos, phi) q^{d-1}
     * \f]
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @param cos_quadrature Quadrature rule for the integral over \f$\cos\f$.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT two_angle_integrate(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent, const double k,
                           const QGauss<1> &cos_quadrature)
    {
      static_assert(d > 2, "Dimension must be greater than 2, otherwise the angular integral not defined.");

      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();
      const auto &cos_q_p = cos_quadrature.get_points();
      const auto &cos_q_w = cos_quadrature.get_weights();
      const int cos_size = cos_quadrature.size();
      const auto &phi_q_p = cos_quadrature.get_points();
      const auto &phi_q_w = cos_quadrature.get_weights();
      const int phi_size = cos_quadrature.size();

      static_assert(d > 1, "Dimension must be greater than one");
      // Area of unit sphere in d dimensions
      const double S_d = 2. * std::pow(M_PI, d / 2.) / std::tgamma(d / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_d                    // angular integral
                               * powr<-d>(2. * M_PI); // fourier factors

      const NT result = tbb::parallel_reduce(
          tbb::blocked_range3d<int>(0, x_size, 0, cos_size, 0, phi_size), NT(0),
          [&](const tbb::blocked_range3d<int> &r, NT running_total) -> NT {
            for (int x_it = r.pages().begin(); x_it < r.pages().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              for (int cos_it = r.rows().begin(); cos_it < r.rows().end(); cos_it++) {
                const double cos = (cos_q_p[cos_it][0] - 0.5) * 2.;
                const double cos_weight = cos_q_w[cos_it] * 2.;

                for (int phi_it = r.cols().begin(); phi_it < r.cols().end(); phi_it++) {
                  const double phi = phi_q_p[phi_it][0] * 2. * M_PI;
                  const double phi_weight = cos_q_w[cos_it] * 2. * M_PI;

                  running_total +=
                      x_weight * prefactor *
                      (0.5 * powr<d>(k) * std::pow(x, (d - 2) / 2.)) // integral over x = p^2 / k^2 in d dimensions
                      * 0.5 * cos_weight // integral over cos(theta), 0.5 removes the factor from the angular integral
                      * phi_weight / (2. * M_PI) // integral over phi, 2pi removes the factor from the angular integral
                      * fun(q2, cos, phi);       // integrand
                }
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result;
    }

    /**
     * @brief Performs the integral
     * \f[
     *    \int d\Omega_{d-2} \int_{-1}^1 d\cos \int_0^{2\pi}d\phi \int_0^\infty dq f(q^2, cos, phi) q^{d-1}
     * \f]
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @param cos_quadrature Quadrature rule for the integral over \f$\cos\f$.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT three_angle_integrate(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent, const double k,
                             const QGauss<1> &cos_quadrature)
    {
      static_assert(d > 3, "Dimension must be greater than 3, otherwise the angular integral not defined.");

      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();
      const auto &cos_q_p = cos_quadrature.get_points();
      const auto &cos_q_w = cos_quadrature.get_weights();
      const int cos_size = cos_quadrature.size();
      const auto &phi_q_p = cos_quadrature.get_points();
      const auto &phi_q_w = cos_quadrature.get_weights();
      const int phi_size = cos_quadrature.size();

      static_assert(d > 1, "Dimension must be greater than one");
      // Area of unit sphere in d dimensions
      const double S_d = 2. * std::pow(M_PI, d / 2.) / std::tgamma(d / 2.);
      const double S_4 = 2. * std::pow(M_PI, 4. / 2.) / std::tgamma(4. / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_d                    // angular integral
                               * powr<-d>(2. * M_PI); // fourier factors

      const NT result = tbb::parallel_reduce(
          tbb::blocked_range3d<int>(0, x_size, 0, cos_size, 0, cos_size), NT(0),
          [&](const tbb::blocked_range3d<int> &r, NT running_total) -> NT {
            for (int x_it = r.pages().begin(); x_it < r.pages().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              for (int cos1_it = r.rows().begin(); cos1_it < r.rows().end(); cos1_it++) {
                const double cos1 = (cos_q_p[cos1_it][0] - 0.5) * 2.;
                const double cos1_weight =
                    cos_q_w[cos1_it] * 2. * std::sqrt(1. - powr<2>(cos1)); // integration jacobian

                for (int cos2_it = r.cols().begin(); cos2_it < r.cols().end(); cos2_it++) {
                  const double cos2 = (cos_q_p[cos2_it][0] - 0.5) * 2.;
                  const double cos2_weight = cos_q_w[cos2_it] * 2.;

                  for (int phi_it = 0; phi_it < phi_size; phi_it++) {
                    const double phi = phi_q_p[phi_it][0] * 2. * M_PI;
                    const double phi_weight = phi_q_w[phi_it] * 2. * M_PI;

                    running_total +=
                        x_weight * prefactor *
                        (0.5 * powr<d>(k) * std::pow(x, (d - 2) / 2.)) // integral over x = p^2 / k^2 in d dimensions
                        * cos1_weight                                  // integral over cos(theta1)
                        * cos2_weight                                  // integral over cos(theta2)
                        * phi_weight /
                        S_4 // integral over phi, S_3 = 2 pi^2 removes the factor from the angular integral
                        * fun(q2, cos1, cos2, phi); // integrand
                  }
                }
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result;
    }

    /**
     * @brief Performs the integral
     * \f[
     *    \int_{-\infty}^\infty dq_0\int d\Omega_{d-1} \int_0^\infty dq f(q^2, cos, q0) q^{d-2}
     * \f]
     * with \f$q_0\f$ being the zeroth momentum mode.
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @param m_quadrature Quadrature rule for the integral over \f$q0\f$, specifically for the domain [0,m_extent).
     * @param m_extent Extent of the integral over \f$q0\f$.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT spatial_integrate_and_integrate(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent,
                                       const double k, const QGauss<1> &m_quadrature, const double m_extent)
    {
      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();
      const auto &m_q_p = m_quadrature.get_points();
      const auto &m_q_w = m_quadrature.get_weights();
      const int m_size = m_quadrature.size();

      static_assert(d > 0, "Dimension must be greater than zero");
      // Dimension of the spatial integral
      constexpr int ds = d - 1;
      // Area of unit sphere in ds=d-1 dimensions
      const double S_ds = 2. * std::pow(M_PI, ds / 2.) / std::tgamma(ds / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_ds                    // angular integral
                               * powr<-ds>(2. * M_PI); // fourier factors

      const NT result = tbb::parallel_reduce(
          tbb::blocked_range2d<int>(0, x_size, 0, m_size), NT(0),
          [&](const tbb::blocked_range2d<int> &r, NT running_total) -> NT {
            for (int x_it = r.rows().begin(); x_it < r.rows().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              // Do a "matsubara" (rather p0)-integral
              for (int q0_it = r.cols().begin(); q0_it < r.cols().end(); q0_it++) {
                const double q0 = m_q_p[q0_it][0] * m_extent;
                const double m_weight = m_q_w[q0_it] * m_extent;

                running_total += (m_weight / (2. * M_PI)) // Matsubara integral weight
                                 * x_weight * prefactor *
                                 (0.5 * powr<ds>(k) * std::pow(x, (ds - 2) / 2.)) // integral over x = p^2 / k^2
                                 * (fun(q2, q0) + fun(q2, -q0));                  // integrand
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result;
    }

    /**
     * @brief Performs the integral
     * \f[
     *    \int_{-\infty}^\infty dq_0\int d\Omega_{d-2} \int_{-1}^1 d\cos \int_0^\infty dq f(q^2, cos, q0) q^{d-2}
     * \f]
     * with \f$q_0\f$ being the zeroth momentum mode.
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @param cos_quadrature Quadrature rule for the integral over \f$\cos\f$.
     * @param m_quadrature Quadrature rule for the integral over \f$q0\f$, specifically for the domain [0,m_extent).
     * @param m_extent Extent of the integral over \f$q0\f$.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT spatial_angle_integrate_and_integrate(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent,
                                             const double k, const QGauss<1> &cos_quadrature,
                                             const QGauss<1> &m_quadrature, const double m_extent)
    {
      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();
      const auto &cos_q_p = cos_quadrature.get_points();
      const auto &cos_q_w = cos_quadrature.get_weights();
      const int cos_size = cos_quadrature.size();
      const auto &m_q_p = m_quadrature.get_points();
      const auto &m_q_w = m_quadrature.get_weights();
      const int m_size = m_quadrature.size();

      static_assert(d > 0, "Dimension must be greater than zero");
      // Dimension of the spatial integral
      constexpr int ds = d - 1;
      // Area of unit sphere in ds=d-1 dimensions
      const double S_ds = 2. * std::pow(M_PI, ds / 2.) / std::tgamma(ds / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_ds                    // angular integral
                               * powr<-ds>(2. * M_PI); // fourier factors

      const NT result = tbb::parallel_reduce(
          tbb::blocked_range3d<int>(0, x_size, 0, cos_size, 0, m_size), NT(0),
          [&](const tbb::blocked_range3d<int> &r, NT running_total) -> NT {
            for (int x_it = r.pages().begin(); x_it < r.pages().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              for (int cos_it = r.rows().begin(); cos_it < r.rows().end(); cos_it++) {
                const double cos = (cos_q_p[cos_it][0] - 0.5) * 2.;
                const double cos_weight = cos_q_w[cos_it] * 2.;

                // Do a "matsubara" (rather q0)-integral
                for (int q0_it = r.cols().begin(); q0_it < r.cols().end(); q0_it++) {
                  const double q0 = m_q_p[q0_it][0] * m_extent;
                  const double m_weight = m_q_w[q0_it] * m_extent;

                  running_total +=
                      (m_weight / (2. * M_PI)) // Matsubara integral weight
                      * x_weight * prefactor *
                      (0.5 * powr<ds>(k) * std::pow(x, (ds - 2) / 2.)) // integral over x = p^2 / k^2
                      * 0.5 * cos_weight // integral over cos(theta), 0.5 removes the factor from the angular integral
                      * (fun(q2, cos, q0) + fun(q2, cos, -q0)); // integrand
                }
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result;
    }

    /**
     * @brief Performs the integral
     * \f[
     *    \sum_{q0}\int d\Omega_{d-1} \int_0^\infty dq f(q^2, q0) q^{d-2}
     * \f]
     * with \f$q0 \,\in 2\pi\mathbb{Z}\f$ being a Matsubara frequency. The Matsubara sum is performed only for the first
     * \f$|n| <= m_order\f$ summands, the rest is approximated by an integral.
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @param m_order Number of Matsubara frequencies to be summed over.
     * @param m_quadrature Quadrature rule for the integral over \f$q0\f$.
     * @param m_extent Extent of the integral over \f$q0\f$.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT spatial_integrate_and_sum(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent, const double k,
                                 const int m_order, const QGauss<1> &m_quadrature, const double m_extent,
                                 const double T)
    {
      if (is_close(T, 0.))
        return spatial_integrate_and_integrate<NT, d, FUN>(fun, x_quadrature, x_extent, k, m_quadrature, m_extent);

      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();
      const auto &m_q_p = m_quadrature.get_points();
      const auto &m_q_w = m_quadrature.get_weights();
      const int m_size = m_quadrature.size();

      static_assert(d > 0, "Dimension must be greater than zero");
      // Dimension of the spatial integral
      constexpr int ds = d - 1;
      // Area of unit sphere in ds=d-1 dimensions
      const double S_ds = 2. * std::pow(M_PI, ds / 2.) / std::tgamma(ds / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_ds                    // angular integral
                               * powr<-ds>(2. * M_PI); // fourier factors

      // Do a matsubara sum for the first m_order summands
      const NT result_sum = tbb::parallel_reduce(
          tbb::blocked_range2d<int>(0, x_size, -m_order, m_order + 1), NT(0),
          [&](const tbb::blocked_range2d<int> &r, NT running_total) -> NT {
            for (int x_it = r.rows().begin(); x_it < r.rows().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              for (int q0_it = r.cols().begin(); q0_it < r.cols().end(); q0_it++) {
                const double q0 = 2. * M_PI * T * q0_it;

                running_total += T // Matsubara sum weight
                                 * x_weight * prefactor *
                                 (0.5 * powr<ds>(k) * std::pow(x, (ds - 2) / 2.)) // integral over x = p^2 / k^2
                                 * fun(q2, q0);                                   // integrand
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });

      // Approximate the rest of the sum by an integral
      const NT result_integral = tbb::parallel_reduce(
          tbb::blocked_range2d<int>(0, x_size, 0, m_size), NT(0),
          [&](const tbb::blocked_range2d<int> &r, NT running_total) -> NT {
            for (int x_it = r.rows().begin(); x_it < r.rows().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              for (int q0_it = r.cols().begin(); q0_it < r.cols().end(); q0_it++) {
                const double q0 =
                    (2. * m_order * M_PI * T) + m_q_p[q0_it][0] * (m_extent - 2. * (m_order + 1) * M_PI * T);
                const double m_weight = m_q_w[q0_it] * (m_extent - 2. * m_order * M_PI * T);

                running_total += (1. * m_weight / (2. * M_PI)) // Matsubara integral weight
                                 * x_weight * prefactor *
                                 (0.5 * powr<ds>(k) * std::pow(x, (ds - 2) / 2.)) // integral over x = p^2 / k^2
                                 * (fun(q2, q0) + fun(q2, -q0));                  // integrand
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result_sum + result_integral;
    } // namespace LoopIntegrals

    /**
     * @brief Performs the integral
     * \f[
     *    \sum_{q0}\int d\Omega_{d-2} \int_{-1}^1 d\cos \int_0^\infty dq f(q^2, cos, q0) q^{d-2}
     * \f]
     * with \f$q0 \,\in 2\pi\mathbb{Z}\f$ being a Matsubara frequency. The Matsubara sum is performed only for the first
     * \f$|n| <= m_order\f$ summands, the rest is approximated by an integral.
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @param cos_quadrature Quadrature rule for the integral over \f$\cos\f$.
     * @param m_order Number of Matsubara frequencies to be summed over.
     * @param m_quadrature Quadrature rule for the integral over \f$q0\f$.
     * @param m_extent Extent of the integral over \f$q0\f$.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT spatial_angle_integrate_and_sum(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent,
                                       const double k, const QGauss<1> &cos_quadrature, const int m_order,
                                       const QGauss<1> &m_quadrature, const double m_extent, const double T)
    {
      if (is_close(T, 0.))
        return spatial_angle_integrate_and_integrate<NT, d, FUN>(fun, x_quadrature, x_extent, k, cos_quadrature,
                                                                 m_quadrature, m_extent);

      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();
      const auto &cos_q_p = cos_quadrature.get_points();
      const auto &cos_q_w = cos_quadrature.get_weights();
      const int cos_size = cos_quadrature.size();
      const auto &m_q_p = m_quadrature.get_points();
      const auto &m_q_w = m_quadrature.get_weights();
      const int m_size = m_quadrature.size();

      static_assert(d > 0, "Dimension must be greater than zero");
      // Dimension of the spatial integral
      constexpr int ds = d - 1;
      // Area of unit sphere in ds=d-1 dimensions
      const double S_ds = 2. * std::pow(M_PI, ds / 2.) / std::tgamma(ds / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_ds                    // angular integral
                               * powr<-ds>(2. * M_PI); // fourier factors

      // Do a matsubara sum for the first m_order summands
      const NT result_sum = tbb::parallel_reduce(
          tbb::blocked_range3d<int>(0, x_size, 0, cos_size, -m_order, m_order + 1), NT(0),
          [&](const tbb::blocked_range3d<int> &r, NT running_total) -> NT {
            for (int x_it = r.pages().begin(); x_it < r.pages().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              for (int cos_it = r.rows().begin(); cos_it < r.rows().end(); cos_it++) {
                const double cos = (cos_q_p[cos_it][0] - 0.5) * 2.;
                const double cos_weight = cos_q_w[cos_it] * 2.;

                for (int q0_it = r.cols().begin(); q0_it < r.cols().end(); q0_it++) {
                  const double q0 = 2. * M_PI * T * q0_it;

                  running_total +=
                      T // Matsubara sum weight
                      * x_weight * prefactor *
                      (0.5 * powr<ds>(k) * std::pow(x, (ds - 2) / 2.)) // integral over x = p^2 / k^2
                      * 0.5 * cos_weight  // integral over cos(theta), 0.5 removes the factor from the angular integral
                      * fun(q2, cos, q0); // integrand
                }
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });

      // Approximate the rest of the sum by an integral
      const NT result_integral = tbb::parallel_reduce(
          tbb::blocked_range3d<int>(0, x_size, 0, cos_size, 0, m_size), NT(0),
          [&](const tbb::blocked_range3d<int> &r, NT running_total) -> NT {
            for (int x_it = r.pages().begin(); x_it < r.pages().end(); x_it++) {
              const double x = x_q_p[x_it][0] * x_extent;
              const double x_weight = x_q_w[x_it] * x_extent;
              const double q2 = x * powr<2>(k);

              for (int cos_it = r.rows().begin(); cos_it < r.rows().end(); cos_it++) {
                const double cos = (cos_q_p[cos_it][0] - 0.5) * 2.;
                const double cos_weight = cos_q_w[cos_it] * 2.;

                for (int q0_it = r.cols().begin(); q0_it < r.cols().end(); q0_it++) {
                  const double q0 =
                      (2. * m_order * M_PI * T) + m_q_p[q0_it][0] * (m_extent - 2. * (m_order + 1) * M_PI * T);
                  const double m_weight = m_q_w[q0_it] * (m_extent - 2. * m_order * M_PI * T);

                  running_total +=
                      (1. * m_weight / (2. * M_PI)) // Matsubara integral weight
                      * x_weight * prefactor *
                      (0.5 * powr<ds>(k) * std::pow(x, (ds - 2) / 2.)) // integral over x = p^2 / k^2
                      * 0.5 * cos_weight // integral over cos(theta), 0.5 removes the factor from the angular integral
                      * (fun(q2, cos, q0) + fun(q2, cos, -q0)); // integrand
                }
              }
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });
      return result_sum + result_integral;
    }

    template <typename NT, typename FUN>
    NT sum(const FUN &fun, const int m_order, const QGauss<1> &m_quadrature, const double m_extent, const double T)
    {
      const auto &m_q_p = m_quadrature.get_points();
      const auto &m_q_w = m_quadrature.get_weights();
      const int m_size = m_quadrature.size();

      // Do a matsubara sum for the first m_order summands
      const NT result_sum = is_close(T, 0.) ? 0.
                                            : tbb::parallel_reduce(
                                                  tbb::blocked_range<int>(-m_order, m_order + 1), NT(0),
                                                  [&](const tbb::blocked_range<int> &r, NT running_total) -> NT {
                                                    for (int q0_it = r.begin(); q0_it < r.end(); q0_it++) {
                                                      const double q0 = 2. * M_PI * T * q0_it;
                                                      running_total += T * fun(q0);
                                                    }
                                                    return running_total;
                                                  },
                                                  [](const NT a, const NT b) { return a + b; });

      // Approximate the rest of the sum by an integral
      const NT result_integral = tbb::parallel_reduce(
          tbb::blocked_range<int>(0, m_size), NT(0),
          [&](const tbb::blocked_range<int> &r, NT running_total) -> NT {
            for (int q0_it = r.begin(); q0_it < r.end(); q0_it++) {
              const double q0 =
                  (2. * m_order * M_PI * T) + m_q_p[q0_it][0] * (m_extent - 2. * (m_order + 1) * M_PI * T);
              const double m_weight = m_q_w[q0_it] * (m_extent - 2. * m_order * M_PI * T);

              running_total += (1. * m_weight / (2. * M_PI)) // Matsubara integral weight
                               * (fun(q0) + fun(-q0));       // integrand
            }
            return running_total;
          },
          [](const NT a, const NT b) { return a + b; });

      return result_sum + result_integral;
    }
  } // namespace LoopIntegrals
} // namespace DiFfRG
