// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>

// Standard library
#include <cmath>
#include <string>

namespace DiFfRG
{
  namespace
  {
    /**
     * @brief How far, in units of typical_E, the Monien rule is asked to reach in frequency.
     *
     * The Monien rule with N nodes resolves frequencies up to x_max ~ pi^2 T N^2 / 4, so this
     * constant is what sets N. It exists purely to cover structure ABOVE typical_E: anything
     * within a decade of typical_E (including the shape of a 4D-regulated kernel) is already at
     * machine precision for a reach of ~50. What costs reach is a second, heavier scale in the
     * frequency direction -- which only reaches the sum when the regulator does not cut p0.
     *
     * Measured worst-case relative error (tests/common/quadrature/matsubara_convergence.cc,
     * `[.study]`), over E/T in [0.3, 78], for a heavy mode 100x above typical_E:
     *
     *   reach   2000 -> 8.4e-10     400 -> 1.1e-08     200 -> 1.3e-05     100 -> 3.5e-04
     *
     * and for everything within a decade of typical_E, all of these stay at ~1e-15.
     *
     * 400 keeps a 4x margin over a two-decade scale separation at ~1e-8 -- comfortably under the
     * tolerances the stiff timesteppers run at -- for about half the nodes of the old 2000.
     * The user-facing dial is /integration/matsubara_precision_factor, which multiplies this;
     * setting it to 5 restores the pre-2026-08 *reach* of 2000 (it does not restore the old
     * handover point, which is now decided by thermal_negligible_ratio below).
     */
    constexpr double monien_reach = 400.;

    /**
     * @brief Above this E/T there is no thermal content left to resolve.
     *
     * The relative thermal correction to a pole at E is 2(coth(E/2T) - 1) ~ 4 exp(-E/T):
     * 4.1e-9 at E/T = 20, 1.9e-13 at E/T = 30, 1.3e-15 at E/T = 35 (measured, same study).
     * Past that the vacuum rule is the better answer, and with the tangent map it is also an
     * accurate one.
     */
    constexpr double thermal_negligible_ratio = 30.;
  } // namespace

  template <typename NT> int MatsubaraQuadrature<NT>::predict_size(const NT T, const NT typical_E, const int step)
  {
    if (is_close(T, NT{})) return -vacuum_quad_size;

    const NT ratio = std::fabs(typical_E) / (std::fabs(T) + NT(1e-16)); // E/T

    const NT E_max = monien_reach * precision_factor * std::fabs(typical_E);
    int size = 5 + int(std::sqrt(4. * E_max / (M_PI * M_PI * std::fabs(T))));
    size = (int)std::ceil(size / (double)step) * step;

    // Hand over to the vacuum rule once the thermal content is gone -- but never to a rule that
    // costs MORE than the one it replaces. Deciding this on physics rather than on the node
    // ceiling matters: with a smaller reach the old `size > 2 * max_size` test would not fire
    // until E/T ~ 390, and the band below it would run a clamped 128-node sum where a 64-node
    // vacuum rule is both cheaper and, thermally, indistinguishable.
    if (ratio > thermal_negligible_ratio && size > vacuum_quad_size) return -vacuum_quad_size;

    return size;
  }

  template <typename NT>
  MatsubaraQuadrature<NT>::MatsubaraQuadrature(const NT T, const NT typical_E, const int step, const int min_size,
                                               const int max_size, const int vacuum_quad_size,
                                               const double precision_factor)
  {
    reinit(T, typical_E, step, min_size, max_size, vacuum_quad_size, precision_factor);
  }

  template <typename NT> MatsubaraQuadrature<NT>::MatsubaraQuadrature() : m_size(0), vacuum_quad_size(64) {}

  template <typename NT>
  void MatsubaraQuadrature<NT>::reinit(const NT T, const NT typical_E, const int step, const int min_size,
                                       const int max_size, const int vacuum_quad_size, const double precision_factor)
  {
    if (precision_factor <= 0)
      this->precision_factor = 1;
    else
      this->precision_factor = precision_factor;
    // Warn on the *effective* factor: a non-positive argument has just been coerced to 1, so
    // testing the argument here warned on every default-constructed QuadratureProvider.
    if (this->precision_factor < 1e-1)
      std::cerr
          << "MatsubaraQuadrature: precision_factor is very small (< 1e-1), which may lead to inaccurate results.";

    // This is ludicrously low, but let's cut somewhere
    if (vacuum_quad_size <= 16)
      this->vacuum_quad_size = 16;
    else
      this->vacuum_quad_size = vacuum_quad_size;

    if (max_size < min_size) throw std::invalid_argument("MatsubaraQuadrature: max_size must be larger than min_size.");

    this->T = T;
    this->typical_E = typical_E;

    // Determine the number of nodes in the quadrature rule.
    m_size = predict_size(T, typical_E, step);
    // If m_size is negative, we use the vacuum quadrature
    if (m_size < 0) {
      m_size = abs(m_size);
      reinit_0();
      return;
    }
    // If predict_size wants far more nodes than max_size allows, the truncated Monien
    // quadrature would be less accurate than the vacuum fallback.
    if (m_size > 2 * max_size) {
      m_size = abs(vacuum_quad_size);
      reinit_0();
      return;
    }
    // Truncating to max_size silently violates the resolution law E_max ~ pi^2 N^2 T / 4 -- the
    // rule then reaches (max_size/predicted)^2 less far in frequency than it was asked to, with
    // no other symptom. Say so once; it means max_matsubara_size (or precision_factor) is
    // fighting the sizing rather than bounding it.
    if (m_size > max_size) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        std::cerr << "MatsubaraQuadrature: requested size " << m_size << " exceeds max_matsubara_size " << max_size
                  << " (T = " << T << ", typical_E = " << typical_E << "); the rule reaches "
                  << (double(m_size) / max_size) * (double(m_size) / max_size)
                  << "x less far in frequency than the sizing asked for. Raise max_matsubara_size, or lower "
                     "matsubara_precision_factor if that reach is not needed. (warned once)\n";
      }
    }
    m_size = std::max(min_size, std::min(max_size, m_size));

    build_monien();
  }

  template <typename NT>
  void MatsubaraQuadrature<NT>::reinit_with_size(const int size, const NT T, const NT typical_E)
  {
    if (size <= 0) throw std::invalid_argument("MatsubaraQuadrature: size must be positive.");

    this->T = T;
    this->typical_E = typical_E;
    m_size = size;

    if (is_close(T, NT{})) {
      reinit_0();
      return;
    }
    build_monien();
  }

  template <typename NT> int MatsubaraQuadrature<NT>::modes_below(const NT T, const NT freq_cutoff)
  {
    if (!(T > NT(0)) || !(freq_cutoff > NT(0))) return 0;
    const double n = std::floor(double(std::fabs(freq_cutoff)) / (2. * M_PI * double(std::fabs(T))));
    // Nothing here bounds the count: a caller that hands in a cutoff far above 2 pi T is asking
    // for a sum that large, and it is the caller's job (see QuadratureIntegrator_fT) to prefer
    // the Monien rule once it is the cheaper of the two.
    return n < 0. ? 0 : int(n);
  }

  template <typename NT> void MatsubaraQuadrature<NT>::reinit_exact_sum(const NT T, const NT freq_cutoff)
  {
    if (!(T > NT(0))) throw std::invalid_argument("MatsubaraQuadrature: reinit_exact_sum needs a positive T.");

    this->T = T;
    // Not a "typical energy" in the predict_size sense -- it is the hard support boundary. Stored
    // here so get_typical_E() still identifies which rule this is, and so write_data's view names
    // stay distinguishable.
    this->typical_E = freq_cutoff;
    m_size = modes_below(T, freq_cutoff);

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);
    for (int i = 0; i < m_size; ++i) {
      x[i] = NT(2. * M_PI * double(T) * double(i + 1));
      w[i] = T;
    }

    write_data(x, w);
  }

  template <typename NT>
  void MatsubaraQuadrature<NT>::reinit_finite_interval(const NT cutoff,
                                                       const Kokkos::View<const NT *, CPU_memory> gl_nodes,
                                                       const Kokkos::View<const NT *, CPU_memory> gl_weights)
  {
    if (!(cutoff > NT(0)))
      throw std::invalid_argument("MatsubaraQuadrature: reinit_finite_interval needs a positive cutoff.");

    // Zero, like the T=0 rule: this replaces the sum by an integral, so there is no separate zero
    // mode and sum_size() == size().
    this->T = 0;
    this->typical_E = cutoff;
    m_size = (int)gl_nodes.size();

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);
    for (int i = 0; i < m_size; ++i) {
      x[i] = cutoff * gl_nodes[i];
      w[i] = cutoff * gl_weights[i] / NT(2 * M_PI);
    }

    write_data(x, w);
  }

  template <typename NT> void MatsubaraQuadrature<NT>::build_monien()
  {
    // construct the recurrence relation for the quadrature rule from [1]
    std::vector<NT> a(m_size, 0.);
    std::vector<NT> b(m_size, 0.);

    for (int j = 0; j < m_size; ++j) {
      const double j1 = j + 1;
      a[j] = 2 * powr<2>(M_PI) / (4 * j + 1) / (4 * j + 5);
      b[j] = powr<4>(M_PI) / ((4 * j1 - 1) * (4 * j1 + 3)) / powr<2>(4 * j1 + 1);
    }
    a[0] = powr<2>(M_PI) / 15.;

    const NT mu0 = powr<2>(M_PI) / 6.;

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);

    // compute the nodes and weights of the quadrature rule
    make_quadrature(a, b, mu0, x, w);

    // normalize the weights and scale the nodes
    for (int i = 0; i < m_size; ++i) {
      w[i] = T * w[i] / x[i];
      x[i] = 2. * M_PI * T / std::sqrt(x[i]);
    }

    write_data(x, w);
  }

  /**
   * @brief The vacuum (T=0) rule for the frequency integral.
   *
   * At T=0 the Matsubara sum degenerates to \f$\int dp_0/(2\pi) f(p_0)\f$. Substituting
   * \f$p_0 = E \tan\theta\f$, \f$\theta\in[0,\pi/2)\f$, maps the canonical propagator structure
   * to a constant: for \f$f = 1/(p_0^2+E^2)\f$ the transformed integrand is exactly \f$1/E\f$,
   * so Gauss-Legendre in \f$\theta\f$ is near-exact. An algebraic \f$c/p_0^2\f$ tail maps to the
   * finite limit \f$c/E\f$ at \f$\theta\to\pi/2\f$ and is thus *covered* by the quadrature
   * rather than truncated.
   *
   * The previous construction (2/3 of the nodes linear on [0,3E], 1/3 log-spaced on
   * [3E, 1e11 E]) truncated the tail at 1e11 E, which floored the relative error of a
   * 1/p0^2 tail at ~6e-12 regardless of the node count -- going from 81 to 513 nodes did not
   * improve it at all. The tangent map has no such floor.
   *
   * Caller convention (see sum()): \f$\sum_i w_i (f(x_i)+f(-x_i)) = \int dp_0/(2\pi) f\f$, so
   * for even f we need \f$\sum_i w_i f(x_i) = (1/2\pi)\int_0^\infty f\,dp_0\f$, giving
   * \f[ x_i = E\tan\theta_i, \qquad w_i = \frac{E\,w^{GL}_i}{4\cos^2\theta_i}. \f]
   * Everything is linear in E, i.e. this is one dimensionless table times a single scale.
   */
  template <typename NT> void MatsubaraQuadrature<NT>::reinit_0()
  {
    this->T = 0;
    m_size = abs(m_size);
    if (is_close(typical_E, NT(0))) this->typical_E = 1.;

    // gauss-legendre nodes/weights on [0, 1]
    Quadrature<NT> quad(m_size, QuadratureType::legendre);
    const auto gl_nodes = quad.template nodes<CPU_memory>();
    const auto gl_weights = quad.template weights<CPU_memory>();

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);

    using std::abs, std::cos, std::sin;
    const long double E = abs((long double)typical_E);
    const long double half_pi = (long double)M_PI / 2.0L;

    for (int i = 0; i < m_size; ++i) {
      // Parameterise by the complement delta = pi/2 - theta: the large nodes are the ones
      // that matter for the tail, and cot(delta) is far better conditioned there than
      // tan(theta) with theta -> pi/2.
      const long double delta = half_pi * (1.0L - (long double)gl_nodes[i]);
      const long double s = sin(delta);

      x[i] = (NT)(E * cos(delta) / s);
      w[i] = (NT)(E * (long double)gl_weights[i] / (4.0L * s * s));
    }

    write_data(x, w);
  }

  template <typename NT> void MatsubaraQuadrature<NT>::write_data(const std::vector<NT> &x, const std::vector<NT> &w)
  {
    std::string name = "matsubara_T" + std::to_string(T) + "_typicalE" + std::to_string(typical_E);

    // One slot past the rule itself, holding the zero mode; see sum_nodes(). Allocated
    // unconditionally so that nodes()/sum_nodes() are two subviews of one buffer -- the T=0 rule
    // simply never exposes the extra slot.
    const size_t alloc = size_t(m_size) + 1;

    // copy nodes and weights to std::vector
    device_nodes = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_nodes_" + name + "_" + std::to_string(m_size), alloc);
    device_weights = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_weights_" + name + "_" + std::to_string(m_size), alloc);

    // create a host mirror view
    auto host_mirror_nodes = Kokkos::create_mirror_view(device_nodes);
    auto host_mirror_weights = Kokkos::create_mirror_view(device_weights);

    // copy data from gsl to host mirror view
    for (int i = 0; i < m_size; ++i) {
      host_mirror_nodes(i) = x[i];
      host_mirror_weights(i) = w[i];
    }
    // The zero mode as an ordinary node: a summand evaluated as w * (f(+x) + f(-x)) reproduces
    // T * f(0) exactly at x = 0, w = T/2 (the division by two is exact in binary). Zero-weight and
    // harmless for the T=0 rule, which does not expose it anyway.
    host_mirror_nodes(m_size) = NT(0);
    host_mirror_weights(m_size) = T / NT(2);

    // copy data from host mirror view to device view
    Kokkos::deep_copy(device_nodes, host_mirror_nodes);
    Kokkos::deep_copy(device_weights, host_mirror_weights);

    // copy data from host mirror view to host view
    host_nodes = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_nodes_" + name + "_" + std::to_string(m_size), alloc);
    host_weights = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_weights_" + name + "_" + std::to_string(m_size), alloc);
    Kokkos::deep_copy(host_nodes, host_mirror_nodes);
    Kokkos::deep_copy(host_weights, host_mirror_weights);
  }

  template <typename NT> size_t MatsubaraQuadrature<NT>::size() const { return m_size; }
  template <typename NT> size_t MatsubaraQuadrature<NT>::sum_size() const
  {
    // The T=0 rule integrates rather than sums: its "zero mode" has weight zero, so exposing it
    // would only buy a wasted kernel evaluation per spatial point.
    return size_t(m_size) + (is_close(T, NT{}) ? 0u : 1u);
  }
  template <typename NT> NT MatsubaraQuadrature<NT>::get_T() const { return T; }
  template <typename NT> NT MatsubaraQuadrature<NT>::get_typical_E() const { return typical_E; }

  // explicit instantiation
  template class MatsubaraQuadrature<double>;
  template class MatsubaraQuadrature<float>;
} // namespace DiFfRG