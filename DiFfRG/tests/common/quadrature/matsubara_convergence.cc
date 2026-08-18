#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/regulators.hh>

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace DiFfRG;

// ---------------------------------------------------------------------------------------------
// Convergence study for the Matsubara (Monien) rule and its T=0 fallback.
//
// predict_size() encodes one answer to "how many nodes does this need":
//     E_max = 2e3 * precision_factor * typical_E,   N = 5 + sqrt(4 E_max / (pi^2 T))
// i.e. the rule is sized to reach 2000x the typical energy scale in frequency. That reach is
// what an unregulated (algebraically decaying) frequency direction needs; a regulator that
// also cuts the frequency direction needs far less. This study measures the required N
// directly, per tail behaviour, so the factor can be chosen per model via
// /integration/matsubara_precision_factor rather than guessed.
//
// Not registered with CTest (setup_benchmark): run the binary by hand.
// ---------------------------------------------------------------------------------------------

namespace
{
  constexpr double target_precision = 1e-9;

  // The reach of the Monien rule with N nodes, x_max ~ pi^2 T N^2 / 4, inverted into the
  // precision_factor that predict_size() would need to produce that N.
  double equivalent_precision_factor(const int N, const double T, const double E)
  {
    const double n = std::max(1., double(N) - 5.);
    return (M_PI * M_PI * T * n * n / 4.) / (2e3 * E);
  }

  struct Family {
    std::string name;
    std::string tail;
    std::function<double(double)> f;                 // summand, as a function of the frequency
    std::function<double(double, double)> reference; // (T, E) -> exact bosonic sum; T=0 -> integral
  };

  // T sum_n 1/(w_n^2 + E^2) = coth(E/2T) / (2E),  and  int dp0/(2pi) 1/(p0^2+E^2) = 1/(2E).
  double ref_pole1(const double T, const double E)
  {
    if (T == 0.) return 1. / (2. * E);
    return 1. / (std::tanh(E / (2. * T)) * 2. * E);
  }

  // T sum_n 1/(w_n^2 + E^2)^2 = -d/dE^2 of the above
  //                           = coth(E/2T)/(4E^3) + csch^2(E/2T)/(8 T E^2),
  // and int dp0/(2pi) 1/(p0^2+E^2)^2 = 1/(4E^3).
  double ref_pole2(const double T, const double E)
  {
    if (T == 0.) return 1. / (4. * powr<3>(E));
    const double x = E / (2. * T);
    const double csch2 = 1. / powr<2>(std::sinh(x));
    return 1. / (std::tanh(x) * 4. * powr<3>(E)) + csch2 / (8. * T * powr<2>(E));
  }

  // The pre-2026-08 vacuum rule, reproduced here so the tangent map can be compared against it
  // directly: 2/3 of the nodes Gauss-Legendre linear on [0, 3E], 1/3 log-spaced on [3E, 1e11 E].
  // Kept local to the study; the library no longer contains it.
  double legacy_vacuum_sum(int m_size, const double typical_E, const std::function<double(double)> &f)
  {
    while (m_size % 3 != 0)
      m_size++;

    Quadrature<double> quad_up(m_size / 3, QuadratureType::legendre);
    Quadrature<double> quad_down(m_size / 3 * 2, QuadratureType::legendre);
    const auto up_n = quad_up.nodes<CPU_memory>();
    const auto up_w = quad_up.weights<CPU_memory>();
    const auto dn_n = quad_down.nodes<CPU_memory>();
    const auto dn_w = quad_down.weights<CPU_memory>();

    const long double div = 3 * std::abs(typical_E);
    const long double extent = 1e11 * std::abs(typical_E);
    const long double log_start = std::log(div);
    const long double log_ext = std::log(extent / div);

    double sum = 0.;
    for (int i = 0; i < m_size / 3 * 2; ++i) {
      const double x = dn_n[i] * div;
      const double w = dn_w[i] * div / (2 * M_PI);
      sum += w * (f(x) + f(-x));
    }
    for (int i = 0; i < m_size / 3; ++i) {
      const double x = std::exp(log_start + log_ext * up_n[i]);
      const double w = (up_w[i] * log_ext * x) / (2 * M_PI);
      sum += w * (f(x) + f(-x));
    }
    return sum;
  }

  // Smallest size in `sizes` whose relative error is below `target_precision`, or -1.
  int required_size(const std::vector<int> &sizes, const std::vector<double> &errors)
  {
    for (size_t i = 0; i < sizes.size(); ++i)
      if (errors[i] < target_precision) return sizes[i];
    return -1;
  }
} // namespace

TEST_CASE("Matsubara convergence study", "[.study][matsubara][quadrature]")
{
  DiFfRG::Init();

  using Regulator = PolynomialExpRegulator<>;

  // The regulated (4D) family: dtR_B(k^2, w^2 + l1^2) / (w^2 + l1^2 + m2 + RB)^2 with k = E.
  // No closed form, so the reference is the rule's own high-N limit -- legitimate here because
  // the question is the convergence rate, not the value. l1 and m2 are swept because a real
  // kernel integrates over l1 too, and the pole in w sits at w^2 = -(l1^2 + m2 + RB): the
  // hardest configuration is the one with the smallest such energy.
  auto regulated = [](const double E, const double l1, const double m2) {
    return [E, l1, m2](const double w) {
      const double k2 = powr<2>(E);
      const double q2 = powr<2>(w) + powr<2>(l1);
      return Regulator::RBdot(k2, q2) / powr<2>(q2 + m2 + Regulator::RB(k2, q2));
    };
  };
  // l1 in units of k, and m2 in units of k^2.
  const std::vector<double> l1_over_k = {0., 0.5, 1., 2., 4.};
  const std::vector<double> m2_over_k2 = {0., 1., -0.9};

  const std::vector<int> sizes = {4,  6,  8,  10, 12,  16,  20,  24,  28,  32,  40,
                                  48, 56, 64, 80, 96,  112, 128, 160, 192, 224, 256,
                                  320, 384, 448, 512};
  const std::vector<double> ratios = {0.1, 0.3, 1., 3., 10., 30., 78., 100., 300., 1000.};

  const double E = 1.0;

  // -------------------------------------------------------------------------------------------
  // Part 1: the Monien branch, closed-form families.
  // -------------------------------------------------------------------------------------------
  for (const auto &fam : std::vector<Family>{
           {"1/(w^2+E^2)", "1/w^2", [E](double w) { return 1. / (powr<2>(w) + powr<2>(E)); }, ref_pole1},
           {"1/(w^2+E^2)^2", "1/w^4", [E](double w) { return 1. / powr<2>(powr<2>(w) + powr<2>(E)); },
            ref_pole2}}) {
    std::printf("\n=== Monien branch, summand %s  (tail %s), E = %g ===\n", fam.name.c_str(), fam.tail.c_str(), E);
    std::printf("%10s %10s %10s %12s %14s\n", "E/T", "N_needed", "N_predict", "err@N_pred", "pf_equivalent");

    for (const double ratio : ratios) {
      const double T = E / ratio;

      std::vector<double> errors;
      for (const int N : sizes) {
        MatsubaraQuadrature<double> mq;
        mq.reinit_with_size(N, T, E);
        const double result = mq.sum(fam.f);
        const double ref = fam.reference(T, E);
        errors.push_back(std::abs(result - ref) / std::abs(ref));
      }

      const int n_needed = required_size(sizes, errors);

      // What the shipped heuristic asks for, and what it actually achieves.
      MatsubaraQuadrature<double> shipped;
      shipped.reinit(T, E, 2, 8, 128, 64, 1.);
      const double shipped_err =
          std::abs(shipped.sum(fam.f) - fam.reference(shipped.get_T(), E)) / std::abs(fam.reference(shipped.get_T(), E));

      std::printf("%10.3g %10d %10zu %12.3e %14.3g\n", ratio, n_needed, shipped.size(), shipped_err,
                  n_needed > 0 ? equivalent_precision_factor(n_needed, T, E) : -1.);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 2: the regulated (4D) family -- the case where the 2e3 reach is expected to be waste.
  // Reference is the rule's own value at the largest size.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== Monien branch, regulated summand RBdot/(q2+m2+RB)^2 (PolynomialExpRegulator), k = %g ===\n", E);
  std::printf("(N_needed is the worst case over l1/k in {0,0.5,1,2,4} and m2/k^2 in {0,1,-0.9})\n");
  std::printf("%10s %10s %10s %14s %16s\n", "E/T", "N_needed", "N_predict", "pf_equivalent", "worst (l1,m2)");
  for (const double ratio : ratios) {
    const double T = E / ratio;

    int worst_n = -1;
    double worst_l1 = 0., worst_m2 = 0.;
    for (const double l1 : l1_over_k)
      for (const double m2 : m2_over_k2) {
        const auto f = regulated(E, l1 * E, m2 * powr<2>(E));

        MatsubaraQuadrature<double> ref_q;
        ref_q.reinit_with_size(sizes.back(), T, E);
        const double ref = ref_q.sum(f);
        if (std::abs(ref) < 1e-300) continue;

        std::vector<double> errors;
        for (const int N : sizes) {
          MatsubaraQuadrature<double> mq;
          mq.reinit_with_size(N, T, E);
          errors.push_back(std::abs(mq.sum(f) - ref) / std::abs(ref));
        }

        const int n = required_size(sizes, errors);
        if (n < 0) { // never converged: strictly worse than any converged case
          worst_n = -1;
          worst_l1 = l1;
          worst_m2 = m2;
          goto done;
        }
        if (n > worst_n) {
          worst_n = n;
          worst_l1 = l1;
          worst_m2 = m2;
        }
      }
  done:

    MatsubaraQuadrature<double> shipped;
    shipped.reinit(T, E, 2, 8, 128, 64, 1.);

    std::printf("%10.3g %10d %10zu %14.3g %8.3g,%7.3g\n", ratio, worst_n, shipped.size(),
                worst_n > 0 ? equivalent_precision_factor(worst_n, T, E) : -1., worst_l1, worst_m2);
  }

  // -------------------------------------------------------------------------------------------
  // Part 3: the T=0 fallback rule. This is the branch a large-Lambda flow spends most of its
  // time in (E/T > ~78), and its unit test only demands 3e-6.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== T=0 fallback rule (reinit_0), E = %g ===\n", E);
  std::printf("%10s %14s %14s\n", "N", "err 1/w^2", "err 1/w^4");
  for (const int N : sizes) {
    MatsubaraQuadrature<double> mq;
    mq.reinit_with_size(N, 0., E);

    const auto f1 = [E](double w) { return 1. / (powr<2>(w) + powr<2>(E)); };
    const auto f2 = [E](double w) { return 1. / powr<2>(powr<2>(w) + powr<2>(E)); };

    const double e1 = std::abs(mq.sum(f1) - ref_pole1(0., E)) / ref_pole1(0., E);
    const double e2 = std::abs(mq.sum(f2) - ref_pole2(0., E)) / ref_pole2(0., E);
    std::printf("%10zu %14.3e %14.3e\n", mq.size(), e1, e2);
  }

  // A pole away from the mapping scale is what the tangent map has to survive: the map is
  // built around typical_E, and a propagator pole at E' != typical_E is a peak of width
  // ~E'/E in theta. This sweep sets vacuum_quad_size.
  std::printf("\n=== T=0 rule, pole at E' != typical_E (map built at typical_E = %g) ===\n", E);
  std::printf("%10s", "N");
  const std::vector<double> pole_ratios = {1., 0.3, 0.1, 0.03, 0.01, 3., 10.};
  for (const double r : pole_ratios)
    std::printf(" %11s%-4.3g", "E'/E=", r);
  std::printf("\n");
  for (const int N : sizes) {
    MatsubaraQuadrature<double> mq;
    mq.reinit_with_size(N, 0., E);
    std::printf("%10zu", mq.size());
    for (const double r : pole_ratios) {
      const double Ep = r * E;
      const auto f = [Ep](double w) { return 1. / (powr<2>(w) + powr<2>(Ep)); };
      std::printf(" %15.3e", std::abs(mq.sum(f) - ref_pole1(0., Ep)) / ref_pole1(0., Ep));
    }
    std::printf("\n");
  }

  // -------------------------------------------------------------------------------------------
  // Part 4: sizing the T=0 rule on realistic integrands. A production kernel is not a single
  // pole: it is a product of propagators at different energies times a regulator insertion.
  // This is what decides vacuum_quad_size.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== T=0 rule on realistic integrands (typical_E = k = %g) ===\n", E);
  {
    // Two poles a decade apart: int dp0/(2pi) 1/((p0^2+a^2)(p0^2+b^2)) = 1/(2ab(a+b)).
    const auto two_pole = [](const double a, const double b) {
      return [a, b](const double p0) { return 1. / ((powr<2>(p0) + powr<2>(a)) * (powr<2>(p0) + powr<2>(b))); };
    };
    const auto two_pole_ref = [](const double a, const double b) { return 1. / (2. * a * b * (a + b)); };

    // The actual vacuum kernel shape: dtR_B / (q^2 + R_B)^2 at l1 = 0, k = E.
    const auto reg = [E](const double p0) {
      const double k2 = powr<2>(E);
      const double q2 = powr<2>(p0);
      return Regulator::RBdot(k2, q2) / powr<2>(q2 + Regulator::RB(k2, q2));
    };
    MatsubaraQuadrature<double> reg_ref_q;
    reg_ref_q.reinit_with_size(512, 0., E);
    const double reg_ref = reg_ref_q.sum(reg);

    // Reference for the regulated case from the legacy rule too, so neither rule is graded
    // against its own limit.
    const double reg_ref_legacy = legacy_vacuum_sum(4096, E, reg);
    std::printf("(regulated reference: tangent@512 = %.15e, legacy@4096 = %.15e, rel diff %.3e)\n", reg_ref,
                reg_ref_legacy, std::abs(reg_ref - reg_ref_legacy) / std::abs(reg_ref));

    std::printf("%8s %14s %14s %14s %14s | %14s %14s\n", "N", "poles k,10k", "poles k,100k", "poles 0.1k,k",
                "regulated", "LEGACY reg", "LEGACY k,10k");
    for (const int N : {24, 32, 40, 48, 56, 64, 96, 128}) {
      MatsubaraQuadrature<double> mq;
      mq.reinit_with_size(N, 0., E);

      const auto rel = [&](double got, double want) { return std::abs(got - want) / std::abs(want); };
      std::printf("%8zu %14.3e %14.3e %14.3e %14.3e | %14.3e %14.3e\n", mq.size(),
                  rel(mq.sum(two_pole(E, 10. * E)), two_pole_ref(E, 10. * E)),
                  rel(mq.sum(two_pole(E, 100. * E)), two_pole_ref(E, 100. * E)),
                  rel(mq.sum(two_pole(0.1 * E, E)), two_pole_ref(0.1 * E, E)), rel(mq.sum(reg), reg_ref_legacy),
                  rel(legacy_vacuum_sum(N, E, reg), reg_ref_legacy),
                  rel(legacy_vacuum_sum(N, E, two_pole(E, 10. * E)), two_pole_ref(E, 10. * E)));
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 5: the number that actually decides vacuum_quad_size. In production the p0 rule is one
  // axis of Integrator_fT_p2*: the spatial integral over l1 carries the measure l1^2 dl1, so the
  // l1 = 0 slice -- where the regulator's edge sits exactly at p0 = k and the p0 rule is at its
  // worst -- contributes with vanishing weight. Measure the error of the *assembled* integral.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== T=0: error of the assembled 3+1D integral (l1^2 dl1 x p0 rule), k = %g ===\n", E);
  {
    const double x_extent = 10.; // representative for PolynomialExpRegulator
    const double l1_max = std::sqrt(x_extent) * E;
    const Quadrature<double> l1_quad(256, QuadratureType::legendre);
    const auto l1_n = l1_quad.nodes<CPU_memory>();
    const auto l1_w = l1_quad.weights<CPU_memory>();

    auto assembled = [&](const int N, bool legacy) {
      double total = 0.;
      for (size_t j = 0; j < l1_n.size(); ++j) {
        const double l1 = l1_n[j] * l1_max;
        const double meas = l1_w[j] * l1_max * powr<2>(l1);
        const auto f = [&](const double p0) {
          const double k2 = powr<2>(E);
          const double q2 = powr<2>(p0) + powr<2>(l1);
          return Regulator::RBdot(k2, q2) / powr<2>(q2 + Regulator::RB(k2, q2));
        };
        if (legacy)
          total += meas * legacy_vacuum_sum(N, E, f);
        else {
          MatsubaraQuadrature<double> mq;
          mq.reinit_with_size(N, 0., E);
          total += meas * mq.sum(f);
        }
      }
      return total;
    };

    const double ref = assembled(512, false);
    std::printf("(reference: tangent@512 = %.15e, legacy@1536 = %.15e)\n", ref, assembled(1536, true));
    std::printf("%8s %16s %16s\n", "N", "tangent", "legacy");
    for (const int N : {16, 24, 32, 40, 48, 56, 64, 96}) {
      std::printf("%8d %16.3e %16.3e\n", N, std::abs(assembled(N, false) - ref) / std::abs(ref),
                  std::abs(assembled(N, true) - ref) / std::abs(ref));
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 6: node coverage against a stored Matsubara grid.
  //
  // Kernels that read a dressing off a frequency grid (QCD_current: BosonicMatsubaraValues /
  // BosonicCoordinates1DFiniteT with p0_grid_size = 8) are evaluated at the Monien nodes, which
  // are continuous and reach ~pi^2 T N^2 / 4. Two things happen off the end of that grid:
  //   - make_interpolation_stencil clamps a non-periodic axis, so the dressing is frozen at its
  //     boundary value (safe, but it is an approximation nobody chose explicitly);
  //   - BosonicMatsubaraValues::backward returns an *integer* index (round(y/(2 pi T))), so the
  //     stencil weight t is always 0 and the read is nearest-mode, never interpolated.
  // This prints how much of the quadrature actually sees stored data.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== Monien node coverage of a stored Matsubara grid ===\n");
  {
    const double T = 0.105; // QCD_current
    const int grid_modes = 8;
    std::printf("grid: %d bosonic modes (n = 0..%d), i.e. |p0| <= %.3g GeV at T = %g\n", grid_modes, grid_modes - 1,
                (grid_modes - 1) * 2 * M_PI * T, T);
    std::printf("%8s %14s %14s %12s %12s\n", "N", "max node", "max mode n", "nodes in grid", "fraction");
    for (const int N : {16, 32, 64, 96, 128}) {
      MatsubaraQuadrature<double> mq;
      mq.reinit_with_size(N, T, 1.0);
      const auto nodes = mq.nodes<CPU_memory>();

      int inside = 0;
      double max_node = 0.;
      for (size_t i = 0; i < nodes.size(); ++i) {
        max_node = std::max(max_node, (double)nodes[i]);
        if (std::round(nodes[i] / (2 * M_PI * T)) <= grid_modes - 1) inside++;
      }
      std::printf("%8zu %14.4g %14.4g %12d %11.1f%%\n", mq.size(), max_node, max_node / (2 * M_PI * T), inside,
                  100. * inside / (double)mq.size());
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 7: how a dressing on a Matsubara-valued axis actually responds to a continuous
  // frequency. Data is set to f[n] = n, so an interpolating read at p0 = r * 2 pi T must return
  // r, while a nearest-mode read returns round(r), and a clamped read returns 7 beyond the grid.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== reading a BosonicMatsubaraValues dressing at a continuous frequency ===\n");
  {
    const double T = 0.105;
    const int grid_modes = 8;
    BosonicMatsubaraValues<int, float> coords(0, grid_modes, (float)T);
    LinearInterpolator1D<double, BosonicMatsubaraValues<int, float>> dressing(coords);
    std::vector<double> data(grid_modes);
    for (int n = 0; n < grid_modes; ++n)
      data[n] = n; // f[n] = n
    dressing.update(data.data());

    std::printf("%12s %14s %14s %14s\n", "p0/(2 pi T)", "exact f = r", "read back", "verdict");
    for (const double r : {0.0, 0.5, 1.0, 3.7, 6.5, 7.0, 20.0, 500.0}) {
      const double p0 = r * 2 * M_PI * T;
      const double got = dressing((float)p0);
      const char *verdict = (std::abs(got - r) < 1e-6)             ? "interpolated"
                            : (std::abs(got - grid_modes + 1) < 1e-6 && r > grid_modes - 1) ? "CLAMPED"
                                                                                            : "nearest-mode";
      std::printf("%12.4g %14.4g %14.4g %14s\n", r, r, got, verdict);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 8: choosing the reach constant C in E_max = C * precision_factor * typical_E.
  // The user-facing dial stays at precision_factor = 1; C is what decides whether that means
  // "sane" or "1000x over-provisioned". For each candidate C this reports the N that
  // predict_size would produce and the error actually achieved at that N.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== candidate reach constants C (E_max = C * typical_E), pf = 1 ===\n");
  {
    // predict_size's formula, with C substituted for the shipped 2e3.
    auto N_for = [](const double C, const double r) {
      int n = 5 + int(std::sqrt(4. * C * r / (M_PI * M_PI)));
      n = (int)std::ceil(n / 2.) * 2;
      return std::max(8, std::min(128, n));
    };

    const std::vector<double> Cs = {2000., 400., 200., 100., 50., 20.};
    const std::vector<double> rs = {0.3, 1., 3., 10., 30., 78.};

    std::printf("%8s", "C");
    for (const double r : rs)
      std::printf("  E/T=%-5.3g N near  far  ", r);
    std::printf("\n");

    for (const double C : Cs) {
      std::printf("%8.4g", C);
      for (const double r : rs) {
        const double T = E / r;
        const int N = N_for(C, r);

        MatsubaraQuadrature<double> mq;
        mq.reinit_with_size(N, T, E);

        // reference for the regulated family: same T, converged size
        const auto freg = regulated(E, 0., 0.);
        MatsubaraQuadrature<double> conv;
        conv.reinit_with_size(512, T, E);
        const double reg_ref = conv.sum(freg);

        // Two poles far apart is the realistic stressor: a real kernel carries a heavy mode as
        // well as the light one, and only the light scale is tracked by typical_E = k.
        // T sum 1/((w^2+a^2)(w^2+b^2)) = [coth(a/2T)/(2a) - coth(b/2T)/(2b)] / (b^2 - a^2).
        const auto two_pole = [](const double a, const double b) {
          return [a, b](const double w) {
            return 1. / ((powr<2>(w) + powr<2>(a)) * (powr<2>(w) + powr<2>(b)));
          };
        };
        const auto two_pole_ref = [T](const double a, const double b) {
          return (ref_pole1(T, a) - ref_pole1(T, b)) / (powr<2>(b) - powr<2>(a));
        };

        const auto rel = [](double a, double b) { return std::abs(a - b) / std::abs(b); };
        // "near" = everything whose structure sits within a decade of typical_E, including the
        // 4D-regulated kernel shape. "x100" = a heavy mode two decades above typical_E, which
        // only reaches the frequency integral when the regulator does NOT cut p0 (3D regulator).
        const double near = std::max({rel(mq.sum([E](double w) { return 1. / (powr<2>(w) + powr<2>(E)); }),
                                          ref_pole1(T, E)),
                                      rel(mq.sum([E](double w) { return 1. / powr<2>(powr<2>(w) + powr<2>(E)); }),
                                          ref_pole2(T, E)),
                                      rel(mq.sum(two_pole(E, 10. * E)), two_pole_ref(E, 10. * E)),
                                      rel(mq.sum(freg), reg_ref)});
        const double far = rel(mq.sum(two_pole(E, 100. * E)), two_pole_ref(E, 100. * E));
        std::printf(" %4d %8.1e %8.1e", N, near, far);
      }
      std::printf("\n");
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 9: where the Monien -> vacuum switch belongs.
  //
  // Lowering C shrinks N, which pushes the existing `N > 2*max_size` fallback to much larger
  // E/T -- and in the band between, a clamped 128-node Monien rule would replace today's
  // 64-node vacuum rule, i.e. cost would DOUBLE. The switch therefore has to be governed by
  // physics (when is the thermal content negligible) rather than by the node ceiling.
  // For a pole at E the relative thermal correction is 2(coth(E/2T) - 1) ~ 4 exp(-E/T).
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== thermal content vs E/T (what switching to the T=0 rule discards) ===\n");
  std::printf("%10s %16s %16s\n", "E/T", "exact 1/(w^2+E^2)", "4*exp(-E/T)");
  for (const double r : {5., 10., 15., 20., 25., 30., 35., 40.}) {
    const double T = E / r;
    const double thermal = std::abs(ref_pole1(T, E) - ref_pole1(0., E)) / ref_pole1(0., E);
    std::printf("%10.3g %16.3e %16.3e\n", r, thermal, 4. * std::exp(-r));
  }

  // -------------------------------------------------------------------------------------------
  // Part 10: the 3D-regulated kernel shape -- the actual risk case for the reach constant.
  // Here the regulator argument carries no p0, so as a function of frequency the summand is a
  // pure double pole at E = sqrt(l1^2 + m2 + RB(k^2, l1^2)) with a 1/p0^4 tail and NO cutoff.
  // With m2 ~ k^2 that pole sits on-scale; the danger is the IR, where the physical mass stays
  // finite while k -> 0, so E/typical_E grows without bound.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== 3D-regulated summand: dtRB(k^2, l1^2) / (p0^2 + l1^2 + m2 + RB)^2 ===\n");
  {
    auto reg3d = [](const double k, const double l1, const double m2) {
      return [k, l1, m2](const double p0) {
        const double k2 = powr<2>(k);
        const double l12 = powr<2>(l1);
        return Regulator::RBdot(k2, l12) / powr<2>(powr<2>(p0) + l12 + m2 + Regulator::RB(k2, l12));
      };
    };
    // Exact bosonic sum: the summand is c/(p0^2+E^2)^2 with E^2 = l1^2+m2+RB, c = dtRB.
    auto reg3d_ref = [&](const double T, const double k, const double l1, const double m2) {
      const double k2 = powr<2>(k), l12 = powr<2>(l1);
      const double E2 = l12 + m2 + Regulator::RB(k2, l12);
      return Regulator::RBdot(k2, l12) * ref_pole2(T, std::sqrt(E2));
    };

    std::printf("%10s %10s %8s %12s %12s %12s\n", "m/k", "E/T", "N(400)", "err reach400", "err reach2000", "N(2000)");
    for (const double m_over_k : {0., 1., 10., 100.})
      for (const double r : {1., 10., 30.}) {
        const double k = E, T = k / r;
        const double m2 = powr<2>(m_over_k * k);
        const auto f = reg3d(k, 0.5 * k, m2);
        const double ref = reg3d_ref(T, k, 0.5 * k, m2);

        auto N_for = [&](const double C) {
          int n = 5 + int(std::sqrt(4. * C * r / (M_PI * M_PI)));
          n = (int)std::ceil(n / 2.) * 2;
          return std::max(8, std::min(128, n));
        };
        MatsubaraQuadrature<double> q400, q2000;
        q400.reinit_with_size(N_for(400.), T, k);
        q2000.reinit_with_size(N_for(2000.), T, k);

        std::printf("%10.4g %10.4g %8d %12.2e %12.2e %12d\n", m_over_k, r, N_for(400.),
                    std::abs(q400.sum(f) - ref) / std::abs(ref), std::abs(q2000.sum(f) - ref) / std::abs(ref),
                    N_for(2000.));
      }
  }

  // -------------------------------------------------------------------------------------------
  // Part 11: the same 3D-regulated kernel along a REAL flow trajectory.
  // m/k and E/T are not independent: a physical mass stays fixed while k runs, so m/k is large
  // exactly where E/T = k/T is small. Sweeping them independently invents a regime
  // (heavy mode AND large E/T) that a flow never visits. Here T and m are fixed and k runs.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== 3D-regulated kernel along a flow: T = 0.1, m = 1 GeV fixed, k running ===\n");
  {
    const double T = 0.1, m = 1.0;
    auto reg3d = [](const double k, const double l1, const double m2) {
      return [k, l1, m2](const double p0) {
        const double k2 = powr<2>(k), l12 = powr<2>(l1);
        return Regulator::RBdot(k2, l12) / powr<2>(powr<2>(p0) + l12 + m2 + Regulator::RB(k2, l12));
      };
    };
    auto N_for = [](const double C, const double r) {
      int n = 5 + int(std::sqrt(4. * C * r / (M_PI * M_PI)));
      n = (int)std::ceil(n / 2.) * 2;
      return std::max(8, std::min(128, n));
    };

    std::printf("%10s %8s %8s %8s %12s %8s %12s\n", "k[GeV]", "E/T", "m/k", "N(400)", "err(400)", "N(2000)",
                "err(2000)");
    for (const double k : {10., 3., 1., 0.3, 0.1, 0.03, 0.01, 0.006}) {
      const double r = k / T;
      const double m2 = powr<2>(m);
      const double l1 = 0.5 * k;
      const double k2 = powr<2>(k), l12 = powr<2>(l1);
      const double Epole = std::sqrt(l12 + m2 + Regulator::RB(k2, l12));
      const auto f = reg3d(k, l1, m2);
      const double ref = Regulator::RBdot(k2, l12) * ref_pole2(T, Epole);

      // Above E/T = 30 the shipped logic hands over to the vacuum rule; flag those rows.
      const bool vac = r > 30.;
      MatsubaraQuadrature<double> q400, q2000;
      q400.reinit_with_size(N_for(400., r), T, k);
      q2000.reinit_with_size(N_for(2000., r), T, k);

      std::printf("%10.4g %8.3g %8.3g %8d %12.2e %8d %12.2e%s\n", k, r, m / k, N_for(400., r),
                  std::abs(q400.sum(f) - ref) / std::abs(ref), N_for(2000., r),
                  std::abs(q2000.sum(f) - ref) / std::abs(ref), vac ? "   <- vacuum branch" : "");
    }
  }

  // -------------------------------------------------------------------------------------------
  // Part 12: the max_matsubara_size clamp is no longer silent. With the shipped defaults it is
  // unreachable (the thermal handover fires first), so provoke it with a raised precision
  // factor: reach 2000 at E/T = 25 asks for ~147 nodes against a ceiling of 128, and E/T = 25
  // is below the handover threshold of 30. Expect exactly one warning on stderr.
  // -------------------------------------------------------------------------------------------
  std::printf("\n=== clamp diagnostic (expect one warning on stderr, then silence) ===\n");
  {
    for (const double r : {25., 26., 27.}) {
      MatsubaraQuadrature<double> mq;
      mq.reinit(E / r, E, 2, 8, 128, 64, 5.);
      std::printf("  E/T = %g -> size %zu\n", r, mq.size());
    }
  }

  SUCCEED("study completed");
}
