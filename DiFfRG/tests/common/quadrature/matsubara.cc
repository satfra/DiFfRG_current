#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>

#include <array>
#include <cmath>
#include <string_view>

using namespace DiFfRG;

namespace
{
  void check_matsubara_nodes_and_weights(const MatsubaraQuadrature<double> &mq, const double T,
                                         const double typical_E, const std::string_view context)
  {
    const auto nodes = mq.nodes<CPU_memory>();
    const auto weights = mq.weights<CPU_memory>();

    REQUIRE(mq.size() > 0);
    for (std::size_t i = 0; i < mq.size(); ++i) {
      const double node = nodes[i];
      const double weight = weights[i];
      CAPTURE(context, T, typical_E, mq.size(), i, node, weight);
      REQUIRE(std::isfinite(node));
      REQUIRE(std::isfinite(weight));
      REQUIRE(node > 0.);
      REQUIRE(weight > 0.);
    }
  }
} // namespace

TEST_CASE("Test matsubara quadrature rule", "[double][quadrature][matsubara]")
{
  DiFfRG::Init();

  SECTION("Simple test")
  {
    const double T = GENERATE(take(64, random(0.01, 1.0)));
    MatsubaraQuadrature<double> mq(T, 1.);

    check_matsubara_nodes_and_weights(mq, T, 1., "simple generated quadrature");

    const auto f = [&](const double x) { return 1. / (1. + powr<2>(x)); };

    const double reference = 0.5 * 1. / std::tanh(0.5 / T);
    const double sum = mq.sum(f);

    CAPTURE(T, mq.size(), sum, reference);
    constexpr double expected_precision = 1e-9;
    CHECK(sum == Catch::Approx(reference).epsilon(expected_precision));
  }

  SECTION("Test with different parameters")
  {
    const double T = GENERATE(take(8, random(0.001, 0.1)));
    const double a = GENERATE(take(8, random(0.1, 1.0)));
    const double b = GENERATE(take(8, random(0.01, 0.1)));

    MatsubaraQuadrature<double> mq(T, a);

    check_matsubara_nodes_and_weights(mq, T, a, "generated parameter quadrature");

    const auto f = [&](const double x) { return 1. / (powr<2>(a) + b * x + powr<2>(x)); };

    // Analytically this is sinh(X) / (R * (cosh(X) - cos(y))) with R = sqrt(4a^2 - b^2),
    // X = R/(2T), y = b/(2T) -- but it must not be evaluated in that form. Over these parameter
    // ranges X reaches ~10^3, so sinh(X) and cosh(X) are astronomically large (they overflow
    // outright near the corner T=0.001, a=1) and the value is recovered only as their ratio.
    // Dividing through by cosh(X) first keeps every intermediate bounded: tanh -> 1 and
    // cos(y)/cosh(X) -> 0 (or exactly 0 once cosh overflows, which is the right limit).
    // Verified against a direct Matsubara sum; the unnormalised form disagreed with it by
    // factors of ~10^3 on a few percent of draws, which is what made this test flaky.
    const double R = std::sqrt(4 * std::pow(a, 2) - std::pow(b, 2));
    const double reference = std::tanh(R / (2. * T)) / (R * (1. - std::cos(b / (2. * T)) / std::cosh(R / (2. * T))));

    const double result = mq.sum(f);

    CAPTURE(T, a, b, mq.size(), result, reference);
    constexpr double expected_precision = 1e-6;
    CHECK(result == Catch::Approx(reference).epsilon(expected_precision));
  }

  SECTION("Test at T=0")
  {
    const double T = 0.;
    const double a = GENERATE(take(64, random(0.01, 10.0)));

    MatsubaraQuadrature<double> mq(T, a);

    check_matsubara_nodes_and_weights(mq, T, a, "vacuum generated quadrature");

    const auto f = [&](const double x) { return 1. / (powr<2>(a) + powr<2>(x)); };

    const double reference = 1 / (2. * a);

    const double result = mq.sum(f);

    CAPTURE(T, a, mq.size(), result, reference);
    // The tangent-map rule is essentially exact on this integrand: p0 = a tan(theta) sends
    // 1/(a^2+p0^2) dp0 to the constant d(theta)/a. The old two-piece linear+log rule needed
    // ~60 nodes for 1e-9 here.
    constexpr double expected_precision = 1e-12;
    CHECK(result == Catch::Approx(reference).epsilon(expected_precision));
  }

  SECTION("Test at T=0, pole away from the mapping scale")
  {
    // The rule is built around typical_E, but a propagator pole sits at its own energy. A
    // mismatch of a decade either way is routine (spatial momentum up to sqrt(x_extent)*k,
    // masses above or below k), so it must not cost accuracy at the shipped node count.
    const double T = 0.;
    const double typical_E = 1.;
    const double ratio = GENERATE(0.1, 0.3, 1.0, 3.0, 10.0);
    const double pole = ratio * typical_E;

    MatsubaraQuadrature<double> mq(T, typical_E, 2, 0, 256, 64);

    const auto f = [&](const double x) { return 1. / (powr<2>(pole) + powr<2>(x)); };
    const double reference = 1 / (2. * pole);
    const double result = mq.sum(f);

    CAPTURE(T, typical_E, pole, mq.size(), result, reference);
    CHECK(result == Catch::Approx(reference).epsilon(1e-11));
  }

  SECTION("Test at T=0, a 1/p0^2 tail has no error floor")
  {
    // This is the regression guard for the defect the tangent map fixes. The old rule
    // truncated the frequency integral at 1e11 * typical_E, which floored the relative error
    // of a 1/p0^2 tail at ~6.4e-12 -- going from 81 to 513 nodes did not improve it at all.
    // An unregulated temporal direction is exactly this case, so the floor has to stay gone.
    const double a = 1.;
    const auto f = [&](const double x) { return 1. / (powr<2>(a) + powr<2>(x)); };
    const double reference = 1 / (2. * a);

    for (const int size : {32, 64, 128, 256}) {
      MatsubaraQuadrature<double> mq;
      mq.reinit_with_size(size, 0., a);
      const double err = std::abs(mq.sum(f) - reference) / reference;
      CAPTURE(size, mq.size(), err);
      CHECK(err < 1e-13);
    }
  }
}

TEST_CASE("Matsubara quadrature nodes and weights are finite for representative parameters",
          "[double][quadrature][matsubara][diagnostic]")
{
  DiFfRG::Init();

  constexpr std::array<double, 8> temperatures = {0., 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 5.0e-1, 1.0};
  constexpr std::array<double, 8> typical_energies = {1.0e-2, 5.0e-2, 1.0e-1, 2.5e-1,
                                                      5.0e-1, 1.0,    2.0,    10.0};

  for (const double T : temperatures) {
    for (const double typical_E : typical_energies) {
      MatsubaraQuadrature<double> mq(T, typical_E);
      check_matsubara_nodes_and_weights(mq, T, typical_E, "representative quadrature");
    }
  }
}
