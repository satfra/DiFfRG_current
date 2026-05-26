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

    const double reference =
        -(std::sinh(std::sqrt(4 * std::pow(a, 2) - std::pow(b, 2)) / (2. * T)) /
          (std::sqrt(4 * std::pow(a, 2) - std::pow(b, 2)) *
           (std::cos(b / (2. * T)) - std::cosh(std::sqrt(4 * std::pow(a, 2) - std::pow(b, 2)) / (2. * T)))));

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
    constexpr double expected_precision = 3e-6;
    CHECK(result == Catch::Approx(reference).epsilon(expected_precision));
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
