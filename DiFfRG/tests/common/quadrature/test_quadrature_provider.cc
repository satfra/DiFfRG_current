#include "DiFfRG/common/initialize.hh"
#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/initialize.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

using namespace DiFfRG;

TEST_CASE("Test quadrature provider", "[double][quadrature]")
{
  DiFfRG::Initialize();

  const double T = GENERATE(take(1, random(0.1, 1.0)));
  const double start_k = GENERATE(take(1, random(1.0, 100.0)));
  const double speed_k = GENERATE(take(2, random(1e-5, 1e-2)));
  const int steps = GENERATE(take(2, random(1e+1, 1e+4)));

  const size_t order = GENERATE(take(2, random(8, 64)));

  DiFfRG::QuadratureProvider quadrature_provider;

  SECTION("Bare test")
  {
    double point = 0;
    double weight = 0;

    for (int j = 0; j < steps; ++j) {
      const double k = std::exp((long double)(-j * speed_k)) * start_k;

      const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(order);
      const auto matsubara_nodes = quadrature_provider.template matsubara_nodes<double, CPU_memory>(T, k);
      const auto weights = quadrature_provider.template weights<double, CPU_memory>(order);
      const auto matsubara_weights = quadrature_provider.template matsubara_weights<double, CPU_memory>(T, k);

      for (uint i = 0; i < order; ++i) {
        point += nodes[i] / (double)steps;
        weight += weights[i] / (double)steps;
        if (!isfinite(nodes[i]) || !isfinite(weights[i]))
          throw std::runtime_error("Not finite: Quadrature nodes or weights");
      }
      for (uint i = 0; i < matsubara_nodes.size(); ++i) {
        point += matsubara_nodes[i] / (double)steps;
        weight += matsubara_weights[i] / (double)steps;

        if (!isfinite(matsubara_nodes[i]) || !isfinite(matsubara_weights[i])) {
          std::string error =
              "Not finite: Matsubara nodes or weights for k = " + std::to_string(k) + " and T = " + std::to_string(T);
          error += "\nMatsubara size is " + std::to_string(matsubara_nodes.size());
          throw std::runtime_error(error);
        }
      }
    }

    REQUIRE(point != Catch::Approx(0.));
    REQUIRE(weight != Catch::Approx(0.));
  }

  SECTION("Test with integration") {}
}
