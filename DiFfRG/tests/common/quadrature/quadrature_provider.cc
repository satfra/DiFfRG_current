#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/discretization/data/output_path.hh>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace DiFfRG;

namespace
{
  DiFfRG::ConfigTree quadrature_log_json(const std::filesystem::path &folder, const bool quadrature_log)
  {
    return DiFfRG::json::value(
        {{"output", {{"folder", folder.string()}, {"name", "run"}, {"quadrature_log", quadrature_log}}}});
  }

  std::string read_file(const std::filesystem::path &path)
  {
    std::ifstream stream(path);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  }
} // namespace

TEST_CASE("Test quadrature provider", "[double][quadrature]")
{
  DiFfRG::Init();

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
}

TEST_CASE("Quadrature provider writes its own quadrature log", "[quadrature][output]")
{
  DiFfRG::Init();

  auto path = DiFfRG::OutputPath::temporary(TemporaryRetention::remove_on_destruction, "run", "run");
  const auto log_file = path.run_file("_quadrature", ".log");

  SECTION("Enabled by default")
  {
    {
      DiFfRG::QuadratureProvider quadrature_provider(quadrature_log_json(path.root(), true));
      // Requesting a quadrature has to be reported, since this is what the file exists for. The provider owns the
      // logger, so the records are only guaranteed on disk once it has been destroyed.
      const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(32);
      REQUIRE(nodes.size() == 32);
    }

    REQUIRE(std::filesystem::exists(log_file));
    const auto contents = read_file(log_file);
    CHECK_THAT(contents, Catch::Matchers::ContainsSubstring("Initialized quadrature provider"));
    CHECK_THAT(contents, Catch::Matchers::ContainsSubstring("order = 32"));
  }

  SECTION("Suppressed by /output/quadrature_log")
  {
    {
      DiFfRG::QuadratureProvider quadrature_provider(quadrature_log_json(path.root(), false));
      const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(32);
      REQUIRE(nodes.size() == 32);
    }

    CHECK_FALSE(std::filesystem::exists(log_file));
  }

  SECTION("An explicit port takes precedence over the own log")
  {
    {
      Config::OutputSettings settings;
      RunLogger run_logger(path, settings, true, RunLoggerOptions{"", "run", false});

      DiFfRG::QuadratureProvider quadrature_provider(quadrature_log_json(path.root(), true), run_logger.port());
      const auto nodes = quadrature_provider.template nodes<double, CPU_memory>(32);
      REQUIRE(nodes.size() == 32);
    }

    // The messages belong to the port that was handed in, so no side-channel file is opened.
    CHECK_FALSE(std::filesystem::exists(log_file));
    CHECK_THAT(read_file(path.run_file(".log")), Catch::Matchers::ContainsSubstring("order = 32"));
  }
}
