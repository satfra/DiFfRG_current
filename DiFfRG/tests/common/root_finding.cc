#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

// deliberately included first and alone, so that this TU also checks that the header is
// self-contained
#include <DiFfRG/common/root_finding.hh>

#include <cmath>
#include <stdexcept>

using namespace DiFfRG;

namespace
{
  // The searches below bracket the root at 1 in [0, 3]. The first bisection point is 1.5, which
  // keeps successes and failures alternating; a bracket whose midpoint sits exactly on the root
  // would produce a single success and then fail forever.
  constexpr double root = 1.;
  constexpr double lower = 0.;
  constexpr double upper = 3.;
} // namespace

TEST_CASE("BisectionRootFinder converges the search variable", "[common][root_finding]")
{
  const double tol = 1e-6;

  BisectionRootFinder search([](const double x) -> bool { return x >= root; }, tol, 100);
  search.set_bounds(lower, upper);

  const double x = search.search();

  CHECK(x >= root);
  CHECK(std::abs(x - root) < tol);
}

TEST_CASE("BisectionRootFinderTarget converges the target value", "[common][root_finding]")
{
  const double tol = 1e-6;

  uint calls = 0;
  BisectionRootFinderTarget search(
      [&](const double x, double &target) -> bool {
        // the counter must name the step currently being taken
        CHECK(search.get_iter() == calls);
        ++calls;

        if (x < root) return false; // failures leave the target untouched
        target = x * x;
        return true;
      },
      tol, 100);
  search.set_bounds(lower, upper);

  const double x = search.search();

  CHECK(search.converged());
  CHECK(x >= root);
  CHECK(search.get_target() == Catch::Approx(x * x));
  // |t_k - t_{k-1}| = |x_k^2 - x_{k-1}^2| ~ 2 |x_k - x_{k-1}|, so x is resolved to about tol/2
  CHECK(std::abs(x - root) < tol);
}

TEST_CASE("BisectionRootFinderTarget stops on a flat target", "[common][root_finding]")
{
  BisectionRootFinderTarget search(
      [](const double x, double &target) -> bool {
        target = 42.;
        return x >= root;
      },
      1e-6, 100);
  search.set_bounds(lower, upper);

  const double x = search.search();

  // an x-insensitive target is "converged" as soon as two successes have been seen
  CHECK(search.converged());
  CHECK(search.get_target() == Catch::Approx(42.));
  CHECK(x >= root);
  CHECK(search.get_iter() <= 4);
}

TEST_CASE("BisectionRootFinderTarget reports stagnation instead of throwing", "[common][root_finding]")
{
  // an unreachable tolerance: |t_k - t_{k-1}| < 0 is never true
  BisectionRootFinderTarget search(
      [](const double x, double &target) -> bool {
        target = x * x;
        return x >= root;
      },
      0., 200);
  search.set_bounds(lower, upper);

  double x = 0.;
  REQUIRE_NOTHROW(x = search.search());

  CHECK_FALSE(search.converged());
  CHECK(x >= root);
  CHECK(std::abs(x - root) < 1e-12); // the bracket collapsed all the way down
  CHECK(search.get_target() == Catch::Approx(x * x));
}

TEST_CASE("BisectionRootFinderTarget honours a custom collapse tolerance", "[common][root_finding]")
{
  BisectionRootFinderTarget search(
      [](const double x, double &target) -> bool {
        target = x * x;
        return x >= root;
      },
      0., 200);
  search.set_bounds(lower, upper);
  search.set_x_collapse_tol(1e-3);

  const double x = search.search();

  CHECK_FALSE(search.converged());
  CHECK(x >= root);
  CHECK(std::abs(x - root) < 1e-2);
  CHECK(search.get_iter() < 30);
}

TEST_CASE("BisectionRootFinderTarget throws without a successful evaluation", "[common][root_finding]")
{
  BisectionRootFinderTarget search([](const double, double &) -> bool { return false; }, 1e-6, 100);
  search.set_bounds(lower, upper);

  CHECK_THROWS_AS(search.search(), std::runtime_error);
}

TEST_CASE("BisectionRootFinderTarget throws without bounds", "[common][root_finding]")
{
  BisectionRootFinderTarget search(
      [](const double x, double &target) -> bool {
        target = x;
        return x >= root;
      },
      1e-6, 100);

  CHECK_THROWS_AS(search.search(), std::runtime_error);
}
