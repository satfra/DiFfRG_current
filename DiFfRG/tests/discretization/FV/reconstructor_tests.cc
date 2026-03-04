#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/abstract_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/tvd_reconstructor.hh>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <cstddef>

using NumberType = double;
using namespace dealii;
using Reconstructor = DiFfRG::def::TVDReconstructor<DiFfRG::def::MinModLimiter>;

// ──────────────────────────────────────────────────────────────────────
// Concept check
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVDReconstructor<MinModLimiter> satisfies HasReconstructor concept", "[FV][reconstructor]")
{
  static_assert(DiFfRG::def::HasReconstructor<Reconstructor>);
}

// ──────────────────────────────────────────────────────────────────────
// 1D, one component (moved from KurganovTadmor_tests.cc)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation in 1D, one component", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 1;
  constexpr int n_faces = 2 * dim;
  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {2.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Symmetric neighbours - full gradient")
  {
    u_n = {{{1.0}, {3.0}}};
    NumberType reference = 1.0;
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }

  SECTION("Asymmetric neighbours - smaller slope selected")
  {
    u_n = {{{1.0}, {2.5}}};
    NumberType reference = 0.5;
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }

  SECTION("Opposite-sign slopes - clipped to zero")
  {
    u_n = {{{1.0}, {1.0}}};
    NumberType reference = 0.0;
    auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));

    u_n = {{{5.0}, {4.0}}};
    u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }
}

// ──────────────────────────────────────────────────────────────────────
// 1D, two components (moved from KurganovTadmor_tests.cc)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation in 1D, two components", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 2;
  constexpr int n_faces = 2 * dim;
  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {2.0, 3.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Symmetric neighbours - both components")
  {
    u_n = {{{1.0, 2.0}, {3.0, 4.0}}};
    NumberType reference = 1.0;
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }

  SECTION("Asymmetric neighbours - smaller slope for both components")
  {
    u_n = {{{1.0, 2.0}, {2.5, 3.5}}};
    NumberType reference = 0.5;
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }

  SECTION("Opposite-sign slopes - clipped to zero for both components")
  {
    u_n = {{{1.0, 2.0}, {1.0, 2.0}}};
    NumberType reference = 0.0;
    auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));

    u_n = {{{5.0, 1.0}, {4.0, 0.0}}};
    u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }
}

// ──────────────────────────────────────────────────────────────────────
// 2D, one component (moved from KurganovTadmor_tests.cc)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation in 2D, one component", "[FV][reconstructor]")
{
  constexpr int dim = 2;
  constexpr int n_components = 1;
  constexpr int n_faces = 2 * dim;
  const Point<dim> x_center(1.0, 1.0);
  const std::array<NumberType, n_components> u_center = {2.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0, 1.0), Point<dim>(2.0, 1.0), Point<dim>(1.0, 0.0),
                                               Point<dim>(1.0, 2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Full gradient in both dimensions")
  {
    u_n = {{{1.0}, {3.0}, {0.0}, {4.0}}};
    NumberType reference_1 = 1.0;
    NumberType reference_2 = 2.0;
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference_1));
    CHECK(u_grad[0][1] == Catch::Approx(reference_2));
  }

  SECTION("Smaller slope selected per dimension")
  {
    u_n = {{{1.0}, {2.5}, {2.5}, {-0.5}}};
    NumberType reference = 0.5;
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[0][1] == Catch::Approx(-reference));
  }

  SECTION("Clipping in one dimension only")
  {
    u_n = {{{1.0}, {1.0}, {0.0}, {4.0}}};
    NumberType reference_1 = 0.0;
    NumberType reference_2 = 2.0;
    auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference_1));
    CHECK(u_grad[0][1] == Catch::Approx(reference_2));
  }
}

// ──────────────────────────────────────────────────────────────────────
// 2D, two components (new test)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation in 2D, two components", "[FV][reconstructor]")
{
  constexpr int dim = 2;
  constexpr int n_components = 2;
  constexpr int n_faces = 2 * dim;
  const Point<dim> x_center(1.0, 1.0);
  const std::array<NumberType, n_components> u_center = {2.0, 5.0};
  // neighbours: left, right, bottom, top
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0, 1.0), Point<dim>(2.0, 1.0), Point<dim>(1.0, 0.0),
                                               Point<dim>(1.0, 2.0)};

  SECTION("Independent gradients per component")
  {
    // component 0: left=1, right=3 => slopes +1,+1 => grad_x=1
    //              bottom=0, top=4 => slopes +2,+2 => grad_y=2
    // component 1: left=4, right=6 => slopes +1,+1 => grad_x=1
    //              bottom=3, top=7 => slopes +2,+2 => grad_y=2
    std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0, 4.0}, {3.0, 6.0}, {0.0, 3.0}, {4.0, 7.0}}};
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(1.0));
    CHECK(u_grad[0][1] == Catch::Approx(2.0));
    CHECK(u_grad[1][0] == Catch::Approx(1.0));
    CHECK(u_grad[1][1] == Catch::Approx(2.0));
  }

  SECTION("One component clipped, other has full gradient")
  {
    // component 0: left=3, right=3 => slopes -1,+1 => clipped to 0
    //              bottom=2, top=2 => slopes  0, 0 => 0
    // component 1: left=4, right=6 => slopes +1,+1 => grad_x=1
    //              bottom=4, top=6 => slopes +1,+1 => grad_y=1
    std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{3.0, 4.0}, {3.0, 6.0}, {2.0, 4.0}, {2.0, 6.0}}};
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(0.0));
    CHECK(u_grad[0][1] == Catch::Approx(0.0));
    CHECK(u_grad[1][0] == Catch::Approx(1.0));
    CHECK(u_grad[1][1] == Catch::Approx(1.0));
  }
}

// ──────────────────────────────────────────────────────────────────────
// Non-unit spacing (new test)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("TVD gradient computation with non-unit spacing", "[FV][reconstructor]")
{
  constexpr int dim = 1;
  constexpr int n_components = 1;
  constexpr int n_faces = 2;

  SECTION("dx = 0.5 on both sides, symmetric neighbours")
  {
    const Point<dim> x_center(1.0);
    const std::array<NumberType, n_components> u_center = {2.0};
    // neighbours at distance 0.5 on each side
    const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.5), Point<dim>(1.5)};
    // left slope = (1-2)/(0.5-1) = -1/(-0.5) = 2
    // right slope = (3-2)/(1.5-1) = 1/0.5 = 2
    // minmod(2,2) = 2
    std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0}, {3.0}}};
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(2.0));
  }

  SECTION("Asymmetric spacing - different slopes")
  {
    const Point<dim> x_center(1.0);
    const std::array<NumberType, n_components> u_center = {2.0};
    // left neighbour at 0.0 (dx=1), right neighbour at 1.25 (dx=0.25)
    const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(1.25)};
    // left slope = (1-2)/(0-1) = 1
    // right slope = (2.5-2)/(1.25-1) = 0.5/0.25 = 2
    // minmod(1,2) = 1
    std::array<std::array<NumberType, n_components>, n_faces> u_n = {{{1.0}, {2.5}}};
    const auto u_grad = Reconstructor::compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(1.0));
  }
}
