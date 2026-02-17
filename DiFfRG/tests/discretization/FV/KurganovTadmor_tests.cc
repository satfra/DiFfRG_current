#include "DiFfRG/common/tuples.hh"
#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/discretization/mesh/configuration_mesh.hh"
#include "DiFfRG/model/model.hh"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <algorithm>
#include <autodiff/forward/real.hpp>
#include <cstddef>
#include <deal.II/base/numbers.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <petscvec.h>
#include <tuple>
#include <vector>

using NumberType = double;
using VectorType = dealii::Vector<NumberType>;
using namespace dealii;
namespace KT = DiFfRG::FV::KurganovTadmor;
using KT::internal::compute_gradient;
<<<<<<< HEAD
using KT::internal::compute_kt_flux_and_speeds;
using KT::internal::compute_numerical_flux;
=======
>>>>>>> 912f39fb4f2f4d0f4b7a3952db32fadb3f032cf4
using KT::internal::reconstruct_u;
using KT::internal::ScratchData;
struct CopyData {
};

using FEFunctionDesc = DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u">>;
using Components = DiFfRG::ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

class TestModel : public DiFfRG::def::AbstractModel<TestModel, Components>
{
public:
  template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
  static void
  KurganovTadmor_advection_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                                [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol)
  {
    auto u = get<"fe_functions">(sol);
<<<<<<< HEAD
    F_i[idxf("u")][0] = u[0] * u[0] / 2.0 + x[0];
=======
    F_i[idxf("u")][0] = u * u / 2.0 + x[0];
>>>>>>> 912f39fb4f2f4d0f4b7a3952db32fadb3f032cf4
  }
};

TEST_CASE("u_plus u_minus compoutation", "[KT]")
{
  const int dim = 1;
  const uint n_components = 2;
  using GradComponentType = dealii::Tensor<1, dim, NumberType>;

  const DiFfRG::FV::KurganovTadmor::internal::GradientType<dim, NumberType, n_components> u_grad(
      {GradComponentType({-0.5}), GradComponentType({0.7})});
  const Point<dim> x_center(1.0);
  const Point<dim> x_q(2.0);
  const std::array<NumberType, n_components> u_val = {1.0, 2.0};
  const std::array<NumberType, n_components> u_minus_reference({0.5, 2.7});

  const std::array<NumberType, n_components> u_reconstructed = reconstruct_u(u_val, x_center, x_q, u_grad);
  CHECK(u_minus_reference[0] == Catch::Approx(u_reconstructed[0]));
  CHECK(u_minus_reference[1] == Catch::Approx(u_reconstructed[1]));
}

TEST_CASE("Test Gradient computation in 1D")
{
  const int dim = 1;
  constexpr int n_components = 1;
  constexpr int n_faces = 2 * dim;
  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {2.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Check Normal Derivative Computation")
  {
    u_n = {{{1.0}, {3.0}}};
    NumberType reference = 1.0;
    const auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }

  SECTION("Check choice of smaller gradient to neighboring cells")
  {
    u_n = {{{1.0}, {2.5}}};
    NumberType reference = 0.5;
    const auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }

  SECTION("Check clipping")
  {
    u_n = {{{1.0}, {1.0}}};
    NumberType reference = 0.0;
    auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));

    u_n = {{{5.0}, {4.0}}};
    u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
  }
}

TEST_CASE("Test Gradient computation in 1D with two components")
{
  const int dim = 1;
  constexpr int n_components = 2;
  constexpr int n_faces = 2 * dim;
  const Point<dim> x_center(1.0);
  const std::array<NumberType, n_components> u_center = {2.0, 3.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0), Point<dim>(2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Check Normal Derivative Computation")
  {
    u_n = {{{1.0, 2.0}, {3.0, 4.0}}};
    NumberType reference = 1.0;
    const auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }

  SECTION("Check choice of smaller gradient to neighboring cells")
  {
    u_n = {{{1.0, 2.0}, {2.5, 3.5}}};
    NumberType reference = 0.5;
    const auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }

  SECTION("Check clipping")
  {
    u_n = {{{1.0, 2.0}, {1.0, 2.0}}};
    NumberType reference = 0.0;
    auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));

    u_n = {{{5.0, 1.0}, {4.0, 0.0}}};
    u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[1][0] == Catch::Approx(reference));
  }
}

TEST_CASE("Test Gradient computation in 2D")
{
  constexpr int n_components = 1;
  constexpr int dim = 2;
  constexpr int n_faces = 2 * dim;
  const Point<dim> x_center(1.0, 1.0);
  const std::array<NumberType, n_components> u_center = {2.0};
  const std::array<Point<dim>, n_faces> x_n = {Point<dim>(0.0, 1.0), Point<dim>(2.0, 1.0), Point<dim>(1.0, 0.0),
                                               Point<dim>(1.0, 2.0)};
  std::array<std::array<NumberType, n_components>, n_faces> u_n;

  SECTION("Check Normal Derivative Computation")
  {
    u_n = {{{1.0}, {3.0}, {0.0}, {4.0}}};
    NumberType reference_1 = 1.0;
    NumberType reference_2 = 2.0;

    const auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference_1));
    CHECK(u_grad[0][1] == Catch::Approx(reference_2));
  }

  SECTION("Check choice of smaller gradient to neighboring cells")
  {
    u_n = {{{1.0}, {2.5}, {2.5}, {-0.5}}};
    NumberType reference = 0.5;
    const auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference));
    CHECK(u_grad[0][1] == Catch::Approx(-reference));
  }

  SECTION("Check clipping")
  {
    u_n = {{{1.0}, {1.0}, {0.0}, {4.0}}};
    NumberType reference_1 = 0.0;
    NumberType reference_2 = 2.0;
    auto u_grad = compute_gradient<NumberType, dim, n_components>(x_center, u_center, x_n, u_n);
    CHECK(u_grad[0][0] == Catch::Approx(reference_1));
    CHECK(u_grad[0][1] == Catch::Approx(reference_2));
  }
}

TEST_CASE("Test compute_numerical_flux in 1D with one component", "[KT]")
{
  constexpr int dim = 1;
  constexpr size_t n_components = 1;

  SECTION("Symmetric flux cancels diffusion term")
  {
    // u_plus == u_minus => H = F (no diffusion)
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({3.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({3.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({5.0})};
    std::array<NumberType, n_components> u_plus = {1.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(3.0));
  }

  SECTION("General case")
  {
    // F_plus = 4.0, F_minus = 2.0, a = 1.0, u_plus = 3.0, u_minus = 1.0
    // H = (4+2)/2 - 1*(3-1)/2 = 3 - 1 = 2
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({4.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({1.0})};
    std::array<NumberType, n_components> u_plus = {3.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(2.0));
  }

  SECTION("Zero wave speed reduces to central flux")
  {
    // a = 0 => H = (F_plus + F_minus)/2
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({6.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({0.0})};
    std::array<NumberType, n_components> u_plus = {5.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(4.0));
  }
}

TEST_CASE("Test compute_numerical_flux in 2D with one component", "[KT]")
{
  constexpr int dim = 2;
  constexpr size_t n_components = 1;

  // F_plus = (4, 1), F_minus = (2, 3), a = (1, 2), u_plus = 3, u_minus = 1
  // H[0] = (4+2)/2 - 1*(3-1)/2 = 3 - 1 = 2
  // H[1] = (1+3)/2 - 2*(3-1)/2 = 2 - 2 = 0
  std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({4.0, 1.0})};
  std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0, 3.0})};
  std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({1.0, 2.0})};
  std::array<NumberType, n_components> u_plus = {3.0};
  std::array<NumberType, n_components> u_minus = {1.0};

  const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
  CHECK(H[0][0] == Catch::Approx(2.0));
  CHECK(H[0][1] == Catch::Approx(0.0));
}

TEST_CASE("Test compute_kt_flux_and_speeds with Burgers flux in 1D", "[KT]")
{
  // TestModel: F(u) = u^2/2 + x, so dF/du = u
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  TestModel model;

  SECTION("Flux values and wave speed")
  {
    // At x=0.5, u_plus=2.0, u_minus=1.0
    // F(u_plus)  = 2^2/2 + 0.5 = 2.5
    // F(u_minus) = 1^2/2 + 0.5 = 1.0
    // dF/du(u_plus) = 2.0, dF/du(u_minus) = 1.0
    // a = max(|2|, |1|) = 2.0
    const Point<dim> x_q(0.5);
    const std::array<NumberType, n_components> u_plus = {2.0};
    const std::array<NumberType, n_components> u_minus = {1.0};

    const auto [F_plus, F_minus, a_half] = compute_kt_flux_and_speeds(u_plus, u_minus, x_q, model);
    CHECK(F_plus[0][0] == Catch::Approx(2.5));
    CHECK(F_minus[0][0] == Catch::Approx(1.0));
    CHECK(a_half[0][0] == Catch::Approx(2.0));
  }

  SECTION("Symmetric states give symmetric fluxes")
  {
    // u_plus = u_minus = 3.0, x=1.0
    // F = 3^2/2 + 1 = 5.5
    // a = |3| = 3.0
    const Point<dim> x_q(1.0);
    const std::array<NumberType, n_components> u_plus = {3.0};
    const std::array<NumberType, n_components> u_minus = {3.0};

    const auto [F_plus, F_minus, a_half] = compute_kt_flux_and_speeds(u_plus, u_minus, x_q, model);
    CHECK(F_plus[0][0] == Catch::Approx(5.5));
    CHECK(F_minus[0][0] == Catch::Approx(5.5));
    CHECK(a_half[0][0] == Catch::Approx(3.0));
  }

  SECTION("Negative velocities")
  {
    // u_plus = -1.0, u_minus = -3.0, x=0
    // F(u_plus)  = (-1)^2/2 + 0 = 0.5
    // F(u_minus) = (-3)^2/2 + 0 = 4.5
    // dF/du(u_plus) = -1, dF/du(u_minus) = -3
    // a = max(|-1|, |-3|) = 3.0
    const Point<dim> x_q(0.0);
    const std::array<NumberType, n_components> u_plus = {-1.0};
    const std::array<NumberType, n_components> u_minus = {-3.0};

    const auto [F_plus, F_minus, a_half] = compute_kt_flux_and_speeds(u_plus, u_minus, x_q, model);
    CHECK(F_plus[0][0] == Catch::Approx(0.5));
    CHECK(F_minus[0][0] == Catch::Approx(4.5));
    CHECK(a_half[0][0] == Catch::Approx(3.0));
  }

  SECTION("Full KT numerical flux for Burgers")
  {
    // u_plus=2, u_minus=1, x=0.5
    // F_plus=2.5, F_minus=1.0, a=2.0
    // H = (2.5+1.0)/2 - 2.0*(2.0-1.0)/2 = 1.75 - 1.0 = 0.75
    const Point<dim> x_q(0.5);
    const std::array<NumberType, n_components> u_plus = {2.0};
    const std::array<NumberType, n_components> u_minus = {1.0};

    const auto [F_plus, F_minus, a_half] = compute_kt_flux_and_speeds(u_plus, u_minus, x_q, model);
    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(0.75));
  }
}