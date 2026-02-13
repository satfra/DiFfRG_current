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
    F_i[idxf("u")][0] = u * u / 2.0 + x[0];
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