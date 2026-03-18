#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/model/model.hh"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <autodiff/forward/real.hpp>
#include <cstddef>
#include <deal.II/base/numbers.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <petscvec.h>

using NumberType = double;
using VectorType = dealii::Vector<NumberType>;
using namespace dealii;
namespace KT = DiFfRG::FV::KurganovTadmor;
using KT::internal::compute_kt_flux_and_speeds;
using KT::internal::compute_numerical_flux;
using KT::internal::reconstruct_u;
using Reconstructor = DiFfRG::def::TVDReconstructor<DiFfRG::def::MinModLimiter, NumberType>;
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
    F_i[idxf("u")][0] = u[0] * u[0] / 2.0 + x[0];
  }
};

TEST_CASE("u_plus u_minus compoutation", "[FV][KT]")
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

TEST_CASE("reconstruct_u_derivative 1D single component", "[FV][KT]")
{
  // 1D uniform grid: center at x=0, left neighbor at x=-1, right neighbor at x=1
  // u_center=2, u_left=1, u_right=4
  // slopes: du_left = (1-2)/(-1) = 1, du_right = (4-2)/1 = 2
  // minmod(1, 2) = 1 (picks du_left since |1|<|2|), gradient = 1
  // Reconstruct at x_q = 0.5: u_recon = 2 + 1*0.5 = 2.5
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;

  const Point<dim> center(0.0);
  const Point<dim> x_q(0.5);
  const std::array<Point<dim>, 2> x_n = {Point<dim>(-1.0), Point<dim>(1.0)};

  SECTION("derivative w.r.t. center value")
  {
    // minmod picks du_left = (u_left-u_c)/(-1).
    // d(du_left)/d(u_c) = -1/(-1) = 1, so d(grad)/d(u_c) = 1
    // d(u_recon)/d(u_c) = 1 + 1 * 0.5 = 1.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.5));
  }

  SECTION("derivative w.r.t. left neighbor")
  {
    // d(du_left)/d(u_left) = 1/(-1) = -1, minmod picks du_left
    // d(grad)/d(u_left) = -1
    // d(u_recon)/d(u_left) = 0 + (-1) * 0.5 = -0.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(-0.5));
  }

  SECTION("derivative w.r.t. right neighbor (inactive in minmod)")
  {
    // minmod picks du_left (not du_right), so d(grad)/d(u_right) = 0
    // d(u_recon)/d(u_right) = 0 + 0 = 0
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[1][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[1][0]);
    CHECK(result[0] == Catch::Approx(0.0));
  }
}

TEST_CASE("reconstruct_u_derivative at local extremum (vanishing limiter)", "[FV][KT]")
{
  // Opposite slopes => minmod returns 0, gradient derivative also 0
  // center at x=0, u=3; left at x=-1, u=4; right at x=1, u=4
  // du_left = (4-3)/(-1) = -1, du_right = (4-3)/1 = 1
  // minmod(-1, 1) = 0 => flat reconstruction
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;

  const Point<dim> center(0.0);
  const Point<dim> x_q(0.5);
  const std::array<Point<dim>, 2> x_n = {Point<dim>(-1.0), Point<dim>(1.0)};

  SECTION("derivative w.r.t. center is Kronecker delta only")
  {
    std::array<AD, 1> u_c = {AD(3.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(4.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.0));
  }

  SECTION("derivative w.r.t. neighbor is zero")
  {
    std::array<AD, 1> u_c = {AD(3.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(4.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(0.0));
  }
}

TEST_CASE("reconstruct_u_derivative 2D single component", "[FV][KT]")
{
  // 2D: center at (0,0), u=2
  // x-neighbors: left (-1,0) u=1, right (1,0) u=4
  //   du_x_left = (1-2)/(-1)=1, du_x_right = (4-2)/1=2, minmod=1 (picks left)
  // y-neighbors: bottom (0,-1) u=0.5, top (0,1) u=5
  //   du_y_bottom = (0.5-2)/(-1)=1.5, du_y_top = (5-2)/1=3, minmod=1.5 (picks bottom)
  // gradient = (1, 1.5)
  constexpr int dim = 2;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;

  const Point<dim> center(0.0, 0.0);
  const Point<dim> x_q(0.5, 0.3);
  // neighbors ordered: x-left, x-right, y-left(bottom), y-right(top)
  const std::array<Point<dim>, 4> x_n = {Point<dim>(-1.0, 0.0), Point<dim>(1.0, 0.0), Point<dim>(0.0, -1.0),
                                         Point<dim>(0.0, 1.0)};

  SECTION("derivative w.r.t. center")
  {
    // d(grad_x)/d(u_c) = d(du_x_left)/d(u_c) = -1/(-1) = 1
    // d(grad_y)/d(u_c) = d(du_y_bottom)/d(u_c) = -1/(-1) = 1
    // grad_deriv = (1, 1)
    // d(u_recon)/d(u_c) = 1 + 1*0.5 + 1*0.3 = 1.8
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.8));
  }

  SECTION("derivative w.r.t. x-left neighbor")
  {
    // d(grad_x)/d(u_xleft) = 1/(-1) = -1 (minmod picks du_x_left)
    // d(grad_y)/d(u_xleft) = 0 (y-dimension independent)
    // d(u_recon)/d(u_xleft) = 0 + (-1)*0.5 + 0*0.3 = -0.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(-0.5));
  }

  SECTION("derivative w.r.t. y-bottom neighbor")
  {
    // d(grad_x)/d(u_ybottom) = 0
    // d(grad_y)/d(u_ybottom) = 1/(-1) = -1 (minmod picks du_y_bottom)
    // d(u_recon)/d(u_ybottom) = 0 + 0*0.5 + (-1)*0.3 = -0.3
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_n[2][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[2][0]);
    CHECK(result[0] == Catch::Approx(-0.3));
  }
}

TEST_CASE("Test compute_numerical_flux in 1D with one component", "[FV][KT]")
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

TEST_CASE("Test compute_numerical_flux in 2D with one component", "[FV][KT]")
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

TEST_CASE("Test compute_kt_flux_and_speeds with Burgers flux in 1D", "[FV][KT]")
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

// ---------------------------------------------------------------------------
// Tests for tag_cell_dofs
// ---------------------------------------------------------------------------

using KT::internal::make_tagged_neighbors;
using KT::internal::tag_cell_dofs;
template <int dim, typename NT, size_t nc> using KTCellData = KT::internal::CellData<dim, NT, nc>;
template <int dim, typename NT, size_t nc, size_t nf>
using KTNeighborData = KT::internal::NeighborData<dim, NT, nc, nf>;
using AD = autodiff::Real<1, NumberType>;

TEST_CASE("tag_cell_dofs 1D single component — matching dof is seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  const KTCellData<dim, NumberType, nc> cell_data{
      .x = Point<dim>(1.0),
      .u = {3.5},
      .dof_indices = {42},
  };

  const auto result = tag_cell_dofs(cell_data, 42);

  CHECK(result.x[0] == Catch::Approx(1.0));
  CHECK(val(result.u[0]) == Catch::Approx(3.5));
  CHECK(result.dof_indices[0] == 42);
  CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
}

TEST_CASE("tag_cell_dofs 1D single component — non-matching dof is not seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  const KTCellData<dim, NumberType, nc> cell_data{
      .x = Point<dim>(1.0),
      .u = {3.5},
      .dof_indices = {42},
  };

  const auto result = tag_cell_dofs(cell_data, 99);

  CHECK(val(result.u[0]) == Catch::Approx(3.5));
  CHECK(derivative(result.u[0]) == Catch::Approx(0.0));
}

TEST_CASE("tag_cell_dofs 1D multi-component — only matching component seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;

  const KTCellData<dim, NumberType, nc> cell_data{
      .x = Point<dim>(0.0),
      .u = {1.0, 2.0},
      .dof_indices = {10, 20},
  };

  SECTION("seed second component")
  {
    const auto result = tag_cell_dofs(cell_data, 20);

    CHECK(val(result.u[0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[0]) == Catch::Approx(0.0));
    CHECK(val(result.u[1]) == Catch::Approx(2.0));
    CHECK(derivative(result.u[1]) == Catch::Approx(1.0));
    CHECK(result.dof_indices[0] == 10);
    CHECK(result.dof_indices[1] == 20);
  }

  SECTION("seed first component")
  {
    const auto result = tag_cell_dofs(cell_data, 10);

    CHECK(val(result.u[0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
    CHECK(val(result.u[1]) == Catch::Approx(2.0));
    CHECK(derivative(result.u[1]) == Catch::Approx(0.0));
  }
}

TEST_CASE("tag_cell_dofs 2D single component — point preserved and seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 2;
  constexpr size_t nc = 1;

  const KTCellData<dim, NumberType, nc> cell_data{
      .x = Point<dim>(0.5, 0.7),
      .u = {4.0},
      .dof_indices = {5},
  };

  const auto result = tag_cell_dofs(cell_data, 5);

  CHECK(result.x[0] == Catch::Approx(0.5));
  CHECK(result.x[1] == Catch::Approx(0.7));
  CHECK(val(result.u[0]) == Catch::Approx(4.0));
  CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
  CHECK(result.dof_indices[0] == 5);
}

// ---------------------------------------------------------------------------
// Tests for make_tagged_neighbors
// ---------------------------------------------------------------------------

TEST_CASE("make_tagged_neighbors 1D single component — matching dof in one face", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  constexpr size_t nf = 2;

  const KTNeighborData<dim, NumberType, nc, nf> nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10},
                      std::array<dealii::types::global_dof_index, nc>{20}},
  };

  const auto result = make_tagged_neighbors(nd, 10);

  // Face 0 matches — seeded
  CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
  CHECK(derivative(result.u[0][0]) == Catch::Approx(1.0));

  // Face 1 does not match — not seeded
  CHECK(val(result.u[1][0]) == Catch::Approx(2.0));
  CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));

  // Positions preserved
  CHECK(result.x[0][0] == Catch::Approx(-1.0));
  CHECK(result.x[1][0] == Catch::Approx(1.0));

  // dof_indices preserved
  CHECK(result.dof_indices[0][0] == 10);
  CHECK(result.dof_indices[1][0] == 20);
}

TEST_CASE("make_tagged_neighbors 1D single component — no matching dof", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;
  constexpr size_t nf = 2;

  const KTNeighborData<dim, NumberType, nc, nf> nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10},
                      std::array<dealii::types::global_dof_index, nc>{20}},
  };

  const auto result = make_tagged_neighbors(nd, 99);

  for (size_t face = 0; face < nf; ++face)
    for (size_t c = 0; c < nc; ++c)
      CHECK(derivative(result.u[face][c]) == Catch::Approx(0.0));
}

TEST_CASE("make_tagged_neighbors 1D multi-component — specific face and component seeded",
          "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;
  constexpr size_t nf = 2;

  const KTNeighborData<dim, NumberType, nc, nf> nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0, 2.0}, std::array<NumberType, nc>{3.0, 4.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10, 20},
                      std::array<dealii::types::global_dof_index, nc>{30, 40}},
  };

  SECTION("tag dof 20 — face 0, component 1")
  {
    const auto result = make_tagged_neighbors(nd, 20);

    // Only face 0, component 1 should be seeded
    CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[0][1]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[1][1]) == Catch::Approx(0.0));

    // Values preserved
    CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
    CHECK(val(result.u[0][1]) == Catch::Approx(2.0));
    CHECK(val(result.u[1][0]) == Catch::Approx(3.0));
    CHECK(val(result.u[1][1]) == Catch::Approx(4.0));
  }

  SECTION("tag dof 30 — face 1, component 0")
  {
    const auto result = make_tagged_neighbors(nd, 30);

    CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[0][1]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[1][0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[1][1]) == Catch::Approx(0.0));
  }
}

TEST_CASE("make_tagged_neighbors 2D single component — 4 faces, one matching", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 2;
  constexpr size_t nc = 1;
  constexpr size_t nf = 4;

  const KTNeighborData<dim, NumberType, nc, nf> nd{
      .x = {Point<dim>(-1.0, 0.0), Point<dim>(1.0, 0.0), Point<dim>(0.0, -1.0), Point<dim>(0.0, 1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}, std::array<NumberType, nc>{3.0},
            std::array<NumberType, nc>{4.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{100},
                      std::array<dealii::types::global_dof_index, nc>{200},
                      std::array<dealii::types::global_dof_index, nc>{300},
                      std::array<dealii::types::global_dof_index, nc>{400}},
  };

  const auto result = make_tagged_neighbors(nd, 300);

  // Only face 2 should be seeded
  CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
  CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));
  CHECK(derivative(result.u[2][0]) == Catch::Approx(1.0));
  CHECK(derivative(result.u[3][0]) == Catch::Approx(0.0));

  // Values preserved
  CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
  CHECK(val(result.u[1][0]) == Catch::Approx(2.0));
  CHECK(val(result.u[2][0]) == Catch::Approx(3.0));
  CHECK(val(result.u[3][0]) == Catch::Approx(4.0));

  // Positions preserved
  CHECK(result.x[0][0] == Catch::Approx(-1.0));
  CHECK(result.x[0][1] == Catch::Approx(0.0));
  CHECK(result.x[2][0] == Catch::Approx(0.0));
  CHECK(result.x[2][1] == Catch::Approx(-1.0));

  // dof_indices preserved
  CHECK(result.dof_indices[2][0] == 300);
}