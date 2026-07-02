#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <boilerplate/kt_models.hh>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cmath>

using namespace DiFfRG;
using namespace dealii;

// Helper: set up a minimal JSON config for the FV assembler and mesh
static JSONValue make_json()
{
  return json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 0},
         {"threads", 1},
         {"batch_size", 64},
         {"overintegration", 0},
         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},
         {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}}}},
       {"output", {{"verbosity", 0}}}});
}

// Helper: ensure the spdlog "log" sink exists (required by FV Discretization)
static void ensure_logger()
{
  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
  }
}

class SourceOnlyModel
    : public def::AbstractModel<SourceOnlyModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<SourceOnlyModel>,
      public def::FlowBoundaries<SourceOnlyModel>,
      public def::FVDefaultBoundaries<SourceOnlyModel>,
      public def::AD<SourceOnlyModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.25 * pos[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const { return {1.0 + 0.25 * pos[0]}; }

  template <typename NT, typename Solution>
  void KurganovTadmor_advection_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/,
                                     const Solution & /*sol*/) const
  {
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> &pos, const Solution & /*sol*/) const
  {
    s_i[0] = 0.5 + pos[0];
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                       [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }
};

class ThirdDerivativeDiffusionModel
    : public def::AbstractModel<ThirdDerivativeDiffusionModel, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
      public def::Time,
      public def::LLFFlux<ThirdDerivativeDiffusionModel>,
      public def::FlowBoundaries<ThirdDerivativeDiffusionModel>,
      public def::FVDefaultBoundaries<ThirdDerivativeDiffusionModel>,
      public def::AD<ThirdDerivativeDiffusionModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = solution(pos)[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const
  {
    const double x = pos[0];
    return {1.0 + 0.2 * x - 0.1 * x * x + 0.5 * x * x * x};
  }

  template <typename NT, typename Solution>
  void KurganovTadmor_advection_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/,
                                     const Solution & /*sol*/) const
  {
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &third_derivatives = get<"fe_third_derivatives">(sol);
    F_i[0][0] = NT(0.05) * third_derivatives[0][0][0][0];
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> & /*pos*/, const Solution & /*sol*/) const
  {
    s_i[0] = 0.0;
  }

  template <typename NT, typename Vector, typename VectorDot>
  void mass(std::array<NT, 1> &m_i, const Point<1> & /*pos*/, const Vector & /*u*/, const VectorDot & /*dt_u*/) const
  {
    m_i[0] = 0.0;
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                       [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }
};

/**
 * Compares the analytic Jacobian from the KT assembler against a central-difference
 * finite-difference Jacobian of residual(). Detects missing terms in jacobian() —
 * in particular the diffusion flux Jacobian which is currently not implemented.
 *
 * With ModelBurgersTravelingWaveKT (nu=100, dx=0.1): missing diffusion entries differ
 * by O(nu/dx) = O(1000), far above the relative tolerance of 1e-4.
 */
TEST_CASE("KT Jacobian matches FD Jacobian for traveling wave model (detects missing diffusion)", "[FV][KT]")
{
  using Model = Testing::ModelBurgersTravelingWaveKT<1>;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<1>>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  const JSONValue json = make_json();

  Model model(p_prm);
  RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());

  VectorType sol_dot(n_dofs); // zero — excluded via weight_mass=0 / alpha=0

  // --- Analytic Jacobian (spatial part only: weight=1, alpha=0, beta=0) ---
  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, /*alpha=*/0.0, /*beta=*/0.0);

  // --- Finite-difference Jacobian via central differences of residual() ---
  // residual(r, u, weight=1, sol_dot, weight_mass=0) gives just the spatial flux/source residual
  const double eps = 1e-8;
  std::vector<std::vector<double>> J_fd(n_dofs, std::vector<double>(n_dofs, 0.0));

  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;

    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, /*weight_mass=*/0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, /*weight_mass=*/0.0);

    for (int i = 0; i < n_dofs; ++i)
      J_fd[i][j] = (r_p[i] - r_m[i]) / (2.0 * eps);
  }

  const int boundary_layer = 3;
  const double tol = 1e-4;
  bool pass = true;
  for (int i = 0; i < n_dofs; ++i) {
    for (int j = 0; j < n_dofs; ++j) {
      // if (j < boundary_layer || j >= n_dofs - boundary_layer) continue;

      const double analytic = J_analytic.el(i, j);
      const double fd = J_fd[i][j];
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > tol * scale) {
        std::cout << "Jacobian mismatch at [" << i << "," << j << "]: "
                  << "analytic=" << analytic << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

/**
 * Sanity check: the same FD consistency test for the pure-advection Burgers model
 * (no diffusion flux). The current Jacobian should already pass this.
 */
TEST_CASE("KT Jacobian matches FD Jacobian for pure advection Burgers model", "[FV][KT]")
{
  using Model = Testing::ModelBurgersKT<1>;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<1>>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = 0.0;
  p_prm.initial_x1[0] = 1.0;
  const JSONValue json = make_json();

  Model model(p_prm);
  RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());

  VectorType sol_dot(n_dofs);

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const double eps = 1e-8;
  std::vector<std::vector<double>> J_fd(n_dofs, std::vector<double>(n_dofs, 0.0));
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;
    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, 0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, 0.0);
    for (int i = 0; i < n_dofs; ++i)
      J_fd[i][j] = (r_p[i] - r_m[i]) / (2.0 * eps);
  }

  const double tol = 1e-4;
  bool pass = true;
  for (int i = 0; i < n_dofs; ++i) {
    for (int j = 0; j < n_dofs; ++j) {

      const double analytic = J_analytic.el(i, j);
      const double fd = J_fd[i][j];
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > tol * scale) {
        std::cout << "Jacobian mismatch at [" << i << "," << j << "]: "
                  << "analytic=" << analytic << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

TEST_CASE("KT Jacobian matches FD Jacobian for third-derivative diffusion model on interior rows", "[FV][KT]")
{
  using Model = ThirdDerivativeDiffusionModel;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<1>>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  const JSONValue json = make_json();

  Model model;
  RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 8);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 0.7 + 0.11 * static_cast<double>(i) - 0.015 * static_cast<double>(i * i) +
             0.001 * static_cast<double>(i * i * i);

  VectorType sol_dot(n_dofs);

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> J_analytic(sp);
  assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

  const double eps = 1e-8;
  std::vector<std::vector<double>> J_fd(n_dofs, std::vector<double>(n_dofs, 0.0));
  for (int j = 0; j < n_dofs; ++j) {
    VectorType u_p = sol, u_m = sol;
    u_p[j] += eps;
    u_m[j] -= eps;
    VectorType r_p(n_dofs), r_m(n_dofs);
    assembler.residual(r_p, u_p, 1.0, sol_dot, 0.0);
    assembler.residual(r_m, u_m, 1.0, sol_dot, 0.0);
    for (int i = 0; i < n_dofs; ++i)
      J_fd[i][j] = (r_p[i] - r_m[i]) / (2.0 * eps);
  }

  const double tol = 2e-4;
  bool pass = true;
  for (int i = 3; i < n_dofs - 3; ++i) {
    for (int j = 0; j < n_dofs; ++j) {
      const double analytic = J_analytic.el(i, j);
      const double fd = J_fd[i][j];
      const double err = std::abs(analytic - fd);
      const double scale = std::max(1.0, std::abs(fd));
      if (err > tol * scale) {
        std::cout << "Third-derivative Jacobian mismatch at [" << i << "," << j << "]: "
                  << "analytic=" << analytic << "  fd=" << fd << "  rel_err=" << err / scale << "\n";
        pass = false;
      }
    }
  }
  REQUIRE(pass);
}

TEST_CASE("KT Jacobian for x-dependent source-only model is diagonal", "[FV][KT]")
{
  using Model = SourceOnlyModel;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<1>>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  const JSONValue json = make_json();

  Model model;
  RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());

  VectorType sol_dot(n_dofs);
  VectorType residual(n_dofs);
  assembler.residual(residual, sol, 1.0, sol_dot, 0.0);

  bool has_nonzero_source_residual = false;
  for (int i = 0; i < n_dofs; ++i)
    has_nonzero_source_residual = has_nonzero_source_residual || std::abs(residual[i]) > 1e-12;
  REQUIRE(has_nonzero_source_residual);

  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> jacobian(sp);
  assembler.jacobian(jacobian, sol, 1.0, sol_dot, 0.0, 0.0);

  for (int i = 0; i < n_dofs; ++i)
    for (int j = 0; j < n_dofs; ++j)
      CHECK(jacobian.el(i, j) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("KT first-order Jacobian strategy does not use reconstruction-neighbor columns", "[FV][KT]")
{
  using Model = Testing::ModelBurgersKT<1>;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<1>>;
  using ResidualReconstructor =
      def::TVDReconstructor<Discretization::dim, def::MinModLimiter, NumberType>;
  using JacobianReconstructor = def::FirstOrderReconstructor<Discretization::dim, NumberType>;
  using ExactAssembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using ApproxJacobianAssembler =
      FV::KurganovTadmor::Assembler<Discretization, Model, ResidualReconstructor,
                                    FV::KurganovTadmor::MaxEigenvalueWaveSpeed, JacobianReconstructor>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = 0.0;
  p_prm.initial_x1[0] = 1.0;
  const JSONValue json = make_json();

  Model model(p_prm);
  RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  ExactAssembler exact_assembler(discretization, model, json);
  ApproxJacobianAssembler approx_assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 5);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.1 * static_cast<double>(i) + 0.03 * static_cast<double>(i * i);

  VectorType sol_dot(n_dofs);
  VectorType residual_exact(n_dofs);
  VectorType residual_approx(n_dofs);
  exact_assembler.residual(residual_exact, sol, 1.0, sol_dot, 0.0);
  approx_assembler.residual(residual_approx, sol, 1.0, sol_dot, 0.0);

  for (int i = 0; i < n_dofs; ++i)
    CHECK(residual_approx[i] == Catch::Approx(residual_exact[i]).margin(1e-12));

  const SparsityPattern &sp = approx_assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> jacobian(sp);
  approx_assembler.jacobian(jacobian, sol, 1.0, sol_dot, 0.0, 0.0);

  const int row = n_dofs / 2;
  REQUIRE(row >= 2);
  REQUIRE(row + 2 < n_dofs);

  CHECK_FALSE(sp.exists(row, row - 2));
  CHECK_FALSE(sp.exists(row, row + 2));
  CHECK(jacobian.el(row, row - 2) == Catch::Approx(0.0).margin(1e-12));
  CHECK(jacobian.el(row, row + 2) == Catch::Approx(0.0).margin(1e-12));

  const bool has_immediate_neighbor_contribution =
      (sp.exists(row, row - 1) && std::abs(jacobian.el(row, row - 1)) > 1e-12) ||
      (sp.exists(row, row + 1) && std::abs(jacobian.el(row, row + 1)) > 1e-12);
  CHECK(has_immediate_neighbor_contribution);
}

TEST_CASE("KT TVD Jacobian strategy keeps reconstruction-neighbor columns", "[FV][KT]")
{
  using Model = Testing::ModelBurgersKT<1>;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<1>>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = 0.0;
  p_prm.initial_x1[0] = 1.0;
  const JSONValue json = make_json();

  Model model(p_prm);
  RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  VectorType sol = state.spatial_data();
  const int n_dofs = static_cast<int>(sol.size());
  REQUIRE(n_dofs > 5);

  for (int i = 0; i < n_dofs; ++i)
    sol[i] = 1.0 + 0.2 * static_cast<double>(i) - 0.01 * static_cast<double>(i * i);

  VectorType sol_dot(n_dofs);
  const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
  SparseMatrix<NumberType> jacobian(sp);
  assembler.jacobian(jacobian, sol, 1.0, sol_dot, 0.0, 0.0);

  const int row = n_dofs / 2;
  REQUIRE(row >= 2);
  REQUIRE(row + 2 < n_dofs);

  CHECK(sp.exists(row, row - 2));
  CHECK(sp.exists(row, row + 2));

  const bool has_reconstruction_neighbor_contribution =
      std::abs(jacobian.el(row, row - 2)) > 1e-12 || std::abs(jacobian.el(row, row + 2)) > 1e-12;
  CHECK(has_reconstruction_neighbor_contribution);
}
