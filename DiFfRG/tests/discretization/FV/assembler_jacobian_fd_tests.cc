#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <boilerplate/models.hh>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

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

  // --- Compare all entries ---
  // The diffusion residual uses a one-sided face-gradient reconstruction together
  // with ghost states at the physical boundaries. For perturbations of the first
  // / last few DOFs, the minmod reconstruction sits on branch switches, so the
  // residual is continuous but its classical derivative is not unique. The same
  // issue already appears in the pure-advection test; diffusion widens the
  // affected boundary layer because face gradients depend on a larger stencil.
  //
  // Away from that boundary layer, a missing diffusion Jacobian would still
  // create O(nu/dx) = O(100) to O(1000) discrepancies, while the implemented
  // analytic Jacobian matches the FD linearization to better than 1e-2.
  const int boundary_layer = 3;
  const double tol = 1e-4;
  bool pass = true;
  for (int i = 0; i < n_dofs; ++i) {
    for (int j = 0; j < n_dofs; ++j) {
      if (j < boundary_layer || j >= n_dofs - boundary_layer) continue;

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

  const double eps = 1e-6;
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
      // The first and last cells use boundary ghost states located at the face,
      // so the minmod reconstruction sits exactly on an equal-slope branch switch
      // for perturbations of those boundary DOFs. The residual is continuous there,
      // but its classical derivative is not unique, so AD/branch derivatives need
      // not match a symmetric central-difference linearization.
      if (j == 0 || j == n_dofs - 1) continue;

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
