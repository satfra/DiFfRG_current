#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>

#include "DiFfRG/discretization/mesh/h_adaptivity.hh"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>
#include <DiFfRG/timestepping/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------
//
// These tests exercise the hybrid timesteppers, in which the implicit IDA controller
// drives the FEM block and a separate explicit stepper integrates the variable block
// "on demand". Three model families are exercised:
//
//   * ModelHybridRollback : sharp transition at t = 0.5 -> IDA rejects steps; this is the
//     regression test for the rejection-rollback bug (a rejected IDA step leaking into the
//     explicit buffer would corrupt v).
//
//   * ModelHybridSmooth   : du/dt = -K u (no rejections) with explicit.dt set so that each
//     IDA trial step contains only one or two explicit substeps. This is the regime in
//     which a typical FRG production run operates and the regime in which a per-trial-step
//     reset() of the multistep stepper would be most visible.
//
//   * ModelHybridTwoWay   : the FEM source reads the explicit variable v through an
//     extractor, closing the coupling loop (production FRG models do exactly this through
//     wave-function-renormalisation extractors). If serving v back to IDA across Newton
//     iterations is not consistent with the spatial trial state, Newton's convergence
//     breaks down and IDA reports recoverable residual errors.

template <typename Model, typename TimeStepper>
bool run_hybrid(const std::string &test_name, double expected_precision, double cur_dt = 1e-3,
                double final_time = 1.0, double output_dt = 5e-2, double impl_max_dt = 1e-1,
                double impl_rel_tol = 1e-6)
{
  using namespace dealii;
  constexpr uint dim = 1;
  using Discretization = CG::Discretization<typename Model::Components, double, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = CG::Assembler<Discretization, Model>;

  JSONValue json = json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 3},
         {"threads", 4},
         {"batch_size", 64},
         {"overintegration", 0},
         {"output_subdivisions", 2},
         {"EoM_abs_tol", 1e-12},
         {"EoM_max_iter", 100},
         {"grid", {{"x_grid", "0:0.05:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
         {"adaptivity",
          {{"start_adapt_at", 0.},
           {"adapt_dt", 1e-1},
           {"level", 0},
           {"refine_percent", 1e-1},
           {"coarsen_percent", 5e-2}}}}},
       {"timestepping",
        {{"final_time", final_time},
         {"output_dt", output_dt},
         {"explicit",
          {{"dt", cur_dt}, {"minimal_dt", 1e-7}, {"maximal_dt", 1e-2}, {"abs_tol", 1e-13}, {"rel_tol", 1e-10}}},
         {"implicit",
          {{"dt", 1e-3}, {"minimal_dt", 1e-7}, {"maximal_dt", impl_max_dt}, {"abs_tol", 1e-10},
           {"rel_tol", impl_rel_tol}}}}},
       {"output", {{"verbosity", 0}, {"vtk", false}}}});

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
    // logger already set up
  }

  Testing::PhysicalParameters p_prm = {};
  Model model(p_prm);
  RectangularMesh<dim> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);
  DataOutput<dim, VectorType> data_out("./", test_name, test_name + '/', json);
  HAdaptivity mesh_adaptor(assembler, json);
  TimeStepper time_stepper(json, &assembler, &data_out, &mesh_adaptor);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  try {
    time_stepper.run(&initial_condition, 0., final_time);
  } catch (std::exception &e) {
    std::cout << test_name << ": simulation finished with exception " << e.what() << std::endl;
    return false;
  }

  model.set_time(final_time);
  bool valid = true;

  // block 0: FEM solution
  const auto &support_points = discretization.get_support_points();
  for (uint i = 0; i < support_points.size(); ++i) {
    const double is = initial_condition.data().block(0)[i];
    const double should = model.solution(support_points[i]);
    if (!is_close(is, should, expected_precision))
      std::cout << test_name << " FEM u: is " << is << " should be " << should
                << " (rel. error " << std::abs(is - should) / std::max(std::abs(should), 1e-30) << ")"
                << std::endl;
    valid &= is_close(is, should, expected_precision);
  }

  // block 1: explicit variable
  const double v_is = initial_condition.data().block(1)[0];
  const double v_should = model.variable_solution();
  if (!is_close(v_is, v_should, expected_precision))
    std::cout << test_name << " explicit variable v: is " << v_is << " should be " << v_should
              << " (rel. error " << std::abs(v_is - v_should) / std::max(std::abs(v_should), 1e-30) << ")"
              << std::endl;
  valid &= is_close(v_is, v_should, expected_precision);

  if (!valid) std::cerr << "Failed " << test_name << std::endl;
  return valid;
}

//--------------------------------------------
// Original regression tests: rejection-rollback on a sharp logistic FEM
//--------------------------------------------

TEST_CASE("Test SUNDIALS IDA + Boost ABM hybrid stepper", "[timestepping][sundials_ida_boost][abm]")
{
  using Model = Testing::ModelHybridRollback<1>;
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, 1, UMFPack>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_sundials_ida_boost_abm", 1e-3));
}

TEST_CASE("Test SUNDIALS IDA + Boost RK hybrid stepper", "[timestepping][sundials_ida_boost][rk]")
{
  using Model = Testing::ModelHybridRollback<1>;
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostRK<VectorType, SparseMatrixType, 1, UMFPack, 0>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_sundials_ida_boost_rk", 1e-3));
}

//--------------------------------------------
// Diagnostic: smooth problem, ~1 explicit substep per IDA trial step.
//
// FRG production runs typically have explicit.dt of the same order as IDA's accepted
// step, so each request_variables call appends ~1 ABM substep. The integration must
// still match the analytic solution to a sensible precision -- if the per-trial-step
// machinery (rebuild on commit, multistep history reset, spatial interpolation, ...)
// degrades the explicit accuracy on accepted steps, this is where it shows up.
//--------------------------------------------

TEST_CASE("Hybrid ABM stepper retains accuracy on smooth problem at ~1 substep per IDA step",
          "[timestepping][sundials_ida_boost][abm][diag]")
{
  using Model = Testing::ModelHybridSmooth<1>;
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, 1, UMFPack>;
  // explicit.dt = output_dt so each IDA trial step usually triggers exactly one ABM substep
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_abm_smooth", /*tol=*/1e-3, /*cur_dt=*/5e-2,
                                          /*final_time=*/1.0, /*output_dt=*/5e-2,
                                          /*impl_max_dt=*/5e-2));
}

TEST_CASE("Hybrid ABM cur_dt scan on smooth problem", "[timestepping][sundials_ida_boost][abm][scan]")
{
  using Model = Testing::ModelHybridSmooth<1>;
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, 1, UMFPack>;
  // Sweep cur_dt while holding output_dt and IDA max_dt fixed; the test prints the
  // observed v error so the scaling can be read off the log. We do not REQUIRE pass --
  // this is a diagnostic scan, not a regression gate.
  for (double cur_dt : {5e-2, 2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3}) {
    const std::string tag = "test_scan_abm_dt_smooth_" + std::to_string(cur_dt);
    run_hybrid<Model, TimeStepper>(tag, /*tol=*/1e-30, /*cur_dt=*/cur_dt,
                                    /*final_time=*/1.0, /*output_dt=*/5e-2,
                                    /*impl_max_dt=*/5e-2);
  }
}

TEST_CASE("Hybrid ABM cur_dt scan on two-way coupled problem",
          "[timestepping][sundials_ida_boost][abm][scan]")
{
  using Model = Testing::ModelHybridTwoWay<1>;
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, 1, UMFPack>;
  for (double cur_dt : {2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3}) {
    const std::string tag = "test_scan_abm_dt_twoway_" + std::to_string(cur_dt);
    run_hybrid<Model, TimeStepper>(tag, /*tol=*/1e-30, /*cur_dt=*/cur_dt,
                                    /*final_time=*/1.0, /*output_dt=*/5e-2,
                                    /*impl_max_dt=*/5e-2);
  }
}

TEST_CASE("Hybrid RK stepper retains accuracy on smooth problem at ~1 substep per IDA step",
          "[timestepping][sundials_ida_boost][rk][diag]")
{
  using Model = Testing::ModelHybridSmooth<1>;
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostRK<VectorType, SparseMatrixType, 1, UMFPack, 0>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_rk_smooth", /*tol=*/1e-3, /*cur_dt=*/5e-2,
                                          /*final_time=*/1.0, /*output_dt=*/5e-2,
                                          /*impl_max_dt=*/5e-2));
}

//--------------------------------------------
// Diagnostic: two-way coupled FEM <-> variable.
//
// The FEM source reads v through an extractor and v's residual reads u, mirroring how a
// production FRG model couples back to wave-function renormalisations. If the variable
// served back to IDA across Newton iterations is built from speculative, unconverged
// spatial trial values, Newton convergence will fail and IDA will throw "recoverable
// residual errors" -- which is exactly what the QM_LPAp run does against the current fix.
//--------------------------------------------

TEST_CASE("Hybrid ABM stepper handles two-way FEM <-> variable coupling",
          "[timestepping][sundials_ida_boost][abm][diag]")
{
  using Model = Testing::ModelHybridTwoWay<1>;
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, 1, UMFPack>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_abm_twoway", /*tol=*/1e-3, /*cur_dt=*/2e-2,
                                          /*final_time=*/1.0, /*output_dt=*/5e-2,
                                          /*impl_max_dt=*/5e-2));
}

TEST_CASE("Hybrid RK stepper handles two-way FEM <-> variable coupling",
          "[timestepping][sundials_ida_boost][rk][diag]")
{
  using Model = Testing::ModelHybridTwoWay<1>;
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostRK<VectorType, SparseMatrixType, 1, UMFPack, 0>;
  REQUIRE(run_hybrid<Model, TimeStepper>("test_diag_rk_twoway", /*tol=*/1e-3, /*cur_dt=*/2e-2,
                                          /*final_time=*/1.0, /*output_dt=*/5e-2,
                                          /*impl_max_dt=*/5e-2));
}
