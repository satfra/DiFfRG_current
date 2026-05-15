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
// "on demand". The model ModelHybridRollback has a FEM solution with a sharp transition
// (so IDA rejects and retries steps) and an explicit variable coupled to that FEM
// solution with a known exact answer. If a rejected IDA step leaks into the explicit
// variable buffer, the variable is integrated along a trajectory IDA never accepted and
// the final value is wrong -- which is what these tests catch.

template <typename TimeStepper> bool run_hybrid(const std::string &test_name, double expected_precision)
{
  using namespace dealii;
  using Model = Testing::ModelHybridRollback<1>;
  constexpr uint dim = 1;
  using Discretization = CG::Discretization<typename Model::Components, double, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = CG::Assembler<Discretization, Model>;

  // A sharp FEM transition at t = 0.5 forces IDA to reject and retry time steps; the
  // implicit tolerances are deliberately loose enough that IDA takes sizeable steps in
  // the flat region and then gets caught out at the transition.
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
        {{"final_time", 1.},
         {"output_dt", 5e-2},
         {"explicit",
          {{"dt", 1e-3}, {"minimal_dt", 1e-7}, {"maximal_dt", 1e-2}, {"abs_tol", 1e-13}, {"rel_tol", 1e-10}}},
         {"implicit",
          {{"dt", 1e-3}, {"minimal_dt", 1e-7}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-10}, {"rel_tol", 1e-6}}}}},
       {"output", {{"verbosity", 0}}}});

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
    // logger already set up
  }

  const double final_time = json.get_double("/timestepping/final_time");

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
    std::cout << "Simulation finished with exception " << e.what() << std::endl;
    return false;
  }

  model.set_time(final_time);
  bool valid = true;

  // block 0: the FEM solution u(t_final) = tanh(K (t_final - t_mid))
  const auto &support_points = discretization.get_support_points();
  for (uint i = 0; i < support_points.size(); ++i) {
    const double is = initial_condition.data().block(0)[i];
    const double should = model.solution(support_points[i]);
    if (!is_close(is, should, expected_precision))
      std::cout << "FEM u: is " << is << " should be " << should << std::endl;
    valid &= is_close(is, should, expected_precision);
  }

  // block 1: the explicit variable v(t_final). This is the value that gets corrupted if a
  // rejected IDA step leaks into the explicit-variable buffer.
  const double v_is = initial_condition.data().block(1)[0];
  const double v_should = model.variable_solution();
  if (!is_close(v_is, v_should, expected_precision))
    std::cout << "explicit variable v: is " << v_is << " should be " << v_should
              << " (rel. error " << std::abs(v_is - v_should) / std::abs(v_should) << ")" << std::endl;
  valid &= is_close(v_is, v_should, expected_precision);

  if (!valid) std::cerr << "Failed " << test_name << std::endl;
  return valid;
}

TEST_CASE("Test SUNDIALS IDA + Boost ABM hybrid stepper", "[timestepping][sundials_ida_boost][abm]")
{
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, 1, UMFPack>;
  REQUIRE(run_hybrid<TimeStepper>("test_sundials_ida_boost_abm", 1e-3));
}

TEST_CASE("Test SUNDIALS IDA + Boost RK hybrid stepper", "[timestepping][sundials_ida_boost][rk]")
{
  using VectorType = dealii::Vector<double>;
  using SparseMatrixType = dealii::SparseMatrix<double>;
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostRK<VectorType, SparseMatrixType, 1, UMFPack, 0>;
  REQUIRE(run_hybrid<TimeStepper>("test_sundials_ida_boost_rk", 1e-3));
}
