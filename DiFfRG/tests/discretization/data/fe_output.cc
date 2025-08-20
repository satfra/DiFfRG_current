#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test FE output on Constant model", "[output][cg]")
{
  DiFfRG::Init();

  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = CG::Assembler<Discretization, Model>;

  JSONValue json = json::value(
      {{"physical", {}},
       {"integration",
        {{"x_quadrature_order", 32},
         {"angle_quadrature_order", 8},
         {"x0_quadrature_order", 16},
         {"x0_summands", 8},
         {"q0_quadrature_order", 16},
         {"q0_summands", 8},
         {"x_extent_tolerance", 1e-3},
         {"x0_extent_tolerance", 1e-3},
         {"q0_extent_tolerance", 1e-3},
         {"jacobian_quadrature_factor", 0.5}}},
       {"discretization",
        {{"fe_order", 3},
         {"threads", 8},
         {"batch_size", 64},
         {"overintegration", 0},
         {"output_subdivisions", 2},
         {"output_buffer_size", 10},

         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},

         {"grid", {{"x_grid", "0:0.0001:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
         {"adaptivity",
          {{"start_adapt_at", 0.},
           {"adapt_dt", 1e-1},
           {"level", 0},
           {"refine_percent", 1e-1},
           {"coarsen_percent", 5e-2}}}}},
       {"timestepping",
        {{"final_time", 1.},
         {"output_dt", 1e-1},
         {"explicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}},
         {"implicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}}}},
       {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  Testing::PhysicalParameters p_prm = {/*x0_initial = */ 0., /*x1_initial = */ GENERATE(take(1, random(1., 10.)))};

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG Application started");
  } catch (const spdlog::spdlog_ex &e) {
    // nothing, the logger is already set up
  }

  // Define the objects needed to run the simulation
  Model model(p_prm);
  RectangularMesh<dim> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const VectorType &src = initial_condition.spatial_data();

  Timer timer;
  {
    FEOutput<dim, VectorType> fe_output("./testing", "output_name", "other_folder", json);

    HDF5::File file("./testing/output_name.h5", HDF5::File::FileAccessMode::create);
    auto h5_fe_group = std::make_shared<HDF5::Group>(file.create_group("FE"));
    fe_output.set_h5_group(h5_fe_group);

    constexpr uint output_num = 100;
    for (uint i = 0; i < output_num; ++i) {
      fe_output.attach(discretization.get_dof_handler(), src, "solution");
      fe_output.flush(i);
    }
    std::cout << "FEOutput flushed " << output_num << " times in " << timer.wall_time() << " seconds." << std::endl;
  }
  std::cout << "FEOutput finished after " << timer.wall_time() << " seconds." << std::endl;
}