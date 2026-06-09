#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "model.hh"

// Choices for types. This is a pure variable system (no FE functions), so we use the
// Variables assembler (spatial dimension 0) and an explicit Adams-Bashforth-Moulton stepper.
using Model = YangMills;
using VectorType = Vector<double>;
using Assembler = Variables::Assembler<Model>;
using TimeStepper = TimeStepperBoostABM<VectorType, dealii::SparseMatrix<get_type::NumberType<VectorType>>, 0>;

int main(int argc, char *argv[])
{
  Timer timer;

  // Initialize DiFfRG (MPI + Kokkos) and read the parameter file / CLI overrides
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto json = config_helper.get_json();

  // Define the objects needed to run the simulation
  Model model(json);
  Assembler assembler(model, json);
  TimeStepper time_stepper(json, &assembler);

  // Set up the initial condition
  FlowingVariables initial_condition;
  initial_condition.interpolate(model);

  // Start the timestepping
  try {
    time_stepper.run(&initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    spdlog::get("log")->error("Timestepping finished with exception {}", e.what());
    return -1;
  }

  // Read the gluon mass and the ghost dressing at vanishing momentum back from the output.
  HDF5Input hdf5_input(config_helper.get_top_folder() + "/" + config_helper.get_output_name() + ".h5");
  const auto m2A = hdf5_input.load_scalar<double>("m2A").back();
  std::vector<double> Zc(p_grid_size);
  hdf5_input.load_map("Zc", Zc.data());

  spdlog::get("log")->info("Simulation finished after {}. m2A = {}, Zc(0) = {}", time_format(timer.wall_time()), m2A,
                           Zc[0]);
  return 0;
}
