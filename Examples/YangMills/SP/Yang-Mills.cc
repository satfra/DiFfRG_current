#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "model.hh"
#include "tuning.hh"

// Choices for types
using Model = YangMills;
using Assembler = Variables::Assembler<Model>;
using TimeStepper = TimeStepperBoostABM<Assembler>;

bool run(const ConfigTree &json, const OutputPath &output_path, LogPort external_log = {})
{
  // Define the objects needed to run the simulation
  OutputSession<Assembler> data_out(output_path, json);
  const auto log = external_log ? external_log : data_out.log_port();
  Model model(json);
  Assembler assembler(model, json, log);
  TimeStepper time_stepper(json, &assembler, &data_out);

  // Set up the initial condition
  FlowingVariables initial_condition;
  initial_condition.interpolate(model);

  // Start the timestepping
  try {
    time_stepper.run(&initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Timestepping finished with exception {}", e.what());
    log.flush();
    return false;
  }

  HDF5Input hdf5_input(output_path.run_file(".h5").string());
  const auto m2A = hdf5_input.load_scalar<double>("m2A").back();
  std::vector<double> Zc(p_grid_size);
  hdf5_input.load_map("Zc", Zc.data());
  std::cout << "m2A: " << m2A << "\n";
  std::cout << "Zc(0): " << Zc[0] << std::endl;

  if (m2A < 0 || m2A > 1e4) {
    log.error("Diverging result: m2A = {}, Zc(0) = {}", m2A, Zc[0]);
    return false;
  }

  bool Zc_invalid = false;
  Zc_invalid |= Zc[0] < 0;
  Zc_invalid |= Zc[0] > 1;
  if (Zc_invalid) {
    log.error("Diverging result: Zc(0) = {}", Zc[0]);
    return false;
  }

  log.info("Timestepping finished successfully, m2A = {}, Zc(0) = {}", m2A, Zc[0]);
  return true;
}

int main(int argc, char *argv[])
{
  Timer timer;

  // get all needed parameters and parse from the CLI
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  auto json = config_helper.get_json();
  OutputPath output_path(json);

  if (json.get_bool("/tuning/tune_STI")) {
    tune_STI(json, output_path, run);
  } else if (json.get_bool("/tuning/tune_m2A")) {
    tune_m2A(json, output_path, run);
  } else {
    const auto ret_val = run(json, output_path);
    // We print a bit of exit information.
    const auto time = timer.wall_time();
    std::cout << "Program finished after " << time_format(time) << std::endl;
    return ret_val;
  }

  // We print a bit of exit information.
  const auto time = timer.wall_time();
  std::cout << "Program finished after " << time_format(time) << std::endl;
  return 0;
}
