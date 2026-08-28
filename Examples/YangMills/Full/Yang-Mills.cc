#include <DiFfRG/DiFfRG.hh>

#include <optional>
using namespace DiFfRG;

#include "model.hh"
#include "tuning.hh"

// Choices for types
using Model = YangMills;
using VectorType = Vector<double>;
using Assembler = Variables::Assembler<Model>;
using TimeStepper = TimeStepperBoostABM<VectorType, dealii::SparseMatrix<get_type::NumberType<VectorType>>, 0>;

bool run(const ConfigTree &json, const OutputPath &output_path, std::optional<ReportPort> external_log = std::nullopt)
{
  // Define the objects needed to run the simulation
  OutputSession<0, VectorType> data_out(output_path, json);
  const auto log = external_log.value_or(data_out.report_port());
  Model model(json);
  Assembler assembler(model, json);
  TimeStepper time_stepper(json, assembler, data_out);

  // Set up the initial condition
  FlowingVariables initial_condition;
  initial_condition.interpolate(model);

  // Start the timestepping
  try {
    time_stepper.run(initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Timestepping finished with exception {}", e.what());
    log.flush();
    return false;
  }

  HDF5Input hdf5_input(output_path.run_file(".h5").string());

  // Divergence detection: on the Higgs branch the flow blows up at intermediate k and the timestepper
  // stalls there without throwing, so the run "finishes" far from the IR. A genuine massive/scaling
  // solution flows all the way down to k = Lambda*exp(-final_time). If the last recorded k is well
  // above that target, the run diverged before reaching the IR -> treat as the Higgs branch.
  const auto k_final = hdf5_input.load_scalar<double>("k").back();
  const double k_target = json.get_double("/physical/Lambda") * std::exp(-json.get_double("/timestepping/final_time"));
  if (k_final > 100. * k_target) {
    log.error("Diverged before reaching IR: k_final = {} (target {})", k_final, k_target);
    return false;
  }

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

  // The Yang-Mills flow must be tuned in the initial gluon mass parameter m2A: too negative
  // lands on the Higgs branch (the solution diverges), too small on the massive branch, and the
  // scaling regime lies in between. Set "/tuning/tune_m2A" to bisect m2A onto that separatrix.
  if (json.get_bool("/tuning/tune_m2A")) {
    tune_m2A(json, output_path, run);
  } else {
    run(json, output_path);
  }

  // We print a bit of exit information.
  const auto time = timer.wall_time();
  std::cout << "Program finished after " << time_format(time) << std::endl;
  return 0;
}
