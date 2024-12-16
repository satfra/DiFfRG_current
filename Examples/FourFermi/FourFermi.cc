#include <filesystem>

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/csv_reader.hh>
#include <DiFfRG/common/root_finding.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include "DiFfRG/discretization/data/data_output.hh"
#include "model.hh"

using namespace dealii;
using namespace DiFfRG;

// Choices for types
using Model = FourFermi;
using VectorType = Vector<double>;
using Assembler = Variables::Assembler<Model>;
using TimeStepper = TimeStepperBoostABM<VectorType>;

int main(int argc, char *argv[])
{
  Timer timer;

  // get all needed parameters and parse from the CLI
  ConfigurationHelper config_helper(argc, argv);
  auto json = config_helper.get_json();

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
    spdlog::get("log")->flush();
  }

  // We print a bit of exit information.
  const auto time = timer.wall_time();
  spdlog::get("log")->info("Program finished after " + time_format(time));
}