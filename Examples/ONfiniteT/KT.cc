// FV / Kurganov-Tadmor headers must be included before any header that pulls in
// DiFfRG::Quadrature<NT> (e.g. DiFfRG/physics/integration.hh via DiFfRG/DiFfRG.hh),
// otherwise unqualified Quadrature<dim> inside KurganovTadmor.hh resolves to the
// DiFfRG class template instead of dealii::Quadrature<int>.
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/model/fv_boundaries.hh>

#include <tbb/global_control.h>

#include "model_KT.hh"

// Choices for types
using Model = ON_finiteT_KT;
constexpr uint dim = Model::dim;
using Discretization = FV::Discretization<Model::Components, double, RectangularMesh<dim>>;
using VectorType = typename Discretization::VectorType;
using SparseMatrixType = typename Discretization::SparseMatrixType;
using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;
using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;

int main(int argc, char *argv[])
{
  // DIAGNOSTIC: pin the TBB pool used by the quadrature integrator. The JSON
  // /discretization/threads only controls MeshWorker, not TBB.
  tbb::global_control tbb_pin(tbb::global_control::max_allowed_parallelism, 1);

  // Initialize DiFfRG and thus the MPI and Kokkos environments
  const auto config_helper = DiFfRG::Init(argc, argv, "parameter_KT.json").get_configuration_helper();
  // get all needed parameters and parse from the CLI
  const auto json = config_helper.get_json();

  // Define the objects needed to run the simulation
  Model model(json);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  OutputPath output_path(json);
  OutputSession<dim, VectorType> data_out(output_path, json);
  const auto log = data_out.log_port();
  Discretization discretization(mesh, json, log);
  Assembler assembler(discretization, model, json, log);
  // KT requires a fixed rectangular mesh; no h-adaptivity.
  NoAdaptivity<VectorType> mesh_adaptor;
  TimeStepper time_stepper(json, &assembler, &data_out, &mesh_adaptor);

  // Set up the initial condition
  FV::FlowingVariables<Discretization> initial_condition(discretization);
  initial_condition.interpolate(model);

  // Now we start the timestepping
  Timer timer;
  try {
    time_stepper.run(&initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Simulation finished with exception {}", e.what());
    return -1;
  }
  auto time = timer.wall_time();
  assembler.log();
  log.info("Simulation finished after " + time_format(time));
  return 0;
}
