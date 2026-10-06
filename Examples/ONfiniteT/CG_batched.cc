#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "model_batched.hh"

/**
 * CG.cc with a batched flux. /batched/backend selects how the model evaluates it:
 *   per_point -- per-point flux with the TBB integrator (AbstractModel's default flux_source_batch)
 *   tbb       -- one map_points call with the TBB integrator
 *   gpu       -- one map_points call with the GPU integrator in double precision
 *   gpu_float -- one map_points call with the GPU integrator in single precision
 */
template <typename Model> int run(const ConfigTree &config)
{
  constexpr uint dim = Model::dim;
  using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;
  using Assembler = CG::Assembler<Discretization>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  Model model(config);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(config)};
  OutputSession<Assembler> data_out(config);
  const auto log = data_out.report_port();
  Discretization discretization(mesh, config, log);
  Assembler assembler(discretization, model, config);
  HAdaptivity mesh_adaptor(assembler, config);
  TimeStepper time_stepper(config, assembler, data_out, mesh_adaptor);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  Timer timer;
  try {
    time_stepper.run(initial_condition, 0., config.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Simulation finished with exception {}", e.what());
    return -1;
  }
  log.info("Simulation finished after " + time_format(timer.wall_time()));
  return 0;
}

int main(int argc, char *argv[])
{
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto config = config_helper.get_config();

  using namespace ON_batched;
  const std::string backend = config.get_string("/batched/backend", "tbb");
  if (backend == "per_point") return run<Model<Backend::TBB, false>>(config);
  if (backend == "tbb") return run<Model<Backend::TBB>>(config);
  if (backend == "gpu") return run<Model<Backend::GPU>>(config);
  if (backend == "gpu_float") return run<Model<Backend::GPU_float>>(config);
  std::cerr << "Unknown /batched/backend: " << backend << std::endl;
  return 1;
}
