// FV / Kurganov-Tadmor headers must be included before any header that pulls in
// DiFfRG::Quadrature<NT> (e.g. DiFfRG/physics/integration.hh via DiFfRG/DiFfRG.hh),
// otherwise unqualified Quadrature<dim> inside KurganovTadmor.hh resolves to the
// DiFfRG class template instead of dealii::Quadrature<int>.
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>
#include <DiFfRG/model/fv_boundaries.hh>

#include "model_KT_sigma.hh"

// Choices for types
using Model = ON_finiteT_KT_sigma;
constexpr uint dim = Model::dim;
using Discretization = FV::Discretization<Model::Components, double, RectangularMesh<dim>>;
using VectorType = typename Discretization::VectorType;
using SparseMatrixType = typename Discretization::SparseMatrixType;
using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
using WaveSpeed = FV::KurganovTadmor::MaxEigenvalueWaveSpeed;
using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor, WaveSpeed>;
using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;

int main(int argc, char *argv[])
{
  // Initialize DiFfRG and thus the MPI and Kokkos environments.
  const auto config_helper = DiFfRG::Init(argc, argv, "parameter_KT_sigma.json").get_configuration_helper();
  const auto json = config_helper.get_json();

  // Define the objects needed to run the simulation
  Model model(json);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  OutputPath output_path(json);
  OutputSession<dim, VectorType> data_out(output_path, json);
  const auto log = data_out.log_port();
  Discretization discretization(mesh, json, log);
  Assembler assembler(discretization, model, json, log);
  NoAdaptivity<VectorType> mesh_adaptor;
  TimeStepper time_stepper(json, &assembler, &data_out, &mesh_adaptor);

  // Set up the initial condition
  FV::FlowingVariables<Discretization> initial_condition(discretization);
  initial_condition.interpolate(model);

  // EXPERIMENT: mirror the test's initialize_exact_cell_averages — override
  // each cell with the exact cell-averaged value of V_init. If this fixes
  // the IDA failure, the difference is that interpolate() does NOT produce
  // the same values as model.initial_condition for FV.
  {
    const auto &support_points = discretization.get_support_points();
    auto &u = initial_condition.data().block(0);
    const double dx = 1.16e-3;  // matches parameter_sigma.json grid step
    const double m2 = -0.1, lam = 71.6;
    auto V = [&](double s) { return 0.5 * m2 * s * s + (lam / 8.0) * s * s * s * s; };
    for (unsigned int i = 0; i < u.size(); ++i) {
      const double sigma = support_points[i][0];
      u[i] = (V(sigma + 0.5 * dx) - V(sigma - 0.5 * dx)) / dx;
    }
  }

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
