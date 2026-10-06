#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "model_batched.hh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <optional>
#include <sstream>

/**
 * Times residual() and jacobian() of the ONfiniteT CG assembly for one configuration of the flux:
 *   B0  -- per-point flux with the TBB integrator (AbstractModel's default flux_source_batch)
 *   B1  -- flux_source_batch with one map_points call on TBB
 *   G64 -- flux_source_batch with one map_points call on the GPU, double precision
 *   G32 -- flux_source_batch with one map_points call on the GPU, single precision
 *
 * Usage: bench_batched --config B1 --cells 1024 --xorder 32 [--reps 10] [--threads 0] [--policy auto]
 *                      [--parameters parameter.toml]
 * Prints one line "RESULT,<csv>" with the header given by --header.
 */
namespace
{
  struct Options {
    std::string config = "B1";
    uint cells = 64;
    uint xorder = 32;
    uint reps = 10;
    uint threads = 0;
    std::string policy = "auto";
    std::string parameters = "parameter.toml";
  };

  double median(std::vector<double> v)
  {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }

  MapPointsPolicy parse_policy(const std::string &p)
  {
    if (p == "thread") return MapPointsPolicy::thread_per_point;
    if (p == "team") return MapPointsPolicy::team_per_point;
    return MapPointsPolicy::automatic;
  }

  template <typename Model> void run(const Options &opt, const ConfigTree &config)
  {
    constexpr uint dim = Model::dim;
    using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;
    using Assembler = CG::Assembler<Discretization>;
    using VectorType = typename Discretization::VectorType;
    using SparseMatrixType = typename Discretization::SparseMatrixType;
    using clock = std::chrono::steady_clock;

    Model model(config);
    model.set_time(1.);
    model.integrator().integrator.set_map_points_policy(parse_policy(opt.policy));
    model.integrator().integrator_AD.set_map_points_policy(parse_policy(opt.policy));

    RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(config)};
    Discretization discretization(mesh, config);
    Assembler assembler(discretization, model, config);

    FE::FlowingVariables state(discretization);
    state.interpolate(model);
    const VectorType &u = state.spatial_data();
    const VectorType u_dot(u);
    VectorType residual(u);
    SparseMatrixType jacobian(assembler.get_sparsity_pattern_jacobian());

    const auto time_residual = [&] {
      residual = 0;
      const auto t0 = clock::now();
      assembler.residual(residual, u, 1., u_dot, 1.);
      return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };
    const auto time_jacobian = [&] {
      jacobian = 0;
      const auto t0 = clock::now();
      assembler.jacobian(jacobian, u, 1., u_dot, 1., 1.);
      return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    for (int i = 0; i < 3; ++i) {
      time_residual();
      time_jacobian();
    }
    assembler.reset_phase_times();

    std::vector<double> t_res, t_jac;
    for (uint i = 0; i < opt.reps; ++i)
      t_res.push_back(time_residual());
    for (uint i = 0; i < opt.reps; ++i)
      t_jac.push_back(time_jacobian());

    const auto &r = assembler.residual_phase_times();
    const auto &j = assembler.jacobian_phase_times();
    const std::array<double, 6> phases = {r.gather / r.calls * 1e3,   r.evaluate / r.calls * 1e3,
                                          r.scatter / r.calls * 1e3,  j.gather / j.calls * 1e3,
                                          j.evaluate / j.calls * 1e3, j.scatter / j.calls * 1e3};

    const size_t n_points = discretization.get_triangulation().n_active_cells() * (discretization.get_fe().degree + 1);
    std::ostringstream line;
    line << std::setprecision(6) << "RESULT," << opt.config << "," << opt.cells << "," << opt.xorder << "," << n_points << ","
         << (opt.threads == 0 ? DiFfRG::n_threads() : opt.threads) << "," << opt.policy << "," << median(t_res) << "," << median(t_jac);
    for (const double p : phases)
      line << "," << p;
    line << std::setprecision(15) << "," << residual.l2_norm() << "," << jacobian.frobenius_norm();
    std::cout << line.str() << std::endl;
  }
} // namespace

int main(int argc, char *argv[])
{
  Options opt;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--config")
      opt.config = value;
    else if (key == "--cells")
      opt.cells = std::stoul(value);
    else if (key == "--xorder")
      opt.xorder = std::stoul(value);
    else if (key == "--reps")
      opt.reps = std::stoul(value);
    else if (key == "--threads")
      opt.threads = std::stoul(value);
    else if (key == "--policy")
      opt.policy = value;
    else if (key == "--parameters")
      opt.parameters = value;
    else {
      std::cerr << "Unknown option " << key << std::endl;
      return 1;
    }
  }
  if (argc == 2 && std::strcmp(argv[1], "--header") == 0) {
    std::cout << "config,cells,xorder,points,threads,policy,residual_ms,jacobian_ms,res_gather_ms,res_evaluate_ms,"
                 "res_scatter_ms,jac_gather_ms,jac_evaluate_ms,jac_scatter_ms,residual_norm,jacobian_norm"
              << std::endl;
    return 0;
  }

  // Must outlive Init and the assembler: TBB and Kokkos pick up the active parallelism at Init.
  std::optional<tbb::global_control> thread_limit;
  if (opt.threads > 0) thread_limit.emplace(tbb::global_control::max_allowed_parallelism, opt.threads);
  char *init_argv[] = {argv[0]};
  DiFfRG::Init(1, init_argv);

  ConfigTree config(opt.parameters);
  std::ostringstream step;
  step << std::setprecision(17) << 0.015 / opt.cells;
  config.set_string("/discretization/grid/x_grid", "0:" + step.str() + ":0.015");
  config.set_uint("/discretization/grid/refine", 0);
  config.set_uint("/integration/x_order", opt.xorder);
  config.set_uint("/output/verbosity", 0);

  using namespace ON_batched;
  if (opt.config == "B0")
    run<Model<Backend::TBB, false>>(opt, config);
  else if (opt.config == "B1")
    run<Model<Backend::TBB>>(opt, config);
  else if (opt.config == "G64")
    run<Model<Backend::GPU>>(opt, config);
  else if (opt.config == "G32")
    run<Model<Backend::GPU_float>>(opt, config);
  else {
    std::cerr << "Unknown --config " << opt.config << std::endl;
    return 1;
  }
  return 0;
}
