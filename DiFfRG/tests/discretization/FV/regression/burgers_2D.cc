#include "kt_regression_helpers.hh"

#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/discretization/mesh/no_adaptivity.hh"
#include "DiFfRG/timestepping/timestepping.hh"

#include <DiFfRG/common/json.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 2;
  constexpr std::size_t n_components = 1;
  constexpr unsigned int n_coarse_cells = 60;
  constexpr unsigned int n_medium_cells = 120;
  constexpr unsigned int n_fine_cells = 240;
  constexpr double domain_min = -1.5;
  constexpr double domain_max = 1.5;
  constexpr double domain_length = domain_max - domain_min;
  constexpr double final_time = 0.5;
  constexpr double disk_radius = 0.4;
  constexpr double positive_center_x = -0.5;
  constexpr double positive_center_y = -0.5;
  constexpr double negative_center_x = 0.5;
  constexpr double negative_center_y = 0.5;
  constexpr double implicit_dt = 1.0e-2;
  constexpr double l1_tolerance = 7.6e-2;
  constexpr double l2_tolerance = 9.5e-2;
  constexpr double linf_tolerance = 3.2e-1;
  constexpr double medium_l1_refinement_factor = 0.75;
  constexpr double medium_l2_refinement_factor = 0.85;
  constexpr double min_tolerance = -1.05;
  constexpr double max_tolerance = 1.05;
  constexpr double mass_tolerance = 2.0e-3;
  constexpr double antisymmetry_tolerance = 1.0e-6;

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMesh<dim>;
  using Discretization = FV::Discretization<Components, NumberType, Mesh>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, class Burgers2DExample11Model, Reconstructor>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;

  bool inside_disk(const Point<dim> &pos, const double center_x, const double center_y)
  {
    const double dx = pos[0] - center_x;
    const double dy = pos[1] - center_y;
    return dx * dx + dy * dy <= disk_radius * disk_radius;
  }

  double initial_value(const Point<dim> &pos)
  {
    if (inside_disk(pos, positive_center_x, positive_center_y)) return 1.0;
    if (inside_disk(pos, negative_center_x, negative_center_y)) return -1.0;
    return 0.0;
  }

  class Burgers2DExample11Model : public def::AbstractModel<Burgers2DExample11Model, Components>,
                                  public def::Time,
                                  public def::LLFFlux<Burgers2DExample11Model>,
                                  public def::FlowBoundaries<Burgers2DExample11Model>,
                                  public def::FVDefaultBoundaries<Burgers2DExample11Model>,
                                  public def::AD<Burgers2DExample11Model>
  {
  public:
    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      values[0] = initial_value(pos);
    }

    template <typename NT, typename Solution>
    void KurganovTadmor_advection_flux(std::array<Tensor<1, dim, NT>, n_components> &F_i,
                                       [[maybe_unused]] const Point<dim> &pos, const Solution &sol) const
    {
      const auto &u = get<0>(sol);
      const auto flux = u[0] * u[0];
      F_i[0][0] = flux;
      F_i[0][1] = flux;
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, n_components> &s_i, [[maybe_unused]] const Point<dim> &pos,
                [[maybe_unused]] const Solution &sol) const
    {
      s_i[0] = NT(0.0);
    }
  };

  std::string grid_axis_string(const unsigned int n_cells)
  {
    std::ostringstream stream;
    stream << std::setprecision(17) << domain_min << ':' << domain_length / static_cast<double>(n_cells) << ':'
           << domain_max;
    return stream.str();
  }

  JSONValue make_run_json(const unsigned int n_cells)
  {
    const auto axis = grid_axis_string(n_cells);
    return json::value(
        {{"physical", {{"Lambda", 1.0}}},
         {"discretization",
          {{"fe_order", 0},
           {"threads", 1},
           {"batch_size", 64},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", axis}, {"y_grid", axis}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", final_time},
           {"explicit",
            {{"dt", 1.0e-3}, {"minimal_dt", 1.0e-8}, {"maximal_dt", 1.0e-3}, {"abs_tol", 1.0e-9}, {"rel_tol", 1.0e-9}}},
           {"implicit",
            {{"dt", implicit_dt},
             {"minimal_dt", 1.0e-8},
             {"maximal_dt", implicit_dt},
             {"abs_tol", 1.0e-6},
             {"rel_tol", 1.0e-5}}}}},
         {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", false}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config(const unsigned int n_cells)
  {
    const double dx = domain_length / static_cast<double>(n_cells);
    const std::vector<Config::GridAxis> axis{Config::GridAxis(domain_min, dx, domain_max)};
    return Config::ConfigurationMesh<dim>(0u, axis, axis);
  }

  std::size_t row_major_cell_index(const Point<dim> &pos, const unsigned int n_cells)
  {
    const double dx = domain_length / static_cast<double>(n_cells);
    const auto index = [dx, n_cells](const double coordinate) {
      const auto raw = static_cast<int>(std::floor((coordinate - domain_min) / dx));
      return static_cast<unsigned int>(std::clamp(raw, 0, static_cast<int>(n_cells) - 1));
    };
    const unsigned int ix = index(pos[0]);
    const unsigned int iy = index(pos[1]);
    return static_cast<std::size_t>(iy) * n_cells + ix;
  }

  std::vector<double> row_major_cell_values(const Discretization &discretization, const VectorType &values,
                                            const unsigned int n_cells)
  {
    std::vector<double> result(static_cast<std::size_t>(n_cells) * n_cells, std::numeric_limits<double>::quiet_NaN());
    std::vector<unsigned int> counts(result.size(), 0);

    const auto &support_points = discretization.get_support_points();
    if (support_points.size() != values.size())
      throw std::runtime_error("Unexpected support-point/value size mismatch.");

    for (unsigned int dof = 0; dof < values.size(); ++dof) {
      const std::size_t row_major = row_major_cell_index(support_points[dof], n_cells);
      if (counts[row_major] != 0) throw std::runtime_error("Duplicate row-major cell index while sampling FV state.");
      result[row_major] = values[dof];
      ++counts[row_major];
    }

    for (const auto count : counts)
      if (count != 1) throw std::runtime_error("Missing row-major cell index while sampling FV state.");

    return result;
  }

  std::vector<double> run_burgers(const unsigned int n_cells, const std::string &run_name)
  {
    kt_regression::ensure_logger();

    const JSONValue json = make_run_json(n_cells);
    Burgers2DExample11Model model;
    Mesh mesh(make_mesh_config(n_cells));

    Discretization discretization(mesh, json, DiFfRG::LogPort{});
    Assembler assembler(discretization, model, json, DiFfRG::LogPort{});

    auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, run_name, "output");
    OutputSession<dim, VectorType> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());

    time_stepper.run(&state, 0.0, final_time);
    return row_major_cell_values(discretization, state.spatial_data(), n_cells);
  }

  std::vector<double> average_to_grid(const std::vector<double> &fine_values, const unsigned int fine_cells,
                                      const unsigned int target_cells)
  {
    REQUIRE(fine_values.size() == static_cast<std::size_t>(fine_cells) * fine_cells);
    REQUIRE(fine_cells % target_cells == 0);

    const unsigned int ratio = fine_cells / target_cells;

    std::vector<double> target(static_cast<std::size_t>(target_cells) * target_cells, 0.0);
    std::vector<unsigned int> counts(target.size(), 0);

    for (unsigned int fine_y = 0; fine_y < fine_cells; ++fine_y) {
      const unsigned int target_y = fine_y / ratio;
      for (unsigned int fine_x = 0; fine_x < fine_cells; ++fine_x) {
        const unsigned int target_x = fine_x / ratio;
        const std::size_t fine_index = static_cast<std::size_t>(fine_y) * fine_cells + fine_x;
        const std::size_t target_index = static_cast<std::size_t>(target_y) * target_cells + target_x;
        target[target_index] += fine_values[fine_index];
        ++counts[target_index];
      }
    }

    for (std::size_t i = 0; i < target.size(); ++i) {
      REQUIRE(counts[i] == ratio * ratio);
      target[i] /= static_cast<double>(counts[i]);
    }
    return target;
  }

  struct RegressionMetrics {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
    double mass = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double antisymmetry = 0.0;
  };

  RegressionMetrics compute_metrics(const unsigned int n_cells, const std::vector<double> &numerical,
                                    const std::vector<double> &downsampled_fine)
  {
    REQUIRE(numerical.size() == static_cast<std::size_t>(n_cells) * n_cells);
    REQUIRE(downsampled_fine.size() == numerical.size());

    const double dx = domain_length / static_cast<double>(n_cells);
    const double cell_measure = dx * dx;
    RegressionMetrics metrics;
    for (unsigned int y = 0; y < n_cells; ++y) {
      for (unsigned int x = 0; x < n_cells; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * n_cells + x;
        const double value = numerical[index];
        const double error = value - downsampled_fine[index];
        const double abs_error = std::abs(error);
        metrics.l1 += cell_measure * abs_error;
        metrics.l2 += cell_measure * error * error;
        metrics.linf = std::max(metrics.linf, abs_error);
        metrics.mass += cell_measure * value;
        metrics.min = std::min(metrics.min, value);
        metrics.max = std::max(metrics.max, value);

        const std::size_t reflected = static_cast<std::size_t>(n_cells - 1 - y) * n_cells + (n_cells - 1 - x);
        metrics.antisymmetry = std::max(metrics.antisymmetry, std::abs(value + numerical[reflected]));
      }
    }
    metrics.l2 = std::sqrt(metrics.l2);
    return metrics;
  }

  void require_finite_metrics(const RegressionMetrics &metrics)
  {
    REQUIRE(std::isfinite(metrics.l1));
    REQUIRE(std::isfinite(metrics.l2));
    REQUIRE(std::isfinite(metrics.linf));
    REQUIRE(std::isfinite(metrics.mass));
    REQUIRE(std::isfinite(metrics.min));
    REQUIRE(std::isfinite(metrics.max));
    REQUIRE(std::isfinite(metrics.antisymmetry));
  }

  void check_physical_invariants(const RegressionMetrics &metrics)
  {
    CHECK(metrics.min >= min_tolerance);
    CHECK(metrics.max <= max_tolerance);
    CHECK(std::abs(metrics.mass) < mass_tolerance);
    CHECK(metrics.antisymmetry < antisymmetry_tolerance);
  }
} // namespace

TEST_CASE("2D Burgers disk benchmark matches downsampled fine-grid run", "[2d][FV][KT][burgers]")
{
  const auto fine = run_burgers(n_fine_cells, "burgers_2D_fine_grid");

  const auto downsampled_fine_coarse = average_to_grid(fine, n_fine_cells, n_coarse_cells);
  const auto coarse = run_burgers(n_coarse_cells, "burgers_2D_regression");
  const RegressionMetrics coarse_metrics = compute_metrics(n_coarse_cells, coarse, downsampled_fine_coarse);

  const auto downsampled_fine_medium = average_to_grid(fine, n_fine_cells, n_medium_cells);
  const auto medium = run_burgers(n_medium_cells, "burgers_2D_medium_grid");
  const RegressionMetrics medium_metrics = compute_metrics(n_medium_cells, medium, downsampled_fine_medium);

  std::cout << std::scientific << std::setprecision(8)
            << "2D Burgers 60x60 vs downsampled 240x240: L1=" << coarse_metrics.l1 << ", L2=" << coarse_metrics.l2
            << ", Linf=" << coarse_metrics.linf << ", mass=" << coarse_metrics.mass << ", min=" << coarse_metrics.min
            << ", max=" << coarse_metrics.max << ", antisymmetry=" << coarse_metrics.antisymmetry << '\n'
            << "2D Burgers 120x120 vs downsampled 240x240: L1=" << medium_metrics.l1 << ", L2=" << medium_metrics.l2
            << ", Linf=" << medium_metrics.linf << ", mass=" << medium_metrics.mass << ", min=" << medium_metrics.min
            << ", max=" << medium_metrics.max << ", antisymmetry=" << medium_metrics.antisymmetry << '\n';

  require_finite_metrics(coarse_metrics);
  require_finite_metrics(medium_metrics);

  CHECK(coarse_metrics.l1 < l1_tolerance);
  CHECK(coarse_metrics.l2 < l2_tolerance);
  CHECK(coarse_metrics.linf < linf_tolerance);
  CHECK(medium_metrics.l1 < medium_l1_refinement_factor * coarse_metrics.l1);
  CHECK(medium_metrics.l2 < medium_l2_refinement_factor * coarse_metrics.l2);

  check_physical_invariants(coarse_metrics);
  check_physical_invariants(medium_metrics);
}
