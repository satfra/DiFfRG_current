#include "../kt_regression_helpers.hh"
#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/timestepping/linear_solver/ScaledGMRES.hh"
#include "DiFfRG/timestepping/timestepping.hh"

#include <DiFfRG/common/json.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>
#include <deal.II/base/multithread_info.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 2;
  constexpr std::size_t n_components = 2;
  constexpr double phi_min = -10.0;
  constexpr double phi_max = 10.0;
  constexpr std::size_t default_n_cells_per_direction = 400;
  constexpr double default_final_time = 60.0;
  constexpr double gamma_reference = 0.178669819;
  constexpr double gamma_tolerance = 5.0e-3;
  constexpr double point_tolerance = 1.0e-10;

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMesh<dim>;
  using Discretization = FV::Discretization<Components, NumberType, Mesh>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;

  class O2_Model_VI_B_III : public def::AbstractModel<O2_Model_VI_B_III, Components>,
                            public def::fRG,
                            public def::LLFFlux<O2_Model_VI_B_III>,
                            public def::FlowBoundaries<O2_Model_VI_B_III>,
                            public def::FVDefaultBoundaries<O2_Model_VI_B_III>,
                            public def::AD<O2_Model_VI_B_III>
  {
  public:
    O2_Model_VI_B_III() : def::fRG(1.0) {}
    explicit O2_Model_VI_B_III(const JSONValue &json) : def::fRG(json) {}

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      const double phi_1 = pos[0];
      const double phi_2 = pos[1];
      const double r2 = phi_1 * phi_1 + phi_2 * phi_2;
      const double factor = 1.0 - (1.0 / 5.0) * r2 + (1.0 / 120.0) * r2 * r2;

      values[0] = phi_1 * factor;
      values[1] = phi_2 * factor;
    }

    template <typename NT, typename Solution>
    void KurganovTadmor_advection_flux(std::array<Tensor<1, dim, NT>, n_components> &F_i,
                                       [[maybe_unused]] const Point<dim> &pos,
                                       [[maybe_unused]] const Solution &sol) const
    {
      for (auto &flux_i : F_i)
        flux_i = Tensor<1, dim, NT>();
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
              [[maybe_unused]] const Solution &sol) const
    {
      for (auto &flux_i : F_i)
        flux_i = Tensor<1, dim, NT>();

      const auto &fe_derivatives = get<1>(sol);
      const NT du_dphi1 = fe_derivatives[0][0];
      const NT du_dphi2 = fe_derivatives[0][1];
      const NT dv_dphi1 = fe_derivatives[1][0];
      const NT dv_dphi2 = fe_derivatives[1][1];

      const double r = k;
      const NT denominator = (NT(r) + du_dphi1) * (NT(r) + dv_dphi2) - dv_dphi1 * du_dphi2;
      const NT Q = NT(0.5 * r) * (NT(2.0 * r) + du_dphi1 + dv_dphi2) / denominator;

      F_i[0][0] = Q;
      F_i[1][1] = Q;
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, n_components> &s_i, [[maybe_unused]] const Point<dim> &pos,
                [[maybe_unused]] const Solution &sol) const
    {
      for (auto &source_i : s_i)
        source_i = NT(0.0);
    }
  };

  using O2Assembler = FV::KurganovTadmor::Assembler<Discretization, O2_Model_VI_B_III, Reconstructor>;

  struct GammaRunParameters {
    std::size_t n_cells_per_direction = default_n_cells_per_direction;
    double final_time = default_final_time;
    bool retain_output = true;
  };

  double cell_width(const GammaRunParameters &params)
  {
    return (phi_max - phi_min) / static_cast<double>(params.n_cells_per_direction);
  }

  double center_min(const GammaRunParameters &params) { return phi_min + 0.5 * cell_width(params); }

  std::string grid_axis_expression(const GammaRunParameters &params)
  {
    return std::to_string(phi_min) + ":" + std::to_string(cell_width(params)) + ":" + std::to_string(phi_max);
  }

  JSONValue make_run_json(const GammaRunParameters &params)
  {
    const std::string phi_grid = grid_axis_expression(params);
    return json::value(
        {{"physical", {{"Lambda", 1.0e12}}},
         {"discretization",
          {{"fe_order", 0},
           {"threads", static_cast<std::int64_t>(dealii::MultithreadInfo::n_threads())},
           {"batch_size", 64},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"output_buffer_size", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", phi_grid}, {"y_grid", phi_grid}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
         {"timestepping",
          {{"final_time", params.final_time},
           {"output_dt", 1.0},
           {"explicit",
            {{"dt", 1.0e-3}, {"minimal_dt", 1.0e-10}, {"maximal_dt", 1.0}, {"abs_tol", 1.0e-12}, {"rel_tol", 1.0e-10}}},
           {"implicit",
            {{"dt", 1.0e-3},
             {"minimal_dt", 1.0e-14},
             {"maximal_dt", 1.0},
             {"abs_tol", 1.0e-12},
             {"rel_tol", 1.0e-10},
             {"max_steps", 5000000}}}}},
         {"output", {{"verbosity", 3}, {"vtk", true}, {"hdf5", true}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config(const GammaRunParameters &params)
  {
    const std::vector<Config::GridAxis> phi_axis{Config::GridAxis(phi_min, cell_width(params), phi_max)};
    return Config::ConfigurationMesh<dim>(0u, phi_axis, phi_axis);
  }

  struct SampledGrid {
    std::size_t n_cells_per_direction = 0;
    double cell_width = 0.0;
    std::vector<double> u;
    std::vector<double> v;

    std::size_t index(const std::size_t ix, const std::size_t iy) const { return iy * n_cells_per_direction + ix; }

    double value(const std::vector<double> &component, const std::size_t ix, const std::size_t iy) const
    {
      return component[index(ix, iy)];
    }
  };

  std::size_t coordinate_index(const double x, const GammaRunParameters &params)
  {
    const double width = cell_width(params);
    const double floating_index = (x - center_min(params)) / width;
    const auto index = static_cast<long>(std::llround(floating_index));
    if (index < 0 || index >= static_cast<long>(params.n_cells_per_direction))
      throw std::runtime_error("Sampled support point lies outside the expected O(2) Gamma grid.");
    const double expected = center_min(params) + static_cast<double>(index) * width;
    if (std::abs(x - expected) > point_tolerance)
      throw std::runtime_error("Sampled support point does not lie on the expected uniform grid.");
    return static_cast<std::size_t>(index);
  }

  SampledGrid sample_final_grid(const FV::FlowingVariables<Discretization> &state, const Discretization &discretization,
                                const GammaRunParameters &params)
  {
    const auto &dof_handler = discretization.get_dof_handler();
    const auto &fe = discretization.get_fe();
    const auto &solution = state.spatial_data();

    if (dof_handler.get_triangulation().n_active_cells() != params.n_cells_per_direction * params.n_cells_per_direction)
      throw std::runtime_error("The O(2) Gamma regression does not have the expected number of active FV cells.");

    SampledGrid grid;
    grid.n_cells_per_direction = params.n_cells_per_direction;
    grid.cell_width = cell_width(params);
    const std::size_t n_grid_points = params.n_cells_per_direction * params.n_cells_per_direction;
    grid.u.assign(n_grid_points, std::numeric_limits<double>::quiet_NaN());
    grid.v.assign(n_grid_points, std::numeric_limits<double>::quiet_NaN());

    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
      cell->get_dof_indices(dof_indices);
      const std::size_t ix = coordinate_index(cell->center()[0], params);
      const std::size_t iy = coordinate_index(cell->center()[1], params);
      const std::size_t i = grid.index(ix, iy);

      for (unsigned int local_dof = 0; local_dof < fe.dofs_per_cell; ++local_dof) {
        const auto component = fe.system_to_component_index(local_dof).first;
        if (component == 0) {
          if (std::isfinite(grid.u[i])) throw std::runtime_error("Duplicate u sample in O(2) Gamma grid.");
          grid.u[i] = solution[dof_indices[local_dof]];
        } else if (component == 1) {
          if (std::isfinite(grid.v[i])) throw std::runtime_error("Duplicate v sample in O(2) Gamma grid.");
          grid.v[i] = solution[dof_indices[local_dof]];
        }
      }
    }

    for (std::size_t i = 0; i < n_grid_points; ++i) {
      if (!std::isfinite(grid.u[i]) || !std::isfinite(grid.v[i]))
        throw std::runtime_error("Missing or non-finite sample in O(2) Gamma grid.");
    }

    return grid;
  }

  struct Gamma2 {
    double gamma_11 = 0.0;
    double gamma_22 = 0.0;
    double gamma_12_from_u = 0.0;
    double gamma_12_from_v = 0.0;
    double gamma_12 = 0.0;
  };

  Gamma2 extract_origin_gamma2(const SampledGrid &grid)
  {
    if (grid.n_cells_per_direction % 2 != 0)
      throw std::runtime_error("Origin Gamma extraction expects an even number of cells per direction.");
    const std::size_t lower = grid.n_cells_per_direction / 2 - 1;
    const std::size_t upper = grid.n_cells_per_direction / 2;

    const auto u = [&](const std::size_t ix, const std::size_t iy) { return grid.value(grid.u, ix, iy); };
    const auto v = [&](const std::size_t ix, const std::size_t iy) { return grid.value(grid.v, ix, iy); };

    Gamma2 gamma;
    gamma.gamma_11 = 0.5 * ((u(upper, lower) - u(lower, lower)) / grid.cell_width +
                            (u(upper, upper) - u(lower, upper)) / grid.cell_width);
    gamma.gamma_22 = 0.5 * ((v(lower, upper) - v(lower, lower)) / grid.cell_width +
                            (v(upper, upper) - v(upper, lower)) / grid.cell_width);
    gamma.gamma_12_from_u = 0.5 * ((u(lower, upper) - u(lower, lower)) / grid.cell_width +
                                   (u(upper, upper) - u(upper, lower)) / grid.cell_width);
    gamma.gamma_12_from_v = 0.5 * ((v(upper, lower) - v(lower, lower)) / grid.cell_width +
                                   (v(upper, upper) - v(lower, upper)) / grid.cell_width);
    gamma.gamma_12 = 0.5 * (gamma.gamma_12_from_u + gamma.gamma_12_from_v);
    return gamma;
  }

  void log_gamma2(const std::string &label, const Gamma2 &gamma)
  {
    std::clog << std::setprecision(17) << "[O2 DIAG] " << label << " Gamma2: gamma_11=" << gamma.gamma_11
              << ", gamma_22=" << gamma.gamma_22 << ", gamma_12_from_u=" << gamma.gamma_12_from_u
              << ", gamma_12_from_v=" << gamma.gamma_12_from_v << ", gamma_12=" << gamma.gamma_12 << '\n';
  }

  double extracted_gamma2(const Gamma2 &gamma) { return 0.5 * (gamma.gamma_11 + gamma.gamma_22); }

  void log_gamma2_reference_comparison(const std::string &label, const Gamma2 &gamma)
  {
    const double extracted = extracted_gamma2(gamma);
    const double absolute_error = std::abs(extracted - gamma_reference);
    const double relative_error = absolute_error / std::abs(gamma_reference);
    std::clog << std::setprecision(17) << "[O2 DIAG] " << label
              << " Gamma2 reference comparison: extracted=" << extracted << ", absolute_error=" << absolute_error
              << ", relative_error=" << relative_error << ", tolerance=" << gamma_tolerance << '\n';
  }

  Gamma2 run_o2_vi_b_iii_gamma_regression(const GammaRunParameters &params = {})
  {
    kt_regression::ensure_logger();

    const JSONValue json = make_run_json(params);
    O2_Model_VI_B_III model(json);
    Mesh mesh(make_mesh_config(params));
    Discretization discretization(mesh, json);
    O2Assembler assembler(discretization, model, json);

    kt_regression::TemporaryDirectory tmp_dir("o2_model_kt_2D_case_iii_regression", !params.retain_output);
    std::clog << "[O2 DIAG] " << (params.retain_output ? "retaining" : "using temporary") << " output directory "
              << tmp_dir.path << '\n';
    DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), "o2_model_kt_2D_case_iii_regression", "output", json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());

    time_stepper.run(&state, 0.0, params.final_time);
    return extract_origin_gamma2(sample_final_grid(state, discretization, params));
  }
} // namespace

TEST_CASE("O2 Model VI_B_III exposes phi6 initial condition and diagonal diffusion flux", "[2d][FV][KT][O2][VI_B_III]")
{
  const O2_Model_VI_B_III model{};
  const Point<dim> point(0.25, 0.5);
  const std::array<double, n_components> fe_functions{{1.0, 2.0}};
  const std::array<Tensor<1, dim, double>, n_components> fe_derivatives{
      {Tensor<1, dim, double>({1.0, 1.0}), Tensor<1, dim, double>({1.0, 2.0})}};
  const auto solution = std::tuple{fe_functions, fe_derivatives};

  const auto expanded_phi6_initial_condition = [](const double phi_1, const double phi_2) {
    const double phi_1_2 = phi_1 * phi_1;
    const double phi_2_2 = phi_2 * phi_2;
    return std::array<double, n_components>{
        phi_1 - (1.0 / 5.0) * (phi_1 * phi_1_2 + phi_1 * phi_2_2) +
            (1.0 / 120.0) * (phi_1 * phi_1_2 * phi_1_2 + 2.0 * phi_1 * phi_1_2 * phi_2_2 + phi_1 * phi_2_2 * phi_2_2),
        phi_2 - (1.0 / 5.0) * (phi_1_2 * phi_2 + phi_2 * phi_2_2) +
            (1.0 / 120.0) * (phi_1_2 * phi_1_2 * phi_2 + 2.0 * phi_1_2 * phi_2 * phi_2_2 + phi_2 * phi_2_2 * phi_2_2)};
  };

  std::array<double, n_components> initial_values{{1.0, 1.0}};
  for (const auto &test_point :
       {Point<dim>(0.0, 0.0), Point<dim>(1.0, 0.0), Point<dim>(0.0, 1.0), Point<dim>(1.0, 1.0), Point<dim>(2.0, 0.0)}) {
    model.initial_condition(test_point, initial_values);
    const auto expected = expanded_phi6_initial_condition(test_point[0], test_point[1]);
    CAPTURE(test_point[0], test_point[1]);
    CHECK(initial_values[0] == Catch::Approx(expected[0]).margin(1.0e-15));
    CHECK(initial_values[1] == Catch::Approx(expected[1]).margin(1.0e-15));
  }

  std::array<Tensor<1, dim, double>, n_components> advection_flux{{Tensor<1, dim, double>(), Tensor<1, dim, double>()}};
  for (auto &flux_i : advection_flux)
    for (uint d = 0; d < dim; ++d)
      flux_i[d] = 1.0;
  model.KurganovTadmor_advection_flux(advection_flux, point, solution);
  for (const auto &flux_i : advection_flux)
    for (uint d = 0; d < dim; ++d)
      CHECK(flux_i[d] == 0.0);

  std::array<Tensor<1, dim, double>, n_components> diffusion_flux{{Tensor<1, dim, double>(), Tensor<1, dim, double>()}};
  for (auto &flux_i : diffusion_flux)
    for (uint d = 0; d < dim; ++d)
      flux_i[d] = 1.0;
  model.flux(diffusion_flux, point, solution);
  CHECK(diffusion_flux[0][0] == 0.5);
  CHECK(diffusion_flux[0][1] == 0.0);
  CHECK(diffusion_flux[1][0] == 0.0);
  CHECK(diffusion_flux[1][1] == 0.5);

  std::array<double, n_components> source{{1.0, 1.0}};
  model.source(source, point, solution);
  for (const auto source_i : source)
    CHECK(source_i == 0.0);
}

TEST_CASE("O2 Model VI_B_III run extracts Gamma2 at the origin", "[2d][FV][KT][O2][VI_B_III][slow]")
{
  const Gamma2 gamma = run_o2_vi_b_iii_gamma_regression();
  log_gamma2("400x400", gamma);
  log_gamma2_reference_comparison("400x400", gamma);

  const double extracted_gamma = extracted_gamma2(gamma);
  const double absolute_error = std::abs(extracted_gamma - gamma_reference);
  const double relative_error = absolute_error / std::abs(gamma_reference);

  CAPTURE(gamma.gamma_11, gamma.gamma_22, gamma.gamma_12_from_u, gamma.gamma_12_from_v, gamma.gamma_12, extracted_gamma,
          absolute_error, relative_error);
  REQUIRE(std::isfinite(gamma.gamma_11));
  REQUIRE(std::isfinite(gamma.gamma_22));
  REQUIRE(std::isfinite(gamma.gamma_12_from_u));
  REQUIRE(std::isfinite(gamma.gamma_12_from_v));
  REQUIRE(std::isfinite(gamma.gamma_12));
  REQUIRE(std::isfinite(extracted_gamma));

  CHECK_THAT(gamma.gamma_11, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(gamma.gamma_22, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(extracted_gamma, Catch::Matchers::WithinAbs(gamma_reference, gamma_tolerance));
  CHECK_THAT(gamma.gamma_11 - gamma.gamma_22, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_u, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_v, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
}

TEST_CASE("O2 Model VI_B_III coarse 40x40 run extracts Gamma2 at the origin", "[2d][FV][KT][O2][VI_B_III][coarse]")
{
  const Gamma2 gamma = run_o2_vi_b_iii_gamma_regression(
      {.n_cells_per_direction = 40, .final_time = default_final_time, .retain_output = false});
  log_gamma2("40x40", gamma);

  CAPTURE(gamma.gamma_11, gamma.gamma_22, gamma.gamma_12_from_u, gamma.gamma_12_from_v, gamma.gamma_12);
  REQUIRE(std::isfinite(gamma.gamma_11));
  REQUIRE(std::isfinite(gamma.gamma_22));
  REQUIRE(std::isfinite(gamma.gamma_12_from_u));
  REQUIRE(std::isfinite(gamma.gamma_12_from_v));
  REQUIRE(std::isfinite(gamma.gamma_12));

  CHECK_THAT(gamma.gamma_11 - gamma.gamma_22, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_u, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12_from_v, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
  CHECK_THAT(gamma.gamma_12, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
}
