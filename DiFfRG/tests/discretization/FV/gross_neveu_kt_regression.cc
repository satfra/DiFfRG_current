#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"

#include <DiFfRG/common/json.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/thermodynamics.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 1;
  constexpr double pi = std::numbers::pi_v<double>;
  constexpr double d_gamma = 2.0;
  constexpr double h = 1.0;
  constexpr double N_flavors = 2.0;
  constexpr double temperature = 0.00625;
  constexpr double beta = powr<-1>(temperature);
  constexpr double chemical_potential = 0.6;
  constexpr double Lambda = 1.0e5;
  constexpr double k_ir = 1.0e-4;
  constexpr double final_time = 20.72326583694641;
  constexpr double sigma_max = 6.0;
  constexpr std::size_t n_cells = 1000;
  constexpr double delta_sigma = sigma_max / static_cast<double>(n_cells - 1);
  constexpr double abs_tol = 5.0e-14;
  constexpr double rel_tol = 2.0e-14;
  constexpr double grid_tol = 1.0e-12;
  constexpr double fixture_derivative_tol = 1.0e-4;

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using NumberType = double;
  using Mesh = RectangularMesh<dim>;
  using Discretization = FV::Discretization<Components, NumberType, Mesh>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  template <typename Model>
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  using ExplicitTimeStepper = TimeStepperBoostRK54<VectorType, SparseMatrixType, dim>;

  void ensure_logger()
  {
    try {
      auto log = spdlog::stdout_color_mt("log");
      log->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  template <typename T> double scalar_value(const T &x) { return static_cast<double>(x); }

  double json_number_to_double(const json::value &value)
  {
    if (value.is_double()) return value.as_double();
    if (value.is_int64()) return static_cast<double>(value.as_int64());
    if (value.is_uint64()) return static_cast<double>(value.as_uint64());
    throw std::runtime_error("Expected JSON number.");
  }

  std::vector<double> json_array_to_doubles(const json::array &values)
  {
    std::vector<double> result;
    result.reserve(values.size());
    for (const auto &value : values)
      result.push_back(json_number_to_double(value));
    return result;
  }

  struct GraphData {
    std::string label;
    std::string ylabel;
    std::vector<double> x;
    std::vector<double> y;
  };

  GraphData parse_graph(const json::object &graph)
  {
    GraphData result;
    result.label = std::string(graph.at("label").as_string().c_str());
    result.ylabel = std::string(graph.at("ylabel").as_string().c_str());
    result.x = json_array_to_doubles(graph.at("x").as_array());
    result.y = json_array_to_doubles(graph.at("y").as_array());
    return result;
  }

  double parse_time_label(const std::string &label)
  {
    if (!label.starts_with("t=")) throw std::runtime_error("Unexpected graph label: " + label);
    return std::stod(label.substr(2));
  }

  std::vector<double> numerical_derivative(const std::vector<double> &x, const std::vector<double> &y)
  {
    if (x.size() != y.size()) throw std::runtime_error("Derivative input size mismatch.");
    if (x.size() < 3) throw std::runtime_error("Need at least three points to compute a derivative.");

    std::vector<double> derivative(y.size(), 0.0);
    derivative.front() = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (x[2] - x[0]);
    for (std::size_t i = 1; i + 1 < y.size(); ++i)
      derivative[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
    derivative.back() =
        (3.0 * y[y.size() - 1] - 4.0 * y[y.size() - 2] + y[y.size() - 3]) / (x[x.size() - 1] - x[x.size() - 3]);
    return derivative;
  }

  std::vector<double> reconstruct_potential_from_u(const std::vector<double> &u_values)
  {
    std::vector<double> U_values(u_values.size(), 0.0);
    for (std::size_t i = 0; i < u_values.size(); ++i) {
      double sum = 0.0;
      for (std::size_t m = 0; m <= i; ++m) {
        double denominator = 1.0;
        if (m == 0) denominator += 1.0;
        if (m == i) denominator += 1.0;
        sum += u_values[m] / denominator;
      }
      U_values[i] = delta_sigma * sum;
    }
    return U_values;
  }

  std::optional<std::string> compare_profiles(const std::string &name, const std::vector<double> &x_values,
                                              const std::vector<double> &simulated,
                                              const std::vector<double> &reference,
                                              const double abs_tolerance = abs_tol,
                                              const double rel_tolerance = rel_tol)
  {
    if (x_values.size() != simulated.size() || x_values.size() != reference.size())
      return std::string("Size mismatch while comparing ") + name + ".";

    std::ostringstream mismatch;
    mismatch << std::setprecision(17);

    std::size_t mismatch_count = 0;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    for (std::size_t i = 0; i < simulated.size(); ++i) {
      const double abs_error = std::abs(simulated[i] - reference[i]);
      const double scale = std::max(1.0, std::abs(reference[i]));
      const double rel_error = abs_error / scale;
      max_abs_error = std::max(max_abs_error, abs_error);
      max_rel_error = std::max(max_rel_error, rel_error);

      if (abs_error <= abs_tolerance || rel_error <= rel_tolerance) continue;

      if (mismatch_count == 0) mismatch << name << " comparison failed.\n";
      if (mismatch_count < 8) {
        mismatch << "  i=" << i << ", sigma=" << x_values[i] << ", simulated=" << simulated[i]
                 << ", reference=" << reference[i] << ", abs_error=" << abs_error << ", rel_error=" << rel_error
                 << '\n';
      }
      ++mismatch_count;
    }

    if (mismatch_count == 0) return std::nullopt;

    mismatch << "  total mismatches=" << mismatch_count << ", max_abs_error=" << max_abs_error
             << ", max_rel_error=" << max_rel_error;
    return mismatch.str();
  }

  struct TemporaryDirectory {
    TemporaryDirectory()
    {
      const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
      path = std::filesystem::temp_directory_path() /
             ("gross_neveu_kt_regression_" + std::to_string(static_cast<std::int64_t>(nonce)));
      std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory()
    {
      std::error_code ec;
      std::filesystem::remove_all(path, ec);
    }

    std::filesystem::path path;
  };

  template <typename Derived>
  class GrossNeveuKTSourceOnlyBase : public def::AbstractModel<Derived, Components>,
                                     public def::fRG,
                                     public def::OriginOddLinearExtrapolationBoundaries<Derived>,
                                     public def::AD<Derived>
  {
  public:
    GrossNeveuKTSourceOnlyBase(const JSONValue &json) : def::fRG(json) {}

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double sigma = x[0];
      const double inverse_sqrt = 1.0 / std::sqrt(1.0 + powr<-2>(Lambda));
      const double prefactor = d_gamma / (2.0 * pi);
      values[0] = prefactor * sigma * (std::atanh(inverse_sqrt) - inverse_sqrt);
    }

    template <int spatial_dim, typename FluxNumberType, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, FluxNumberType>, n_fe_functions> &F_i,
              [[maybe_unused]] const Point<spatial_dim> &x,
              [[maybe_unused]] const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "GrossNeveuKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "GrossNeveuKTModel expects a single FE function.");
      F_i[0][0] = FluxNumberType(0.0);
    }

    template <int spatial_dim, typename SourceNumberType, typename Solutions, std::size_t n_fe_functions>
    void source(std::array<SourceNumberType, n_fe_functions> &s_i, const Point<spatial_dim> &x,
                [[maybe_unused]] const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "GrossNeveuKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "GrossNeveuKTModel expects a single FE function.");

      const double sigma = x[0];
      const double E_f = std::sqrt(k2 + powr<2>(h * sigma));
      const double n_plus = nF(E_f + chemical_potential, temperature);
      const double n_minus = nF(E_f - chemical_potential, temperature);
      const double g = 1.0 - n_plus - n_minus;
      const double dg_dE = beta * (n_plus * (1.0 - n_plus) + n_minus * (1.0 - n_minus));
      const double A = (d_gamma / pi) * (k3 / 2.0);
      const double S = A * sigma * (dg_dE / powr<2>(E_f) - g / powr<3>(E_f));

      // The KT assembler adds source terms to the residual, so the PDE source enters with a minus sign here.
      s_i[0] = static_cast<SourceNumberType>(-S);
    }
  };

  class GrossNeveuKTMeanFieldModel : public GrossNeveuKTSourceOnlyBase<GrossNeveuKTMeanFieldModel>
  {
  public:
    using GrossNeveuKTSourceOnlyBase<GrossNeveuKTMeanFieldModel>::GrossNeveuKTSourceOnlyBase;
  };

  class GrossNeveuKTFluxModel : public GrossNeveuKTSourceOnlyBase<GrossNeveuKTFluxModel>
  {
  public:
    using GrossNeveuKTSourceOnlyBase<GrossNeveuKTFluxModel>::GrossNeveuKTSourceOnlyBase;

    template <int spatial_dim, typename FluxNumberType, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, FluxNumberType>, n_fe_functions> &F_i,
              [[maybe_unused]] const Point<spatial_dim> &x, const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "GrossNeveuKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "GrossNeveuKTModel expects a single FE function.");

      const auto &fe_derivatives = get<1>(sol);
      const FluxNumberType eb_argument = FluxNumberType(k2) + fe_derivatives[0][0];
      const double eb_argument_value = scalar_value(eb_argument);
      if (!(isfinite(eb_argument_value) && eb_argument_value > 0.0))
        throw std::runtime_error("Encountered non-positive k^2 + d_sigma u in Gross-Neveu diffusion flux.");

      using std::sqrt;

      const FluxNumberType E_b = sqrt(eb_argument);
      const FluxNumberType thermal_factor = cothS(E_b, temperature);

      F_i[0][0] = FluxNumberType(-1.0 / (pi * N_flavors)) * FluxNumberType(k3 / 2.0) * thermal_factor / E_b;
    }
  };

  JSONValue make_json()
  {
    return json::value(
        {{"physical", {{"Lambda", Lambda}}},
         {"discretization",
          {{"fe_order", 0},
           {"threads", 1},
           {"batch_size", 64},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"output_buffer_size", 1},
           {"EoM_abs_tol", 1e-10},
           {"EoM_max_iter", 0}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", 1.0},
           {"explicit",
            {{"dt", 1e-8}, {"minimal_dt", 1e-8}, {"maximal_dt", 1.0}, {"abs_tol", 1e-14}, {"rel_tol", 1e-12}}},
           {"implicit",
            {{"dt", 1e-8},
             {"minimal_dt", 1e-8},
             {"maximal_dt", 1.0},
             {"abs_tol", 1e-15},
             {"rel_tol", 1e-15},
             {"max_steps", 5000000}}}}},
         {"output", {{"verbosity", 0}, {"vtk", false}, {"hdf5", false}}}});
  }

  std::filesystem::path repo_root()
  {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path().parent_path();
  }

  std::filesystem::path fixture_path(const std::string &filename)
  {
    const auto local_fixture = std::filesystem::path(__FILE__).parent_path() / "data" / filename;
    if (std::filesystem::exists(local_fixture)) return local_fixture;

    const auto ancillary_fixture = repo_root() / "ancillary_2108_10616" / "extracted" / "anc" / filename;
    if (std::filesystem::exists(ancillary_fixture)) return ancillary_fixture;

    throw std::runtime_error("Could not find regression fixture: " + filename);
  }

  std::pair<GraphData, GraphData> load_reference_ir_pair(const std::filesystem::path &path,
                                                         const double derivative_tolerance = fixture_derivative_tol)
  {
    const double expected_final_time = std::log(Lambda / k_ir);

    const JSONValue fixture_json(path.string());
    json::value fixture_value = static_cast<json::value>(fixture_json);
    const auto &root = fixture_value.as_object();
    const auto &graphs = root.at("graphs").as_array();

    REQUIRE(graphs.size() >= 2);

    const GraphData reference_U = parse_graph(graphs[graphs.size() - 2].as_object());
    const GraphData reference_u = parse_graph(graphs[graphs.size() - 1].as_object());

    REQUIRE(reference_U.x.size() == n_cells);
    REQUIRE(reference_U.y.size() == n_cells);
    REQUIRE(reference_u.x.size() == n_cells);
    REQUIRE(reference_u.y.size() == n_cells);
    REQUIRE(reference_U.x == reference_u.x);
    REQUIRE(parse_time_label(reference_U.label) == Catch::Approx(expected_final_time).margin(1e-12));
    REQUIRE(parse_time_label(reference_u.label) == Catch::Approx(expected_final_time).margin(1e-12));

    for (std::size_t i = 0; i < reference_u.x.size(); ++i)
      REQUIRE(reference_u.x[i] == Catch::Approx(static_cast<double>(i) * delta_sigma).margin(grid_tol));

    const auto reference_derivative = numerical_derivative(reference_U.x, reference_U.y);
    double max_fixture_derivative_error = 0.0;
    for (std::size_t i = 0; i < reference_u.y.size(); ++i)
      max_fixture_derivative_error =
          std::max(max_fixture_derivative_error, std::abs(reference_derivative[i] - reference_u.y[i]));
    REQUIRE(max_fixture_derivative_error < derivative_tolerance);

    return {reference_U, reference_u};
  }

  struct SimulationResult {
    std::vector<double> x;
    std::vector<double> u;
    std::vector<double> U;
  };

  Config::ConfigurationMesh<1> make_mesh_config()
  {
    const Config::GridAxis sigma_axis(-0.5 * delta_sigma, delta_sigma, sigma_max + 0.5 * delta_sigma);
    return Config::ConfigurationMesh<1>(0u, std::vector<Config::GridAxis>{sigma_axis});
  }

  template <typename Model, typename Stepper> SimulationResult run_flow_to_ir(const JSONValue &json)
  {
    Model model(json);
    Mesh mesh(make_mesh_config());
    Discretization discretization(mesh, json);
    Assembler<Model> assembler(discretization, model, json);

    TemporaryDirectory tmp_dir;
    DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), "gross_neveu_kt_regression", "output", json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    Stepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == n_cells);

    std::vector<double> initial_support_points;
    initial_support_points.reserve(support_points.size());
    for (const auto &point : support_points)
      initial_support_points.push_back(point[0]);
    std::sort(initial_support_points.begin(), initial_support_points.end());

    for (std::size_t i = 0; i < initial_support_points.size(); ++i)
      REQUIRE(initial_support_points[i] == Catch::Approx(static_cast<double>(i) * delta_sigma).margin(grid_tol));

    try {
      time_stepper.run(&state, 0.0, final_time);
    } catch (const std::exception &exception) {
      FAIL(std::string("Gross-Neveu KT regression solve failed: ") + exception.what());
    }

    std::vector<std::pair<double, double>> sampled_u;
    sampled_u.reserve(state.spatial_data().size());
    for (unsigned int i = 0; i < state.spatial_data().size(); ++i)
      sampled_u.emplace_back(support_points[i][0], state.spatial_data()[i]);
    std::sort(sampled_u.begin(), sampled_u.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    SimulationResult result;
    result.x.reserve(sampled_u.size());
    result.u.reserve(sampled_u.size());
    for (const auto &[sigma, u_value] : sampled_u) {
      result.x.push_back(sigma);
      result.u.push_back(u_value);
    }
    result.U = reconstruct_potential_from_u(result.u);
    return result;
  }

  template <typename Model> VectorType assemble_uv_residual(const JSONValue &json)
  {
    Model model(json);
    Mesh mesh(make_mesh_config());
    Discretization discretization(mesh, json);
    Assembler<Model> assembler(discretization, model, json);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    VectorType uv_residual(state.spatial_data().size());
    VectorType uv_dt(state.spatial_data().size());
    uv_residual = 0.0;
    uv_dt = 0.0;
    assembler.residual(uv_residual, state.spatial_data(), 1.0, uv_dt, 1.0);
    return uv_residual;
  }
} // namespace

TEST_CASE("KT Gross-Neveu source-only regression matches the MF IR flow at T=0.00625, mu=0.6",
          "[FV][KT][GrossNeveu][MF][regression]")
{
  ensure_logger();
  constexpr double mf_profile_abs_tol = 1.0e-6;
  constexpr double mf_profile_rel_tol = 1.0e-7;

  const auto [reference_U, reference_u] =
      load_reference_ir_pair(fixture_path("flow_MF_T=0.00625,mu=0.6.json"), 1.0e-3);
  const JSONValue json = make_json();
  const auto simulation = run_flow_to_ir<GrossNeveuKTMeanFieldModel, ExplicitTimeStepper>(json);

  for (std::size_t i = 0; i < simulation.x.size(); ++i)
    REQUIRE(simulation.x[i] == Catch::Approx(reference_u.x[i]).margin(grid_tol));

  const auto u_mismatch =
      compare_profiles("u(sigma)", simulation.x, simulation.u, reference_u.y, mf_profile_abs_tol, mf_profile_rel_tol);
  if (u_mismatch) FAIL(*u_mismatch);

  const auto U_mismatch =
      compare_profiles("U(sigma)", simulation.x, simulation.U, reference_U.y, mf_profile_abs_tol, mf_profile_rel_tol);
  if (U_mismatch) FAIL(*U_mismatch);
}

TEST_CASE("KT Gross-Neveu flux model extends the source-only model with a non-zero UV transport residual",
          "[FV][KT][GrossNeveu][flux]")
{
  ensure_logger();

  const JSONValue json = make_json();
  const VectorType source_only_residual = assemble_uv_residual<GrossNeveuKTMeanFieldModel>(json);
  const VectorType flux_residual = assemble_uv_residual<GrossNeveuKTFluxModel>(json);

  REQUIRE(source_only_residual.size() == flux_residual.size());

  double max_abs_difference = 0.0;
  for (unsigned int i = 0; i < source_only_residual.size(); ++i) {
    REQUIRE(std::isfinite(source_only_residual[i]));
    REQUIRE(std::isfinite(flux_residual[i]));
    max_abs_difference = std::max(max_abs_difference, std::abs(flux_residual[i] - source_only_residual[i]));
  }

  REQUIRE(max_abs_difference > 1.0e-8);
}
