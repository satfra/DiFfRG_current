#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/timestepping/timestepping.hh"
#include "kt_regression_helpers.hh"

#include <DiFfRG/common/json.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>
#include <autodiff/forward/real/real.hpp>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;

  constexpr uint dim = 2;
  constexpr double pi = std::numbers::pi_v<double>;
  constexpr double missing_runtime_value = std::numeric_limits<double>::quiet_NaN();

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMesh<dim>;
  using Discretization = FV::Discretization<Components, NumberType, Mesh>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;

  struct PaperBenchmarkResult {
    bool completed = false;
    std::string failure;
    double gamma_11 = missing_runtime_value;
    double gamma_22 = missing_runtime_value;
    double gamma_12 = missing_runtime_value;
    double min_phi_1 = missing_runtime_value;
    double min_phi_2 = missing_runtime_value;
  };

  struct PaperBenchmarkGrid {
    std::vector<double> phi_1;
    std::vector<double> phi_2;
    std::vector<double> u;
    std::vector<double> v;
    std::size_t nx = 0;
    std::size_t ny = 0;

    std::size_t index(const std::size_t ix, const std::size_t iy) const { return iy * nx + ix; }
  };

  enum class PaperBenchmarkKind {
    ZeroDimO2,
    ZeroDimNonO2,
    ZeroDimRotatedPyramid,
    ZeroDimONxOM,
  };

  struct PaperBenchmark {
    std::string_view id;
    PaperBenchmarkKind kind;
    std::string_view paper_reference;
    double phi_max;
    double lambda;
    double t_ir;
    double rtol;
    double atol;
    double exact_gamma_11 = missing_runtime_value;
    double exact_gamma_22 = missing_runtime_value;
    double exact_gamma_12 = missing_runtime_value;
    double exact_min_phi_1 = missing_runtime_value;
    double exact_min_phi_2 = missing_runtime_value;
    double rotation_angle = 0.0;
    double n_bar = missing_runtime_value;
    double m_bar = missing_runtime_value;
  };

  constexpr PaperBenchmark o2_case_i{"I", PaperBenchmarkKind::ZeroDimO2, "Eqs. (48), (55), Table I", 10.0, 1.0e12,
                                     60.0, 1.0e-10, 1.0e-12,
                                     0.2957022746, 0.2957022746};
  constexpr PaperBenchmark o2_case_ii{"II", PaperBenchmarkKind::ZeroDimO2, "Eqs. (49), (55), Table I", 10.0, 1.0e12,
                                      60.0, 1.0e-10, 1.0e-12,
                                      0.3163677894, 0.3163677894};
  constexpr PaperBenchmark o2_case_iii{"III", PaperBenchmarkKind::ZeroDimO2, "Eqs. (50), (55), Table I", 10.0, 1.0e12,
                                       60.0, 1.0e-10, 1.0e-12,
                                       0.1786698196, 0.1786698196};
  constexpr PaperBenchmark o2_case_iv{"IV", PaperBenchmarkKind::ZeroDimO2, "Eqs. (51), (55), Table I", 10.0, 1.0e12,
                                      60.0, 1.0e-10, 1.0e-12,
                                      0.3214613360, 0.3214613360};

  constexpr PaperBenchmark non_o2_case_v{"V",
                                         PaperBenchmarkKind::ZeroDimNonO2,
                                         "Eqs. (52), (53), (54)",
                                         6.0,
                                         1.0e12,
                                         60.0,
                                         1.0e-12,
                                         1.0e-12,
                                         57.0873196170,
                                         14.1514431655,
                                         -9.3081326369,
                                         -2.0409061306,
                                         0.3306875296};

  constexpr PaperBenchmark pyramid_case_vi_alpha_0{"VI alpha=0",
                                                   PaperBenchmarkKind::ZeroDimRotatedPyramid,
                                                   "Eqs. (56), (57)",
                                                   10.0,
                                                   1.0e12,
                                                   60.0,
                                                   1.0e-10,
                                                   1.0e-12,
                                                   0.3287806584,
                                                   0.3287806584,
                                                   0.0,
                                                   missing_runtime_value,
                                                   missing_runtime_value,
                                                   0.0};
  constexpr PaperBenchmark pyramid_case_vi_alpha_03{"VI alpha=0.3",
                                                    PaperBenchmarkKind::ZeroDimRotatedPyramid,
                                                    "Eqs. (56), (57)",
                                                    10.0,
                                                    1.0e12,
                                                    60.0,
                                                    1.0e-10,
                                                    1.0e-12,
                                                    0.3287806584,
                                                    0.3287806584,
                                                    0.0,
                                                    missing_runtime_value,
                                                    missing_runtime_value,
                                                    0.3};
  constexpr PaperBenchmark pyramid_case_vi_alpha_05{"VI alpha=0.5",
                                                    PaperBenchmarkKind::ZeroDimRotatedPyramid,
                                                    "Eqs. (56), (57)",
                                                    10.0,
                                                    1.0e12,
                                                    60.0,
                                                    1.0e-10,
                                                    1.0e-12,
                                                    0.3287806584,
                                                    0.3287806584,
                                                    0.0,
                                                    missing_runtime_value,
                                                    missing_runtime_value,
                                                    0.5};
  constexpr PaperBenchmark pyramid_case_vi_alpha_pi4{"VI alpha=pi/4",
                                                     PaperBenchmarkKind::ZeroDimRotatedPyramid,
                                                     "Eqs. (56), (57)",
                                                     10.0,
                                                     1.0e12,
                                                     60.0,
                                                     1.0e-10,
                                                     1.0e-12,
                                                     0.3287806584,
                                                     0.3287806584,
                                                     0.0,
                                                     missing_runtime_value,
                                                     missing_runtime_value,
                                                     pi / 4.0};

  constexpr PaperBenchmark on_om_case_vii{"VII",
                                          PaperBenchmarkKind::ZeroDimONxOM,
                                          "Eqs. (32), (33), (58), (59)",
                                          7.0,
                                          1.0e12,
                                          60.0,
                                          1.0e-12,
                                          1.0e-12,
                                          0.3703505404,
                                          0.3726964081,
                                          0.0,
                                          missing_runtime_value,
                                          missing_runtime_value,
                                          0.0,
                                          2.0,
                                          3.0};

  void ensure_logger()
  {
    try {
      auto log = spdlog::stdout_color_mt("log");
      log->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  JSONValue make_paper_json(const PaperBenchmark &benchmark, const double step,
                            const double final_time = missing_runtime_value)
  {
    const double flow_final_time = std::isfinite(final_time) ? final_time : benchmark.t_ir;
    const auto grid = std::to_string(-benchmark.phi_max) + ":" + std::to_string(step) + ":" +
                      std::to_string(benchmark.phi_max);
    return json::value({{"physical", {{"Lambda", benchmark.lambda}}},
                        {"discretization",
                         {{"fe_order", 0},
                          {"threads", 1},
                          {"batch_size", 16},
                          {"overintegration", 0},
                          {"output_subdivisions", 1},
                          {"EoM_abs_tol", 1.0e-10},
                          {"EoM_max_iter", 0},
                          {"grid", {{"x_grid", grid}, {"y_grid", grid}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
                        {"timestepping",
                         {{"final_time", flow_final_time},
                          {"output_dt", flow_final_time},
                          {"explicit",
                           {{"dt", 1.0e-3},
                            {"minimal_dt", 1.0e-10},
                            {"maximal_dt", 1.0},
                            {"abs_tol", benchmark.atol},
                            {"rel_tol", benchmark.rtol}}},
                          {"implicit",
                           {{"dt", 1.0e-3},
                            {"minimal_dt", 1.0e-10},
                            {"maximal_dt", 1.0},
                            {"abs_tol", benchmark.atol},
                            {"rel_tol", benchmark.rtol}}}}},
                        {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", false}}}});
  }

  template <typename T> T square(const T value) { return value * value; }

  template <typename T> T cube(const T value) { return value * value * value; }

  template <typename T> T heaviside(const T value) { return value > T(0.0) ? T(1.0) : T(0.0); }

  template <typename T> T paper_uv_potential(const PaperBenchmark &benchmark, const T phi_1, const T phi_2)
  {
    const T r2 = square(phi_1) + square(phi_2);
    const T radius = std::sqrt(r2);

    switch (benchmark.kind) {
      case PaperBenchmarkKind::ZeroDimO2:
        if (benchmark.id == std::string_view("I")) {
          if (radius <= T(2.0)) return T(-0.5) * r2;
          if (radius <= T(3.0)) return T(-2.0);
          return T(0.5) * (r2 - T(13.0));
        }
        if (benchmark.id == std::string_view("II"))
          return T(-0.5) * r2 + square(r2) / T(24.0);
        if (benchmark.id == std::string_view("III"))
          return T(0.5) * r2 - square(r2) / T(20.0) + cube(r2) / T(720.0);
        if (benchmark.id == std::string_view("IV")) {
          if (radius <= std::sqrt(T(8.0))) return -std::pow(r2, T(1.0) / T(3.0));
          return T(0.5) * r2 - T(6.0);
        }
        break;

      case PaperBenchmarkKind::ZeroDimNonO2:
        if (radius <= T(3.0))
          return T(2.0) * (cube(phi_1) + phi_1 * phi_2) * (std::cos(T(pi) * r2 / T(9.0)) + T(1.0));
        return r2 - T(9.0);

      case PaperBenchmarkKind::ZeroDimRotatedPyramid: {
        const T theta_1 = std::cos(T(benchmark.rotation_angle)) * phi_1 +
                          std::sin(T(benchmark.rotation_angle)) * phi_2;
        const T theta_2 = -std::sin(T(benchmark.rotation_angle)) * phi_1 +
                          std::cos(T(benchmark.rotation_angle)) * phi_2;
        return T(5.0) * (T(2.0) - std::abs(theta_1) - std::abs(theta_2)) *
                   heaviside(T(2.0) - std::abs(theta_1) - std::abs(theta_2)) +
               heaviside(r2 - T(9.0)) * (r2 - T(9.0));
      }

      case PaperBenchmarkKind::ZeroDimONxOM: {
        const T rho_1 = T(0.5) * square(phi_1);
        const T rho_2 = T(0.5) * square(phi_2);
        return T(4.0) * rho_1 * square(rho_2) * std::sin(T(2.0 * pi / 9.0) * (rho_1 + rho_2)) *
                   heaviside(T(4.5) - rho_1 - rho_2) +
               T(2.0) * (rho_1 + rho_2 - T(8.0)) * heaviside(rho_1 + rho_2 - T(8.0));
      }

    }

    throw std::runtime_error("Paper benchmark has no encoded UV potential.");
  }

  double integrate_potential_jump_over_y(const PaperBenchmark &benchmark, const double x_left, const double x_right,
                                         const double y_center, const double cell_width)
  {
    // 16-point Gauss-Legendre quadrature for Eq. (43), the finite-volume cell average.
    static constexpr std::array<double, 8> nodes{0.09501250983763744, 0.28160355077925891, 0.45801677765722739,
                                                 0.61787624440264375, 0.75540440835500303, 0.86563120238783174,
                                                 0.94457502307323258, 0.98940093499164993};
    static constexpr std::array<double, 8> weights{0.18945061045506850, 0.18260341504492359, 0.16915651939500254,
                                                   0.14959598881657673, 0.12462897125553388, 0.09515851168249278,
                                                   0.06225352393864789, 0.02715245941175409};

    const double half_width = 0.5 * cell_width;
    double integral = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const double offset = half_width * nodes[i];
      for (const double y : {y_center - offset, y_center + offset}) {
        integral += weights[i] *
                    (paper_uv_potential(benchmark, x_right, y) - paper_uv_potential(benchmark, x_left, y));
      }
    }
    return half_width * integral / (cell_width * cell_width);
  }

  double integrate_potential_jump_over_x(const PaperBenchmark &benchmark, const double x_center, const double y_bottom,
                                         const double y_top, const double cell_width)
  {
    static constexpr std::array<double, 8> nodes{0.09501250983763744, 0.28160355077925891, 0.45801677765722739,
                                                 0.61787624440264375, 0.75540440835500303, 0.86563120238783174,
                                                 0.94457502307323258, 0.98940093499164993};
    static constexpr std::array<double, 8> weights{0.18945061045506850, 0.18260341504492359, 0.16915651939500254,
                                                   0.14959598881657673, 0.12462897125553388, 0.09515851168249278,
                                                   0.06225352393864789, 0.02715245941175409};

    const double half_width = 0.5 * cell_width;
    double integral = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const double offset = half_width * nodes[i];
      for (const double x : {x_center - offset, x_center + offset}) {
        integral += weights[i] *
                    (paper_uv_potential(benchmark, x, y_top) - paper_uv_potential(benchmark, x, y_bottom));
      }
    }
    return half_width * integral / (cell_width * cell_width);
  }

  std::array<double, 2> paper_initial_cell_average(const PaperBenchmark &benchmark, const Point<2> &pos,
                                                   const double cell_width)
  {
    const double x_left = pos[0] - 0.5 * cell_width;
    const double x_right = pos[0] + 0.5 * cell_width;
    const double y_bottom = pos[1] - 0.5 * cell_width;
    const double y_top = pos[1] + 0.5 * cell_width;

    return {integrate_potential_jump_over_y(benchmark, x_left, x_right, pos[1], cell_width),
            integrate_potential_jump_over_x(benchmark, pos[0], y_bottom, y_top, cell_width)};
  }

  double uniform_spacing(const std::vector<double> &values, const std::string_view axis)
  {
    if (values.size() < 3) {
      std::ostringstream message;
      message << "Cannot extract paper observables: " << axis << " axis has fewer than three support points.";
      throw std::runtime_error(message.str());
    }

    const double spacing = values[1] - values[0];
    if (!(spacing > 0.0)) {
      std::ostringstream message;
      message << "Cannot extract paper observables: " << axis << " axis is not strictly increasing.";
      throw std::runtime_error(message.str());
    }

    for (std::size_t i = 2; i < values.size(); ++i) {
      const double current_spacing = values[i] - values[i - 1];
      if (std::abs(current_spacing - spacing) > 1.0e-10 * std::max(1.0, std::abs(spacing))) {
        std::ostringstream message;
        message << "Cannot extract paper observables: " << axis << " axis is not uniformly spaced.";
        throw std::runtime_error(message.str());
      }
    }
    return spacing;
  }

  std::size_t lower_bound_index(const std::vector<double> &values, const double value)
  {
    const auto iterator = std::lower_bound(values.begin(), values.end(), value);
    if (iterator == values.end()) return values.size() - 1;
    return static_cast<std::size_t>(iterator - values.begin());
  }

  std::size_t nearest_index(const std::vector<double> &values, const double value)
  {
    const std::size_t upper = lower_bound_index(values, value);
    if (upper == 0) return 0;
    if (upper >= values.size()) return values.size() - 1;
    const std::size_t lower = upper - 1;
    return std::abs(values[lower] - value) <= std::abs(values[upper] - value) ? lower : upper;
  }

  PaperBenchmarkGrid sample_final_grid(const FV::FlowingVariables<Discretization> &state,
                                       const Discretization &discretization)
  {
    constexpr std::size_t n_components = 2;
    const auto &support_points = discretization.get_support_points();
    const auto &solution = state.spatial_data();

    if (support_points.size() != solution.size())
      throw std::runtime_error("Cannot sample 2D paper benchmark: support point/data size mismatch.");
    if (solution.size() % n_components != 0)
      throw std::runtime_error("Cannot sample 2D paper benchmark: solution size is not divisible by two components.");

    std::vector<double> x_values;
    std::vector<double> y_values;
    const std::size_t n_points = solution.size() / n_components;
    x_values.reserve(n_points);
    y_values.reserve(n_points);

    constexpr double point_tol = 1.0e-12;
    for (std::size_t point = 0; point < n_points; ++point) {
      const std::size_t base = point * n_components;
      for (std::size_t component = 1; component < n_components; ++component) {
        if (support_points[base].distance(support_points[base + component]) > point_tol)
          throw std::runtime_error("Cannot sample 2D paper benchmark: component support points are not grouped.");
      }
      x_values.push_back(support_points[base][0]);
      y_values.push_back(support_points[base][1]);
    }

    std::sort(x_values.begin(), x_values.end());
    x_values.erase(std::unique(x_values.begin(), x_values.end(),
                               [](const double lhs, const double rhs) { return std::abs(lhs - rhs) < point_tol; }),
                   x_values.end());
    std::sort(y_values.begin(), y_values.end());
    y_values.erase(std::unique(y_values.begin(), y_values.end(),
                               [](const double lhs, const double rhs) { return std::abs(lhs - rhs) < point_tol; }),
                   y_values.end());

    PaperBenchmarkGrid grid;
    grid.phi_1 = std::move(x_values);
    grid.phi_2 = std::move(y_values);
    grid.nx = grid.phi_1.size();
    grid.ny = grid.phi_2.size();
    if (grid.nx * grid.ny != n_points)
      throw std::runtime_error("Cannot sample 2D paper benchmark: support points do not form a rectangular grid.");

    grid.u.assign(n_points, missing_runtime_value);
    grid.v.assign(n_points, missing_runtime_value);

    for (std::size_t point = 0; point < n_points; ++point) {
      const std::size_t base = point * n_components;
      const std::size_t ix = nearest_index(grid.phi_1, support_points[base][0]);
      const std::size_t iy = nearest_index(grid.phi_2, support_points[base][1]);
      const std::size_t grid_index = grid.index(ix, iy);
      if (std::isfinite(grid.u[grid_index]) || std::isfinite(grid.v[grid_index]))
        throw std::runtime_error("Cannot sample 2D paper benchmark: duplicate grid support point.");
      grid.u[grid_index] = solution[base];
      grid.v[grid_index] = solution[base + 1];
    }

    for (std::size_t i = 0; i < n_points; ++i) {
      if (!std::isfinite(grid.u[i]) || !std::isfinite(grid.v[i]))
        throw std::runtime_error("Cannot sample 2D paper benchmark: final state contains non-finite values.");
    }

    uniform_spacing(grid.phi_1, "phi_1");
    uniform_spacing(grid.phi_2, "phi_2");
    return grid;
  }

  PaperBenchmarkResult extract_observables(const PaperBenchmarkGrid &grid)
  {
    PaperBenchmarkResult result;

    double best_norm = std::numeric_limits<double>::infinity();
    std::size_t min_ix = 0;
    std::size_t min_iy = 0;
    for (std::size_t iy = 0; iy < grid.ny; ++iy) {
      for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const std::size_t i = grid.index(ix, iy);
        const double norm = std::hypot(grid.u[i], grid.v[i]);
        if (norm < best_norm) {
          best_norm = norm;
          min_ix = ix;
          min_iy = iy;
        }
      }
    }

    if (min_ix == 0 || min_iy == 0 || min_ix + 1 >= grid.nx || min_iy + 1 >= grid.ny) {
      std::ostringstream message;
      message << "Cannot extract paper observables: minimum candidate is on the boundary at phi=("
              << grid.phi_1[min_ix] << ", " << grid.phi_2[min_iy] << ").";
      throw std::runtime_error(message.str());
    }

    const double dx = uniform_spacing(grid.phi_1, "phi_1");
    const double dy = uniform_spacing(grid.phi_2, "phi_2");

    result.min_phi_1 = grid.phi_1[min_ix];
    result.min_phi_2 = grid.phi_2[min_iy];
    result.gamma_11 =
        (grid.u[grid.index(min_ix + 1, min_iy)] - grid.u[grid.index(min_ix - 1, min_iy)]) / (2.0 * dx);
    result.gamma_22 =
        (grid.v[grid.index(min_ix, min_iy + 1)] - grid.v[grid.index(min_ix, min_iy - 1)]) / (2.0 * dy);

    const double du_dphi_2 =
        (grid.u[grid.index(min_ix, min_iy + 1)] - grid.u[grid.index(min_ix, min_iy - 1)]) / (2.0 * dy);
    const double dv_dphi_1 =
        (grid.v[grid.index(min_ix + 1, min_iy)] - grid.v[grid.index(min_ix - 1, min_iy)]) / (2.0 * dx);
    result.gamma_12 = 0.5 * (du_dphi_2 + dv_dphi_1);

    result.completed = true;
    return result;
  }

  class PaperBenchmarkModel : public def::AbstractModel<PaperBenchmarkModel, Components>,
                              public def::fRG,
                              public def::LLFFlux<PaperBenchmarkModel>,
                              public def::FlowBoundaries<PaperBenchmarkModel>,
                              public def::AD<PaperBenchmarkModel>
  {
  public:
    PaperBenchmarkModel(const PaperBenchmark &benchmark, const JSONValue &json, const double cell_width)
        : def::fRG(json), benchmark(benchmark), cell_width(cell_width)
    {
    }

    void set_time(const double t_)
    {
      this->t = t_;
      k = std::exp(-this->t) * this->Lambda;
    }

    template <typename Vector> void initial_condition(const Point<2> &pos, Vector &values) const
    {
      const auto initial_average = paper_initial_cell_average(benchmark, pos, cell_width);
      values[0] = initial_average[0];
      values[1] = initial_average[1];
    }

    template <typename NT, typename Solution>
    void KurganovTadmor_advection_flux(std::array<Tensor<1, 2, NT>, 2> &F_i, const Point<2> &pos,
                                       const Solution &sol) const
    {
      F_i[0] = Tensor<1, 2, NT>();
      F_i[1] = Tensor<1, 2, NT>();

      if (benchmark.kind != PaperBenchmarkKind::ZeroDimONxOM)
        return;

      const auto &fe_functions = get<"fe_functions">(sol);
      const NT u = fe_functions[0];
      const NT v = fe_functions[1];
      const NT regulator = regulator_mass<NT>();
      const NT regulator_insertion = regulator_prefactor<NT>();
      const NT sigma_1 = NT(pos[0]);
      const NT sigma_2 = NT(pos[1]);
      const NT u_over_sigma_1 = std::abs(pos[0]) < 1.0e-12 ? NT(0.0) : u / sigma_1;
      const NT v_over_sigma_2 = std::abs(pos[1]) < 1.0e-12 ? NT(0.0) : v / sigma_2;
      const NT advection = NT(benchmark.n_bar - 1.0) * regulator_insertion / (regulator + u_over_sigma_1) +
                           NT(benchmark.m_bar - 1.0) * regulator_insertion / (regulator + v_over_sigma_2);

      // Paper Eqs. (32)-(33), with the same sign convention as the existing
      // one-dimensional KT models.
      F_i[0][0] = advection;
      F_i[1][1] = advection;
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, 2, NT>, 2> &F_i, const Point<2> & /*pos*/, const Solution &sol) const
    {
      const auto &du = get<"fe_derivatives">(sol);
      const NT regulator = regulator_mass<NT>();
      const NT trace = du[0][0] + du[1][1];
      const NT determinant = (regulator + du[0][0]) * (regulator + du[1][1]) - du[1][0] * du[0][1];
      const NT q = regulator_prefactor<NT>() * (NT(2.0) * regulator + trace) / determinant;

      // Paper Eq. (16) for zero-dimensional two-field models and Eq. (32) for
      // O(Nbar)xO(Mbar).
      F_i[0][0] = q;
      F_i[0][1] = NT(0.0);
      F_i[1][0] = NT(0.0);
      F_i[1][1] = q;
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, 2> &s_i, const Point<2> & /*pos*/, const Solution & /*sol*/) const
    {
      s_i[0] = NT(0.0);
      s_i[1] = NT(0.0);
    }

    template <int mdim, typename NT, std::size_t n_components>
    bool apply_boundary_stencil(def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                                def::BoundaryStencilPoints<mdim> &x_stencil,
                                const Point<mdim> &x_face) const
    {
      static_assert(mdim == 2, "The paper benchmark model is intentionally two-dimensional.");
      static_assert(n_components == 2, "The paper benchmark model has two field-space flow components.");

      using namespace def::BoundaryStencilIndex;
      unsigned int axis = 0;
      double max_distance = std::abs(x_face[0] - x_stencil[physical_cell][0]);
      for (unsigned int d = 1; d < mdim; ++d) {
        const double distance = std::abs(x_face[d] - x_stencil[physical_cell][d]);
        if (distance > max_distance) {
          axis = d;
          max_distance = distance;
        }
      }

      const bool lower_boundary = x_face[axis] <= x_stencil[physical_cell][axis];
      const auto apply_affine = [&]() {
        const double delta = lower_boundary ? (x_stencil[upper_inner][axis] - x_stencil[physical_cell][axis])
                                            : (x_stencil[physical_cell][axis] - x_stencil[lower_inner][axis]);
        if (lower_boundary) {
          x_stencil[lower_inner] = x_stencil[physical_cell];
          x_stencil[lower_outer] = x_stencil[physical_cell];
          x_stencil[lower_inner][axis] = x_stencil[physical_cell][axis] - delta;
          x_stencil[lower_outer][axis] = x_stencil[physical_cell][axis] - 2.0 * delta;
          for (std::size_t c = 0; c < n_components; ++c) {
            u_stencil[lower_inner][c] = NT(2.0) * u_stencil[physical_cell][c] - u_stencil[upper_inner][c];
            u_stencil[lower_outer][c] = NT(3.0) * u_stencil[physical_cell][c] - NT(2.0) * u_stencil[upper_inner][c];
          }
          return;
        }

        x_stencil[upper_inner] = x_stencil[physical_cell];
        x_stencil[upper_outer] = x_stencil[physical_cell];
        x_stencil[upper_inner][axis] = x_stencil[physical_cell][axis] + delta;
        x_stencil[upper_outer][axis] = x_stencil[physical_cell][axis] + 2.0 * delta;
        for (std::size_t c = 0; c < n_components; ++c) {
          u_stencil[upper_inner][c] = NT(2.0) * u_stencil[physical_cell][c] - u_stencil[lower_inner][c];
          u_stencil[upper_outer][c] = NT(3.0) * u_stencil[physical_cell][c] - NT(2.0) * u_stencil[lower_inner][c];
        }
      };

      if (benchmark.kind != PaperBenchmarkKind::ZeroDimONxOM || !lower_boundary) {
        apply_affine();
        return true;
      }

      const double delta = x_stencil[upper_inner][axis] - x_stencil[physical_cell][axis];
      x_stencil[lower_inner] = x_stencil[physical_cell];
      x_stencil[lower_outer] = x_stencil[physical_cell];
      x_stencil[lower_inner][axis] = x_stencil[physical_cell][axis] - delta;
      x_stencil[lower_outer][axis] = x_stencil[physical_cell][axis] - 2.0 * delta;

      const std::size_t odd_component = axis;
      const std::size_t even_component = 1U - axis;
      u_stencil[lower_inner][odd_component] = -u_stencil[upper_inner][odd_component];
      u_stencil[lower_outer][odd_component] = -u_stencil[upper_outer][odd_component];
      u_stencil[physical_cell][odd_component] = NT(0.0);
      u_stencil[lower_inner][even_component] = u_stencil[physical_cell][even_component];
      u_stencil[lower_outer][even_component] = u_stencil[upper_inner][even_component];
      return true;
    }

    const PaperBenchmark benchmark;

  private:
    template <typename NT> NT regulator_mass() const
    {
      return NT(k);
    }

    template <typename NT> NT regulator_prefactor() const
    {
      return NT(0.5 * k);
    }

    double cell_width;
  };

  PaperBenchmarkResult run_paper_benchmark(const PaperBenchmark &benchmark, const double step)
  {
    ensure_logger();
    PaperBenchmarkResult result;
    try {
      auto json = make_paper_json(benchmark, step);
      PaperBenchmarkModel model(benchmark, json, step);
      Config::ConfigurationMesh<dim> mesh_config(json);
      Mesh mesh(mesh_config);
      Discretization discretization(mesh, json);
      FV::KurganovTadmor::Assembler<Discretization, PaperBenchmarkModel> assembler(discretization, model, json);

      kt_regression::TemporaryDirectory tmp_dir("zero_dim_2d_kt_regression");
      DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), "zero_dim_2d_kt_regression", "output", json);
      auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
      ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

      FV::FlowingVariables<Discretization> state(discretization);
      state.interpolate(model);

      time_stepper.run(&state, 0.0, benchmark.t_ir);
      return extract_observables(sample_final_grid(state, discretization));
    } catch (const std::exception &exception) {
      std::ostringstream message;
      message << "Paper benchmark " << benchmark.id << " did not complete the real KT timestepper/extraction path: "
              << exception.what();
      result.failure = message.str();
    }
    return result;
  }

  PaperBenchmarkResult run_paper_benchmark(const PaperBenchmark &benchmark)
  {
    return run_paper_benchmark(benchmark, 0.5);
  }

  void check_reference_value(const std::string_view observable, const double actual, const double expected,
                             const double tolerance)
  {
    CAPTURE(observable, actual, expected, tolerance);
    REQUIRE(std::isfinite(actual));
    CHECK_THAT(actual, Catch::Matchers::WithinAbs(expected, tolerance));
  }

  void require_paper_benchmark_matches_reference(const PaperBenchmark &benchmark, const double step)
  {
    PaperBenchmarkResult result;
    CAPTURE(benchmark.id, benchmark.paper_reference, step);
    result = run_paper_benchmark(benchmark, step);
    if (!result.completed) {
      const std::string failure_message =
          result.failure.empty() ? "Paper benchmark did not complete." : result.failure;
      FAIL(failure_message);
    }

    constexpr double gamma_tolerance = 5.0e-3;
    constexpr double minimum_tolerance = 5.0e-3;

    if (std::isfinite(benchmark.exact_gamma_11))
      check_reference_value("Gamma_11", result.gamma_11, benchmark.exact_gamma_11, gamma_tolerance);
    if (std::isfinite(benchmark.exact_gamma_22))
      check_reference_value("Gamma_22", result.gamma_22, benchmark.exact_gamma_22, gamma_tolerance);
    if (std::isfinite(benchmark.exact_gamma_12))
      check_reference_value("Gamma_12", result.gamma_12, benchmark.exact_gamma_12, gamma_tolerance);
    if (std::isfinite(benchmark.exact_min_phi_1))
      check_reference_value("min_phi_1", result.min_phi_1, benchmark.exact_min_phi_1, minimum_tolerance);
    if (std::isfinite(benchmark.exact_min_phi_2))
      check_reference_value("min_phi_2", result.min_phi_2, benchmark.exact_min_phi_2, minimum_tolerance);

    if (benchmark.kind == PaperBenchmarkKind::ZeroDimO2) {
      check_reference_value("O(2) off-diagonal Gamma_12", result.gamma_12, 0.0, gamma_tolerance);
      CAPTURE(result.gamma_11, result.gamma_22, gamma_tolerance);
      REQUIRE(std::isfinite(result.gamma_11));
      REQUIRE(std::isfinite(result.gamma_22));
      CHECK_THAT(result.gamma_11 - result.gamma_22, Catch::Matchers::WithinAbs(0.0, gamma_tolerance));
    }
  }

  void require_paper_benchmark_matches_reference(const PaperBenchmark &benchmark)
  {
    require_paper_benchmark_matches_reference(benchmark, 0.5);
  }

  PaperBenchmarkModel make_boundary_test_model(const PaperBenchmark &benchmark)
  {
    auto json = make_paper_json(benchmark, 0.5);
    return PaperBenchmarkModel(benchmark, json, 0.5);
  }

  double paper_reference_error(const PaperBenchmark &benchmark, const PaperBenchmarkResult &result)
  {
    double error = 0.0;
    if (std::isfinite(benchmark.exact_gamma_11))
      error = std::max(error, std::abs(result.gamma_11 - benchmark.exact_gamma_11));
    if (std::isfinite(benchmark.exact_gamma_22))
      error = std::max(error, std::abs(result.gamma_22 - benchmark.exact_gamma_22));
    if (std::isfinite(benchmark.exact_gamma_12))
      error = std::max(error, std::abs(result.gamma_12 - benchmark.exact_gamma_12));
    if (std::isfinite(benchmark.exact_min_phi_1))
      error = std::max(error, std::abs(result.min_phi_1 - benchmark.exact_min_phi_1));
    if (std::isfinite(benchmark.exact_min_phi_2))
      error = std::max(error, std::abs(result.min_phi_2 - benchmark.exact_min_phi_2));
    return error;
  }

  void require_completed_paper_run(const PaperBenchmark &benchmark, const double step, PaperBenchmarkResult &result)
  {
    CAPTURE(benchmark.id, benchmark.paper_reference, step);
    result = run_paper_benchmark(benchmark, step);
    if (!result.completed) {
      const std::string failure_message =
          result.failure.empty() ? "Paper benchmark did not complete." : result.failure;
      FAIL(failure_message);
    }
  }

  void run_paper_convergence_check(const PaperBenchmark &benchmark)
  {
    PaperBenchmarkResult coarse;
    PaperBenchmarkResult fine;
    require_completed_paper_run(benchmark, 0.5, coarse);
    require_completed_paper_run(benchmark, 0.25, fine);

    const double coarse_error = paper_reference_error(benchmark, coarse);
    const double fine_error = paper_reference_error(benchmark, fine);
    CAPTURE(benchmark.id, coarse_error, fine_error);
    REQUIRE(std::isfinite(coarse_error));
    REQUIRE(std::isfinite(fine_error));
    CHECK(fine_error < coarse_error);
  }

  template <typename NT>
  using PaperFluxGradient = std::array<Tensor<1, 2, NT>, 2>;

  template <typename NT>
  PaperFluxGradient<NT> evaluate_paper_diffusion_flux(const PaperBenchmarkModel &model,
                                                      const std::array<NT, 2> &fe_values,
                                                      const PaperFluxGradient<NT> &gradient)
  {
    PaperFluxGradient<NT> flux{};
    auto sol = FV::KurganovTadmor::internal::flux_tie(fe_values, gradient);
    model.flux(flux, Point<2>(0.25, -0.5), sol);
    return flux;
  }

  double analytic_jacobian_entry(const SparseMatrixType &matrix, const int row, const int column)
  {
    return matrix.get_sparsity_pattern().exists(row, column) ? matrix.el(row, column) : 0.0;
  }

  struct PaperJacobianMismatchSummary {
    double max_relative_error = 0.0;
    double max_boundary_relative_error = 0.0;
    double max_interior_relative_error = 0.0;
    int mismatches = 0;
    int missing_sparsity_mismatches = 0;
  };

  PaperJacobianMismatchSummary compare_paper_jacobian_to_fd(
      FV::KurganovTadmor::Assembler<Discretization, PaperBenchmarkModel> &assembler,
      const Discretization &discretization, const VectorType &sol)
  {
    const int n_dofs = static_cast<int>(sol.size());
    VectorType sol_dot(n_dofs);
    sol_dot = 0.0;

    const SparsityPattern &sp = assembler.get_sparsity_pattern_jacobian();
    SparseMatrixType J_analytic(sp);
    assembler.jacobian(J_analytic, sol, 1.0, sol_dot, 0.0, 0.0);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(static_cast<int>(support_points.size()) == n_dofs);
    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    double y_min = std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();
    for (const Point<2> &point : support_points) {
      x_min = std::min(x_min, point[0]);
      x_max = std::max(x_max, point[0]);
      y_min = std::min(y_min, point[1]);
      y_max = std::max(y_max, point[1]);
    }

    const double boundary_tol = 1.0e-12;
    const auto is_boundary_dof = [&](const int dof) {
      const Point<2> &point = support_points[static_cast<std::size_t>(dof)];
      return std::abs(point[0] - x_min) < boundary_tol || std::abs(point[0] - x_max) < boundary_tol ||
             std::abs(point[1] - y_min) < boundary_tol || std::abs(point[1] - y_max) < boundary_tol;
    };

    constexpr double tolerance = 5.0e-4;
    PaperJacobianMismatchSummary summary;
    for (int column = 0; column < n_dofs; ++column) {
      const double eps = 1.0e-8 * std::max(1.0, std::abs(sol[column]));
      VectorType u_plus = sol;
      VectorType u_minus = sol;
      u_plus[column] += eps;
      u_minus[column] -= eps;

      VectorType r_plus(n_dofs);
      VectorType r_minus(n_dofs);
      assembler.residual(r_plus, u_plus, 1.0, sol_dot, 0.0);
      assembler.residual(r_minus, u_minus, 1.0, sol_dot, 0.0);

      for (int row = 0; row < n_dofs; ++row) {
        const bool in_sparsity = J_analytic.get_sparsity_pattern().exists(row, column);
        const double analytic = in_sparsity ? J_analytic.el(row, column) : 0.0;
        const double fd = (r_plus[row] - r_minus[row]) / (2.0 * eps);
        const double error = std::abs(analytic - fd);
        const double scale = std::max(1.0, std::abs(fd));
        const double relative_error = error / scale;
        summary.max_relative_error = std::max(summary.max_relative_error, relative_error);
        if (is_boundary_dof(row) || is_boundary_dof(column))
          summary.max_boundary_relative_error = std::max(summary.max_boundary_relative_error, relative_error);
        else
          summary.max_interior_relative_error = std::max(summary.max_interior_relative_error, relative_error);

        if (relative_error > tolerance) {
          if (!in_sparsity) ++summary.missing_sparsity_mismatches;
          if (summary.mismatches < 20) {
            std::cout << "Paper Jacobian mismatch at [" << row << "," << column << "]: "
                      << "row_phi=(" << support_points[static_cast<std::size_t>(row)][0] << ", "
                      << support_points[static_cast<std::size_t>(row)][1] << ") "
                      << "column_phi=(" << support_points[static_cast<std::size_t>(column)][0] << ", "
                      << support_points[static_cast<std::size_t>(column)][1] << ") "
                      << "boundary_row=" << is_boundary_dof(row)
                      << " boundary_column=" << is_boundary_dof(column)
                      << " in_sparsity=" << in_sparsity
                      << " analytic=" << analytic << " fd=" << fd
                      << " rel_err=" << relative_error << "\n";
          }
          ++summary.mismatches;
        }
      }
    }
    return summary;
  }

  void require_paper_jacobian_matches_fd_at_probe(const PaperBenchmark &benchmark, const double step,
                                                  const double t_probe, const std::string &temporary_name)
  {
    ensure_logger();

    auto json = make_paper_json(benchmark, step, t_probe);
    PaperBenchmarkModel model(benchmark, json, step);
    Config::ConfigurationMesh<dim> mesh_config(json);
    Mesh mesh(mesh_config);
    Discretization discretization(mesh, json);
    FV::KurganovTadmor::Assembler<Discretization, PaperBenchmarkModel> assembler(discretization, model, json);

    kt_regression::TemporaryDirectory tmp_dir(temporary_name);
    DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), temporary_name, "output", json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    time_stepper.run(&state, 0.0, t_probe);
    model.set_time(t_probe);

    const PaperJacobianMismatchSummary summary =
        compare_paper_jacobian_to_fd(assembler, discretization, state.spatial_data());
    CAPTURE(benchmark.id, benchmark.paper_reference, step, t_probe, summary.max_relative_error,
            summary.max_boundary_relative_error, summary.max_interior_relative_error, summary.mismatches,
            summary.missing_sparsity_mismatches);
    CHECK(summary.max_boundary_relative_error < 5.0e-4);
    CHECK(summary.max_interior_relative_error < 5.0e-4);
    REQUIRE(summary.mismatches == 0);
  }
} // namespace

TEST_CASE("KT paper determinant diffusion flux AD matches finite differences", "[FV][KT][paper][2d]")
{
  using AD = autodiff::Real<1, double>;

  auto model = make_boundary_test_model(o2_case_i);
  model.set_time(30.3);

  const std::array<double, 2> fe_values{{0.12, -0.08}};
  PaperFluxGradient<double> gradient{};
  gradient[0][0] = 0.21;
  gradient[0][1] = -0.03;
  gradient[1][0] = 0.04;
  gradient[1][1] = 0.18;

  const auto flux = evaluate_paper_diffusion_flux(model, fe_values, gradient);
  REQUIRE(std::isfinite(flux[0][0]));
  CHECK_THAT(flux[0][1], Catch::Matchers::WithinAbs(0.0, 1.0e-14));
  CHECK_THAT(flux[1][0], Catch::Matchers::WithinAbs(0.0, 1.0e-14));
  CHECK_THAT(flux[1][1], Catch::Matchers::WithinAbs(flux[0][0], 1.0e-14));

  for (unsigned int row = 0; row < dim; ++row) {
    for (unsigned int column = 0; column < dim; ++column) {
      const double eps = 1.0e-6 * std::max(1.0, std::abs(gradient[row][column]));
      auto gradient_plus = gradient;
      auto gradient_minus = gradient;
      gradient_plus[row][column] += eps;
      gradient_minus[row][column] -= eps;

      const double fd = (evaluate_paper_diffusion_flux(model, fe_values, gradient_plus)[0][0] -
                         evaluate_paper_diffusion_flux(model, fe_values, gradient_minus)[0][0]) /
                        (2.0 * eps);

      std::array<AD, 2> fe_values_ad{{AD(fe_values[0]), AD(fe_values[1])}};
      PaperFluxGradient<AD> gradient_ad{};
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          gradient_ad[i][j] = AD(gradient[i][j]);
      autodiff::detail::seed<1>(gradient_ad[row][column], 1.0);

      const auto flux_ad = evaluate_paper_diffusion_flux(model, fe_values_ad, gradient_ad);
      const double ad = autodiff::derivative(flux_ad[0][0]);
      const double ad_duplicate = autodiff::derivative(flux_ad[1][1]);
      const double scale = std::max(1.0, std::abs(fd));
      CAPTURE(row, column, ad, ad_duplicate, fd);
      CHECK(std::abs(ad - fd) <= 1.0e-6 * scale);
      CHECK(std::abs(ad_duplicate - fd) <= 1.0e-6 * scale);
      CHECK_THAT(autodiff::derivative(flux_ad[0][1]), Catch::Matchers::WithinAbs(0.0, 1.0e-14));
      CHECK_THAT(autodiff::derivative(flux_ad[1][0]), Catch::Matchers::WithinAbs(0.0, 1.0e-14));
    }
  }
}

TEST_CASE("KT paper case I assembler Jacobian matches FD near stalled flow", "[FV][KT][paper][2d][slow]")
{
  require_paper_jacobian_matches_fd_at_probe(o2_case_i, 0.5, 30.3, "zero_dim_2d_kt_jacobian_regression_case_i");
}

TEST_CASE("KT paper case V assembler Jacobian matches FD near stalled flow", "[FV][KT][paper][2d][slow]")
{
  require_paper_jacobian_matches_fd_at_probe(non_o2_case_v, 0.5, 25.3,
                                             "zero_dim_2d_kt_jacobian_regression_case_v");
}

TEST_CASE("KT paper case VII assembler Jacobian matches FD near stalled flow", "[FV][KT][paper][2d][slow]")
{
  require_paper_jacobian_matches_fd_at_probe(on_om_case_vii, 0.5, 25.7,
                                             "zero_dim_2d_kt_jacobian_regression_case_vii");
}

TEST_CASE("KT paper boundary model applies affine two-ghost outer stencils", "[FV][KT][paper][2d]")
{
  using namespace def::BoundaryStencilIndex;

  auto model = make_boundary_test_model(o2_case_i);
  def::BoundaryStencilValues<2, double, 2> u_stencil{
      {{{1.0, 3.0}}, {{4.0, 9.0}}, {{7.0, 15.0}}, {{17.0, 18.0}}, {{19.0, 20.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
  def::BoundaryStencilPoints<2> x_stencil{{Point<2>(7.5, 2.0), Point<2>(8.5, 2.0), Point<2>(9.5, 2.0),
                                           Point<2>(20.0, 21.0), Point<2>(21.0, 22.0), Point<2>(), Point<2>()}};

  REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<2>(10.0, 2.0)));

  CHECK_THAT(x_stencil[upper_inner][0], Catch::Matchers::WithinAbs(10.5, 1.0e-12));
  CHECK_THAT(x_stencil[upper_outer][0], Catch::Matchers::WithinAbs(11.5, 1.0e-12));
  CHECK_THAT(x_stencil[upper_inner][1], Catch::Matchers::WithinAbs(2.0, 1.0e-12));
  CHECK_THAT(x_stencil[upper_outer][1], Catch::Matchers::WithinAbs(2.0, 1.0e-12));
  CHECK_THAT(u_stencil[upper_inner][0], Catch::Matchers::WithinAbs(10.0, 1.0e-12));
  CHECK_THAT(u_stencil[upper_outer][0], Catch::Matchers::WithinAbs(13.0, 1.0e-12));
  CHECK_THAT(u_stencil[upper_inner][1], Catch::Matchers::WithinAbs(21.0, 1.0e-12));
  CHECK_THAT(u_stencil[upper_outer][1], Catch::Matchers::WithinAbs(27.0, 1.0e-12));
}

TEST_CASE("KT paper boundary model applies O(Nbar)xO(Mbar) lower-axis symmetry stencils", "[FV][KT][paper][2d]")
{
  using namespace def::BoundaryStencilIndex;

  auto model = make_boundary_test_model(on_om_case_vii);
  def::BoundaryStencilValues<2, double, 2> u_stencil{
      {{{11.0, 21.0}}, {{12.0, 22.0}}, {{5.0, 7.0}}, {{2.0, 13.0}}, {{4.0, 17.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
  def::BoundaryStencilPoints<2> x_stencil{{Point<2>(8.0, 8.0), Point<2>(9.0, 9.0), Point<2>(0.5, 3.0),
                                           Point<2>(1.5, 3.0), Point<2>(2.5, 3.0), Point<2>(), Point<2>()}};

  REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<2>(0.0, 3.0)));

  CHECK_THAT(x_stencil[lower_inner][0], Catch::Matchers::WithinAbs(-0.5, 1.0e-12));
  CHECK_THAT(x_stencil[lower_outer][0], Catch::Matchers::WithinAbs(-1.5, 1.0e-12));
  CHECK_THAT(x_stencil[lower_inner][1], Catch::Matchers::WithinAbs(3.0, 1.0e-12));
  CHECK_THAT(x_stencil[lower_outer][1], Catch::Matchers::WithinAbs(3.0, 1.0e-12));
  CHECK_THAT(u_stencil[physical_cell][0], Catch::Matchers::WithinAbs(0.0, 1.0e-12));
  CHECK_THAT(u_stencil[lower_inner][0], Catch::Matchers::WithinAbs(-2.0, 1.0e-12));
  CHECK_THAT(u_stencil[lower_outer][0], Catch::Matchers::WithinAbs(-4.0, 1.0e-12));
  CHECK_THAT(u_stencil[physical_cell][1], Catch::Matchers::WithinAbs(7.0, 1.0e-12));
  CHECK_THAT(u_stencil[lower_inner][1], Catch::Matchers::WithinAbs(7.0, 1.0e-12));
  CHECK_THAT(u_stencil[lower_outer][1], Catch::Matchers::WithinAbs(13.0, 1.0e-12));
}

TEST_CASE("KT paper zero-dimensional O(2) case I extracts Gamma2 and O(2) symmetry", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(o2_case_i);
}

TEST_CASE("KT paper zero-dimensional O(2) case II extracts Gamma2 and O(2) symmetry", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(o2_case_ii);
}

TEST_CASE("KT paper zero-dimensional O(2) case III extracts Gamma2 and O(2) symmetry", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(o2_case_iii);
}

TEST_CASE("KT paper zero-dimensional O(2) case IV extracts Gamma2 and O(2) symmetry", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(o2_case_iv);
}

TEST_CASE("KT paper zero-dimensional O(2) cases I-IV show decreasing Gamma2 error", "[FV][KT][paper][2d][slow]")
{
  run_paper_convergence_check(o2_case_i);
}

TEST_CASE("KT paper non-O(2) case V extracts minimum and full Gamma2 matrix", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(non_o2_case_v);
}

TEST_CASE("KT paper non-O(2) case V improves minimum extraction with interpolation", "[FV][KT][paper][2d][slow]")
{
  run_paper_convergence_check(non_o2_case_v);
}

TEST_CASE("KT paper rotated pyramid case VI alpha 0 extracts diagonal Gamma2", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(pyramid_case_vi_alpha_0);
}

TEST_CASE("KT paper rotated pyramid case VI alpha 0.3 tracks off-diagonal Gamma2 decay", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(pyramid_case_vi_alpha_03);
}

TEST_CASE("KT paper rotated pyramid case VI alpha 0.5 tracks off-diagonal Gamma2 decay", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(pyramid_case_vi_alpha_05);
}

TEST_CASE("KT paper rotated pyramid case VI alpha pi over 4 tracks off-diagonal Gamma2 decay", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(pyramid_case_vi_alpha_pi4);
}

TEST_CASE("KT paper rotated pyramid case VI angle sweep shows convergence", "[FV][KT][paper][2d][slow]")
{
  run_paper_convergence_check(pyramid_case_vi_alpha_03);
}

TEST_CASE("KT paper O(Nbar)xO(Mbar) case VII extracts both curvature sectors", "[FV][KT][paper][2d]")
{
  require_paper_benchmark_matches_reference(on_om_case_vii);
}

TEST_CASE("KT paper O(Nbar)xO(Mbar) case VII shows near-quadratic Gamma2 convergence", "[FV][KT][paper][2d][slow]")
{
  run_paper_convergence_check(on_om_case_vii);
}
