#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/timestepping/linear_solver/ScaledGMRES.hh"
#include "kt_regression_helpers.hh"

#include <DiFfRG/common/json.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/regulators.hh>
#include <DiFfRG/physics/thermodynamics.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <petscsnes.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 1;
  constexpr double lambda_uv = 0.6;
  constexpr double k_ir = 1.0e-3;
  const double final_time = std::log(lambda_uv / k_ir);
  constexpr double sigma_min = 0.0;
  constexpr double sigma_max_over_lambda = 1.5;
  constexpr double sigma_max = sigma_max_over_lambda * lambda_uv;
  constexpr double delta_sigma = 1.0e-3; // N = (sigma_max - sigma_min)/delta_sigma = 900 cells
  constexpr std::size_t n_cells = static_cast<std::size_t>((sigma_max - sigma_min) / delta_sigma + 0.5);
  constexpr double grid_tol = 1.0e-12;

  struct PaperLitimParameters {
    static constexpr double Nc = 3.0;
    static constexpr double Nf = 2.0;
    static constexpr double T = 1.0e-4;
    static constexpr double muq = 0.0;
    static constexpr double hPhi = 3.41;
    static constexpr double m2Phi = -0.03006;
    static constexpr double lambdaPhi = 19.260;
    static constexpr double cSigma = 0.001695;
  };

  template <typename Model> class ConstrainExplicitBreakingOrigin
  {
  public:
    template <typename Constraints, typename Context>
    void apply_affine_constraints(Constraints &constraints, const Context &context) const
    {
      const auto candidate = def::internal::select_origin_candidate<"u">(context, context.template support<"u">());
      if (!candidate.has_value()) return;

      constraints.add_line(*candidate);
      constraints.set_inhomogeneity(*candidate, -PaperLitimParameters::cSigma);
    }
  };

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using NumberType = double;
  using Mesh = RectangularMesh<dim>;
  using Discretization = FV::Discretization<Components, NumberType, Mesh>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using JacobianReconstructor = def::FirstOrderReconstructor<dim, double>;
  template <typename Model>
  using ExactJacobianAssembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;
  template <typename Model>
  using FirstOrderJacobianAssembler =
      FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor, FV::KurganovTadmor::MaxEigenvalueWaveSpeed,
                                    JacobianReconstructor>;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, ScaledGMRES>;
  using ExplicitTimeStepper = TimeStepperBoostRK54<VectorType, SparseMatrixType, dim>;

  double grid_spacing() { return delta_sigma; }

  double stable_coth(const double x)
  {
    if (std::abs(x) < 1.0e-4) return 1.0 / x + x / 3.0 - x * x * x / 45.0;
    return 1.0 / std::tanh(x);
  }

  double analytic_sigma_litim_loop(const double k, const double m2_sigma, const double T)
  {
    const double energy = std::sqrt(k * k + m2_sigma);
    const double thermal_factor = stable_coth(energy / (2.0 * T));
    return std::pow(k, 5) * thermal_factor / (12.0 * std::numbers::pi * std::numbers::pi * energy);
  }

  double relative_error(const double value, const double reference)
  {
    const double scale = std::max({1.0, std::abs(value), std::abs(reference)});
    return std::abs(value - reference) / scale;
  }

  double initial_potential(const double sigma)
  {
    return 0.5 * PaperLitimParameters::m2Phi * sigma * sigma +
           0.25 * PaperLitimParameters::lambdaPhi * std::pow(sigma, 4) - PaperLitimParameters::cSigma * sigma;
  }

  double initial_potential_second_derivative(const double sigma)
  {
    return PaperLitimParameters::m2Phi + 3.0 * PaperLitimParameters::lambdaPhi * sigma * sigma;
  }

  double finite_volume_second_derivative_at_face(const double face_sigma)
  {
    return initial_potential_second_derivative(face_sigma) +
           0.5 * PaperLitimParameters::lambdaPhi * grid_spacing() * grid_spacing();
  }

  void log_bad_bosonic_mass_argument(const char *label, const double x, const double k, const double m2)
  {
    const double regulated_argument = k * k + m2;
    if (std::isfinite(m2) && std::isfinite(regulated_argument) && regulated_argument >= 0.0) return;

    static int pion_logs = 0;
    static int sigma_logs = 0;
    int &log_count = label[0] == 'p' ? pion_logs : sigma_logs;
    if (log_count >= 8) return;
    ++log_count;

    const auto cell = static_cast<std::size_t>(std::max(0.0, std::floor((x - sigma_min) / delta_sigma)));
    std::clog << "[QM DIAG] " << label << " invalid regulated mass argument"
              << " cell=" << cell << " sigma=" << x << " k=" << k << " k2=" << k * k << " m2=" << m2
              << " k2_plus_m2=" << regulated_argument << '\n';
  }

  template <typename REG> class QMPionKernel
  {
  public:
    using Regulator = REG;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double q, const auto k, const auto /*Nc*/, const auto Nf,
                                                   const auto T, const auto /*muq*/, const auto /*hPhi*/,
                                                   const auto m2Pion)
    {
      using namespace DiFfRG;
      using Kokkos::sqrt;

      const auto rb = REG::RB(powr<2>(k), powr<2>(q));
      const auto rbdot = REG::RBdot(powr<2>(k), powr<2>(q));
      const auto energy = sqrt(powr<2>(q) + rb + m2Pion);
      return -0.25 * (-rbdot) * CothFiniteT(energy, T) * (-1. + powr<2>(Nf)) / energy;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto /*k*/, const auto /*Nc*/, const auto /*Nf*/,
                                                     const auto /*T*/, const auto /*muq*/, const auto /*hPhi*/,
                                                     const auto /*m2Pion*/)
    {
      return 0.;
    }
  };

  template <typename REG> class QMSigmaKernel
  {
  public:
    using Regulator = REG;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double q, const auto k, const auto /*Nc*/, const auto /*Nf*/,
                                                   const auto T, const auto /*muq*/, const auto /*hPhi*/,
                                                   const auto m2Sigma)
    {
      using namespace DiFfRG;
      using Kokkos::sqrt;

      const auto rb = REG::RB(powr<2>(k), powr<2>(q));
      const auto rbdot = REG::RBdot(powr<2>(k), powr<2>(q));
      const auto energy = sqrt(powr<2>(q) + rb + m2Sigma);
      return 0.25 * rbdot * CothFiniteT(energy, T) / energy;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto /*k*/, const auto /*Nc*/, const auto /*Nf*/,
                                                     const auto /*T*/, const auto /*muq*/, const auto /*hPhi*/,
                                                     const auto /*m2Sigma*/)
    {
      return 0.;
    }
  };

  template <typename REG> class QMQuarkKernel
  {
  public:
    using Regulator = REG;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double q, const auto k, const auto Nc, const auto Nf,
                                                   const auto T, const auto muq, const auto hPhi, const auto sigma)
    {
      using namespace DiFfRG;
      using Kokkos::sqrt;

      const auto rf = REG::RF(powr<2>(k), powr<2>(q));
      const auto rfdot = REG::RFdot(powr<2>(k), powr<2>(q));
      const auto q_reg = q + rf;
      const auto m2Quark = powr<2>(hPhi * sigma);
      const auto energy = sqrt(powr<2>(q_reg) + m2Quark);
      return -Nc * Nf * q_reg * (-rfdot) * (TanhFiniteT(muq - energy, T) - TanhFiniteT(muq + energy, T)) / energy;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto /*k*/, const auto /*Nc*/, const auto /*Nf*/,
                                                     const auto /*T*/, const auto /*muq*/, const auto /*hPhi*/,
                                                     const auto /*sigma*/)
    {
      return 0.;
    }
  };

  template <typename Kernel> class IntegratorWrapper
  {
  public:
    IntegratorWrapper(QuadratureProvider &quadrature_provider, const JSONValue &json)
        : integrator(quadrature_provider, json), integrator_ad(quadrature_provider, json),
          integrator_ad2(quadrature_provider, json)
    {
    }

    template <typename... Args> void get(double &dest, Args &&...args) const
    {
      integrator.get(dest, std::forward<Args>(args)...);
    }

    template <typename... Args> void get(autodiff::real &dest, Args &&...args) const
    {
      integrator_ad.get(dest, std::forward<Args>(args)...);
    }

    template <typename... Args> void get(autodiff::Real<2, double> &dest, Args &&...args) const
    {
      integrator_ad2.get(dest, std::forward<Args>(args)...);
    }

    mutable Integrator_p2<3, double, Kernel, TBB_exec> integrator;
    mutable Integrator_p2<3, autodiff::real, Kernel, TBB_exec> integrator_ad;
    mutable Integrator_p2<3, autodiff::Real<2, double>, Kernel, TBB_exec> integrator_ad2;
  };

  class QuarkMesonLitimFlows
  {
  public:
    using Regulator = LitimRegulator<>;
    using PionIntegrator = IntegratorWrapper<QMPionKernel<Regulator>>;
    using SigmaIntegrator = IntegratorWrapper<QMSigmaKernel<Regulator>>;
    using QuarkIntegrator = IntegratorWrapper<QMQuarkKernel<Regulator>>;

    explicit QuarkMesonLitimFlows(const JSONValue &json)
        : quadrature_provider(json), pion(quadrature_provider, json), sigma(quadrature_provider, json),
          quark(quadrature_provider, json)
    {
    }

    void set_k(const double k)
    {
      all_set_k(pion, k);
      all_set_k(sigma, k);
      all_set_k(quark, k);
    }

    void set_T(const double T)
    {
      all_set_T(pion, T);
      all_set_T(sigma, T);
      all_set_T(quark, T);
    }

    QuadratureProvider quadrature_provider;
    PionIntegrator pion;
    SigmaIntegrator sigma;
    QuarkIntegrator quark;
  };

  class QuarkMesonKTModel : public def::AbstractModel<QuarkMesonKTModel, Components>,
                            public def::fRG,
                            public def::OriginShiftedOddLinearExtrapolationBoundaries<QuarkMesonKTModel>,
                            public ConstrainExplicitBreakingOrigin<QuarkMesonKTModel>,
                            public def::AD<QuarkMesonKTModel>
  {
  public:
    explicit QuarkMesonKTModel(const JSONValue &json) : def::fRG(json), flows(json)
    {
      flows.set_k(Lambda);
      flows.set_T(PaperLitimParameters::T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      values[0] = PaperLitimParameters::m2Phi * x[0] + PaperLitimParameters::lambdaPhi * std::pow(x[0], 3) -
                  PaperLitimParameters::cSigma;
    }

    template <typename NT, std::size_t n_components> std::array<NT, n_components> origin_odd_reflection_values() const
    {
      static_assert(n_components == 1, "QuarkMesonKTModel expects one FE function.");
      return {NT(-PaperLitimParameters::cSigma)};
    }

    void set_time(const double t_)
    {
      t = t_;
      k = std::exp(-t) * Lambda;
      flows.set_k(k);
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void KurganovTadmor_advection_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i,
                                       const Point<spatial_dim> &x, const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "QuarkMesonKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "QuarkMesonKTModel expects one FE function.");

      const auto &fe_functions = get<0>(sol);
      const NT u = fe_functions[0];
      const double sigma_value = x[0];
      const NT sigma_nt = NT(sigma_value);
      const NT m2_pion = (u + NT(PaperLitimParameters::cSigma)) / sigma_nt;
      if constexpr (std::is_same_v<std::decay_t<NT>, double>)
        log_bad_bosonic_mass_argument("pion", sigma_value, k, m2_pion);

      NT pion_loop{};
      flows.pion.get(pion_loop, k, PaperLitimParameters::Nc, PaperLitimParameters::Nf, PaperLitimParameters::T,
                     PaperLitimParameters::muq, PaperLitimParameters::hPhi, m2_pion);

      F_i[0][0] = pion_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "QuarkMesonKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "QuarkMesonKTModel expects one FE function.");

      const auto &fe_derivatives = get<1>(sol);
      const NT m2_sigma = fe_derivatives[0][0];
      if constexpr (std::is_same_v<std::decay_t<NT>, double>) log_bad_bosonic_mass_argument("sigma", x[0], k, m2_sigma);

      NT sigma_loop{};
      flows.sigma.get(sigma_loop, k, PaperLitimParameters::Nc, PaperLitimParameters::Nf, PaperLitimParameters::T,
                      PaperLitimParameters::muq, PaperLitimParameters::hPhi, m2_sigma);
      // KT residual sums the fluxes: (H + D)·n. The diffusion flux is the physical flux
      // with the same sign as the advection flux (conservation-law / CG convention). The
      // sigma loop ∝ +1/√(k²+m²_σ) is a decreasing function of m²_σ = ∂_σ u, i.e.
      // ∂F_diff/∂(∂_σ u) < 0 — the forward-diffusion sign under (H + D).
      F_i[0][0] = sigma_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void source(std::array<NT, n_fe_functions> &s_i, const Point<spatial_dim> &x, const Solutions &) const
    {
      static_assert(spatial_dim == dim, "QuarkMesonKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "QuarkMesonKTModel expects one FE function.");

      const NT sigma_value = NT(x[0]);
      NT source_right{};
      NT source_left{};
      constexpr double source_dx = 1.0e-5;
      flows.quark.get(source_right, k, PaperLitimParameters::Nc, PaperLitimParameters::Nf, PaperLitimParameters::T,
                      PaperLitimParameters::muq, PaperLitimParameters::hPhi, sigma_value + NT(source_dx));
      flows.quark.get(source_left, k, PaperLitimParameters::Nc, PaperLitimParameters::Nf, PaperLitimParameters::T,
                      PaperLitimParameters::muq, PaperLitimParameters::hPhi, sigma_value - NT(source_dx));
      s_i[0] = (source_right - source_left) / NT(2.0 * source_dx);
    }

  private:
    mutable QuarkMesonLitimFlows flows;
  };

  JSONValue make_json(const double final_time, const double output_dt = 2.0e-2, const double explicit_dt = 1.0e-8,
                      const double explicit_abs_tol = 1.0e-14, const double explicit_rel_tol = 1.0e-14)
  {
    return json::value(
        {{"physical", {{"Lambda", lambda_uv}, {"cSigma", PaperLitimParameters::cSigma}}},
         {"integration", {{"x_order", 100}, {"x_extent_tolerance", 1.0e-4}, {"jacobian_quadrature_factor", 0.5}}},
         {"discretization",
          {{"fe_order", 0},
           {"threads", 16},
           {"batch_size", 32},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"output_buffer_size", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", output_dt},
           {"explicit",
            {{"dt", explicit_dt},
             {"minimal_dt", 1.0e-12},
             {"maximal_dt", 1.0},
             {"abs_tol", explicit_abs_tol},
             {"rel_tol", explicit_rel_tol}}},
           {"implicit",
            {{"dt", 1.0e-9},
             {"minimal_dt", 1.0e-15},
             {"maximal_dt", 1.0e-5},
             {"abs_tol", 1.0e-12},
             {"rel_tol", 1.0e-12},
             {"max_steps", 20000000}}}}},
         {"output",
          {{"verbosity", 1},
           {"vtk", false},
           {"hdf5", true},
           {"fv_reconstruction_diagnostics", true},
           {"fv_residual_contribution_diagnostics", true}}}});
  }

  Config::ConfigurationMesh<1> make_mesh_config()
  {
    const Config::GridAxis sigma_axis(sigma_min, grid_spacing(), sigma_max);
    return Config::ConfigurationMesh<1>(0u, std::vector<Config::GridAxis>{sigma_axis});
  }

  void initialize_exact_cell_averages(FV::FlowingVariables<Discretization> &state,
                                      const std::vector<Point<dim>> &support_points)
  {
    auto &u = state.spatial_data();
    REQUIRE(u.size() == support_points.size());
    const double delta = grid_spacing();
    for (unsigned int i = 0; i < u.size(); ++i) {
      const double sigma = support_points[i][0];
      const double right = initial_potential(sigma + 0.5 * delta);
      const double left = initial_potential(sigma - 0.5 * delta);
      u[i] = (right - left) / delta;
    }
  }

  template <typename AssemblerType, typename TimeStepperType = ImplicitTimeStepper>
  kt_regression::SampledProfile run_flow(const double final_time, const JSONValue &json,
                                         const std::string &output_prefix = "quark_meson_kt_regression")
  {
    kt_regression::ensure_diffrg_initialized();
    QuarkMesonKTModel model(json);
    Mesh mesh(make_mesh_config());
    Discretization discretization(mesh, json);
    AssemblerType assembler(discretization, model, json);

    kt_regression::TemporaryDirectory tmp_dir(output_prefix, false);
    std::clog << "[QM DIAG] retaining output directory " << tmp_dir.path << '\n';
    DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), output_prefix, "output", json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepperType time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == n_cells);
    initialize_exact_cell_averages(state, support_points);
    discretization.get_constraints().distribute(state.spatial_data());

    time_stepper.run(&state, 0.0, final_time);

    return kt_regression::sample_sorted_profile(state, discretization, grid_tol);
  }

  template <typename AssemblerType, typename TimeStepperType = ImplicitTimeStepper>
  kt_regression::SampledProfile run_flow(const double final_time)
  {
    const JSONValue json = make_json(final_time);
    return run_flow<AssemblerType, TimeStepperType>(final_time, json);
  }

} // namespace

TEST_CASE("QM sigma Litim diffusion loop matches analytic finite-T result near pole",
          "[FV][KT][QuarkMeson][regression][litim-sigma-loop]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  constexpr double t_problem = 0.00172622;
  const double k = lambda_uv * std::exp(-t_problem);
  const JSONValue json = make_json(t_problem);
  QuarkMesonLitimFlows flows(json);
  flows.set_k(k);
  flows.set_T(PaperLitimParameters::T);

  const std::array<double, 4> eps_values = {1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8};

  double max_relative_error = 0.0;
  std::clog << "[QM sigma Litim loop] eps,m2_sigma,numeric,analytic,abs_error,rel_error\n";
  for (const double eps : eps_values) {
    const double m2_sigma = -k * k + eps;
    double numeric = 0.0;
    flows.sigma.get(numeric, k, PaperLitimParameters::Nc, PaperLitimParameters::Nf, PaperLitimParameters::T,
                    PaperLitimParameters::muq, PaperLitimParameters::hPhi, m2_sigma);

    const double analytic = analytic_sigma_litim_loop(k, m2_sigma, PaperLitimParameters::T);
    const double abs_error = std::abs(numeric - analytic);
    const double rel_error = relative_error(numeric, analytic);
    max_relative_error = std::max(max_relative_error, rel_error);

    std::clog << std::scientific << std::setprecision(16) << "[QM sigma Litim loop] " << eps << ',' << m2_sigma << ','
              << numeric << ',' << analytic << ',' << abs_error << ',' << rel_error << '\n';

    CAPTURE(eps, m2_sigma, numeric, analytic, abs_error, rel_error);
    REQUIRE(std::isfinite(numeric));
    REQUIRE(std::isfinite(analytic));
    CHECK(rel_error < 1.0e-5);
  }

  CAPTURE(max_relative_error);
  CHECK(max_relative_error < 1.0e-5);
}

TEST_CASE("QM initial cell averages reconstruct diffusion second derivative on interior faces",
          "[FV][KT][QuarkMeson][regression][diffusion-reconstruction]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  const JSONValue json = make_json(0.0);
  QuarkMesonKTModel model(json);
  Mesh mesh(make_mesh_config());
  Discretization discretization(mesh, json);
  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  const auto &support_points = discretization.get_support_points();
  REQUIRE(support_points.size() == n_cells);
  initialize_exact_cell_averages(state, support_points);

  const auto &cell_averages = state.spatial_data();
  REQUIRE(cell_averages.size() == support_points.size());

  double max_side_mismatch = 0.0;
  double max_fv_error = 0.0;
  double max_pointwise_offset_error = 0.0;
  std::size_t checked_faces = 0;

  for (std::size_t upper_cell = 2; upper_cell + 1 < support_points.size(); ++upper_cell) {
    const std::size_t lower_cell = upper_cell - 1;
    const Point<dim> face(0.5 * (support_points[lower_cell][0] + support_points[upper_cell][0]));
    const double face_sigma = face[0];

    const std::array<Point<dim>, 2> lower_neighbors = {support_points[lower_cell - 1], support_points[upper_cell]};
    const std::array<std::array<double, 1>, 2> lower_neighbor_values = {
        std::array<double, 1>{cell_averages[lower_cell - 1]}, std::array<double, 1>{cell_averages[upper_cell]}};
    const std::array<double, 1> lower_value = {cell_averages[lower_cell]};
    const auto lower_gradient = Reconstructor::template compute_gradient_at_point<1>(
        support_points[lower_cell], face, lower_value, lower_neighbors, lower_neighbor_values);

    const std::array<Point<dim>, 2> upper_neighbors = {support_points[lower_cell], support_points[upper_cell + 1]};
    const std::array<std::array<double, 1>, 2> upper_neighbor_values = {
        std::array<double, 1>{cell_averages[lower_cell]}, std::array<double, 1>{cell_averages[upper_cell + 1]}};
    const std::array<double, 1> upper_value = {cell_averages[upper_cell]};
    const auto upper_gradient = Reconstructor::template compute_gradient_at_point<1>(
        support_points[upper_cell], face, upper_value, upper_neighbors, upper_neighbor_values);

    const double reconstructed_lower = lower_gradient[0][0];
    const double reconstructed_upper = upper_gradient[0][0];
    const double expected_from_cell_averages = (cell_averages[upper_cell] - cell_averages[lower_cell]) /
                                               (support_points[upper_cell][0] - support_points[lower_cell][0]);
    const double expected_fv = finite_volume_second_derivative_at_face(face_sigma);
    const double expected_pointwise = initial_potential_second_derivative(face_sigma);
    const double expected_pointwise_offset = 0.5 * PaperLitimParameters::lambdaPhi * grid_spacing() * grid_spacing();

    const double side_mismatch = std::abs(reconstructed_lower - reconstructed_upper);
    const double fv_error =
        std::max(std::abs(reconstructed_lower - expected_fv), std::abs(reconstructed_upper - expected_fv));
    const double pointwise_offset_error =
        std::abs((reconstructed_lower - expected_pointwise) - expected_pointwise_offset);

    max_side_mismatch = std::max(max_side_mismatch, side_mismatch);
    max_fv_error = std::max(max_fv_error, fv_error);
    max_pointwise_offset_error = std::max(max_pointwise_offset_error, pointwise_offset_error);
    ++checked_faces;

    CAPTURE(upper_cell, face_sigma, reconstructed_lower, reconstructed_upper, expected_from_cell_averages, expected_fv,
            expected_pointwise);
    CHECK(reconstructed_lower == Catch::Approx(expected_from_cell_averages).margin(1.0e-12));
    CHECK(reconstructed_upper == Catch::Approx(expected_from_cell_averages).margin(1.0e-12));
    CHECK(reconstructed_lower == Catch::Approx(expected_fv).margin(2.0e-9));
    CHECK(reconstructed_upper == Catch::Approx(expected_fv).margin(2.0e-9));
    CHECK(reconstructed_lower - expected_pointwise == Catch::Approx(expected_pointwise_offset).margin(2.0e-9));
  }

  CAPTURE(checked_faces, max_side_mismatch, max_fv_error, max_pointwise_offset_error);
  CHECK(checked_faces == n_cells - 3);
  CHECK(max_side_mismatch < 1.0e-12);
  CHECK(max_fv_error < 2.0e-9);
  CHECK(max_pointwise_offset_error < 2.0e-9);
}

TEST_CASE("QM initialized flow state satisfies the explicit-breaking origin constraint",
          "[FV][KT][QuarkMeson][regression][origin-constraint]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  const JSONValue json = make_json(0.0);
  QuarkMesonKTModel model(json);
  Mesh mesh(make_mesh_config());
  Discretization discretization(mesh, json);
  [[maybe_unused]] ExactJacobianAssembler<QuarkMesonKTModel> assembler(discretization, model, json);
  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  const auto &support_points = discretization.get_support_points();
  REQUIRE(support_points.size() == n_cells);
  initialize_exact_cell_averages(state, support_points);
  discretization.get_constraints().distribute(state.spatial_data());

  CHECK(state.spatial_data()[0] == Catch::Approx(-PaperLitimParameters::cSigma).margin(1.0e-14));
}

TEST_CASE("KT quark-meson LPA Litim flow reaches final time with exact TVD Jacobian",
          "[FV][KT][QuarkMeson][regression][2601.23005]")
{
  kt_regression::ensure_logger();

  const auto exact_profile = run_flow<ExactJacobianAssembler<QuarkMesonKTModel>>(final_time);
  REQUIRE(exact_profile.x.size() == n_cells);

  for (const double value : exact_profile.y)
    REQUIRE(std::isfinite(value));
}

TEST_CASE("KT quark-meson LPA Litim flow reaches final time with first-order Jacobian",
          "[FV][KT][QuarkMeson][regression][2601.23005][first-order-jacobian]")
{
  kt_regression::ensure_logger();

  const auto first_order_profile = run_flow<FirstOrderJacobianAssembler<QuarkMesonKTModel>>(final_time);
  REQUIRE(first_order_profile.x.size() == n_cells);

  for (const double value : first_order_profile.y)
    REQUIRE(std::isfinite(value));
}

TEST_CASE("KT quark-meson explicit RK exposes non-finite derivative near problematic region",
          "[FV][KT][QuarkMeson][regression][2601.23005][explicit-rk][diagnostic]")
{
  kt_regression::ensure_logger();

  constexpr double t_problem = 0.00172622;
  constexpr double output_dt = 1.0e-5;
  constexpr double explicit_final_time = t_problem + output_dt;
  const JSONValue json = make_json(explicit_final_time, output_dt, 1.0e-8, 1.0e-10, 1.0e-10);

  REQUIRE_THROWS_WITH((run_flow<ExactJacobianAssembler<QuarkMesonKTModel>, ExplicitTimeStepper>(
                          explicit_final_time, json, "quark_meson_kt_explicit_regression")),
                      Catch::Matchers::ContainsSubstring("dy is not finite"));
}
