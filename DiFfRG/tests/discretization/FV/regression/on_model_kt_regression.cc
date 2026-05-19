#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed_fd.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed_zero_deriv.hh>
#include "DiFfRG/discretization/FV/discretization.hh"
// CG comparison test (same physics as KT but with continuous Galerkin
// discretisation). See ON_CG_LSM_Model below.
#include <DiFfRG/discretization/FEM/assembler/cg.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include "DiFfRG/timestepping/timestepping.hh"
#include "kt_regression_helpers.hh"

#include <DiFfRG/common/json.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
// Headers needed by the integrator-based smoke test below
// (V_pion_kernel_integrator / V_sigma_kernel_integrator + Integrator_p2 wrappers)
#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/physics.hh>
#include <DiFfRG/physics/regulators.hh>
#include <DiFfRG/physics/thermodynamics.hh>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 1;
  constexpr double final_time = 60.0;
  constexpr double sigma_max = 10.0;
  constexpr std::size_t n_cells = 800;
  constexpr double grid_tol = 1.0e-12;
  constexpr double origin_tol = 1.0e-12;
  constexpr int diagnostic_threads = 1;
  constexpr double scenario_i_lambda = 1.0e6;
  constexpr double scenario_ii_lambda = 1.0e12;
  constexpr double scenario_iii_lambda = 1.0e12;
  constexpr double scenario_iv_lambda = 1.0e8;

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  template <typename Model> using ConstrainUAtOrigin = def::ConstrainOriginSupportPointToZero<"u", Model>;
  using NumberType = double;
  using Mesh = RectangularMesh<dim>;
  using Discretization = FV::Discretization<Components, NumberType, Mesh>;
  using SampledProfile = kt_regression::SampledProfile;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;

  template <typename Model>
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor,
                                                  FV::KurganovTadmor::MaxEigenvalueWaveSpeedZeroDeriv>;
  using ImplicitTimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;

  struct GridSettings {
    std::size_t cells;
    double sigma_min;
    double sigma_max;
  };

  GridSettings default_grid_settings() { return GridSettings{n_cells, 0.0, sigma_max}; }

  double grid_spacing(const GridSettings &grid_settings)
  {
    return (grid_settings.sigma_max - grid_settings.sigma_min) / static_cast<double>(grid_settings.cells - 1);
  }

  enum class ScenarioKind {
    I,
    II,
    III,
    IV,
    // Realistic linear-sigma-model initial potential at physical RG scale:
    //   V_0(sigma) = (m2/2) sigma^2 + (lambda/8) sigma^4
    // with m2 and lambda taken from `LSMPhysicalParameters` (defaulted to the
    // values used by Examples/ONfiniteT_KT/parameter.json: m2 = -0.1,
    // lambda = 71.6). This scenario exercises the analytical Litim flow at the
    // **same physical regime as the user-facing example**, with
    // Lambda = O(1) instead of the 1e6..1e12 used by Scenarios I..IV — i.e.
    // no "warm-up" stretch before the wave speed becomes O(1).
    LinearSigmaModelPhysical,
  };

  struct LSMPhysicalParameters {
    static constexpr double m2 = -0.1;
    static constexpr double lambda_quartic = 71.6;
    static constexpr double T = 0.05;  // used by the integrator-based smoke test below
  };

  struct FlowCase {
    std::string relative_path;
    std::string label;
    ScenarioKind scenario;
    double n_flavors;
    double lambda;
    std::optional<std::size_t> profile_group_index;
    bool has_reference_potential;
    double profile_abs_tol;
    double profile_rel_tol;
  };

  struct ProfileData {
    std::vector<double> x;
    std::vector<double> y;
  };

  struct ReferenceSnapshot {
    double time;
    std::optional<ProfileData> U;
    ProfileData u;
  };

  struct ReferenceFlowData {
    std::vector<ReferenceSnapshot> snapshots;
  };

  template <typename T> double scalar_value(const T &x) { return static_cast<double>(x); }

  template <typename T> double scalar_value(const autodiff::Real<1, T> &x) { return static_cast<double>(x.val()); }

  FlowCase make_flow_case(const std::string_view relative_path, const std::string_view label, const ScenarioKind scenario,
                          const double n_flavors, const double lambda, const std::optional<std::size_t> profile_group_index,
                          const bool has_reference_potential)
  {
    constexpr double regression_profile_abs_tol = 7.5e-3;
    constexpr double regression_profile_rel_tol = 5.0e-3;
    return FlowCase{std::string(relative_path), std::string(label), scenario, n_flavors, lambda, profile_group_index,
                    has_reference_potential, regression_profile_abs_tol, regression_profile_rel_tol};
  }

  template <typename Derived>
  struct ONFlowDescriptor {
    static FlowCase flow_case()
    {
      return make_flow_case(Derived::relative_path, Derived::label, Derived::scenario, Derived::n_flavors,
                            Derived::lambda, Derived::profile_group_index, Derived::has_reference_potential);
    }
  };

  struct ScenarioI_ON1 : ONFlowDescriptor<ScenarioI_ON1> {
    static constexpr auto relative_path = "data/2108_02504/sc_i_on_1_10_100_n_800_xmax_10_lambda_1e6_tir_60_rg_flow.json";
    static constexpr auto label = "Scenario I, N=1";
    static constexpr ScenarioKind scenario = ScenarioKind::I;
    static constexpr double n_flavors = 1.0;
    static constexpr double lambda = scenario_i_lambda;
    static constexpr std::optional<std::size_t> profile_group_index = 0;
    static constexpr bool has_reference_potential = false;
  };

  struct ScenarioI_ON3 : ONFlowDescriptor<ScenarioI_ON3> {
    static constexpr auto relative_path = "data/2108_02504/sc_i_on_3_n_800_xmax_10_lambda_1e6_tir_60_rg_flow.json";
    static constexpr auto label = "Scenario I, N=3";
    static constexpr ScenarioKind scenario = ScenarioKind::I;
    static constexpr double n_flavors = 3.0;
    static constexpr double lambda = scenario_i_lambda;
    static constexpr std::optional<std::size_t> profile_group_index = std::nullopt;
    static constexpr bool has_reference_potential = true;
  };

  struct ScenarioI_ON10 : ONFlowDescriptor<ScenarioI_ON10> {
    static constexpr auto relative_path = "data/2108_02504/sc_i_on_1_10_100_n_800_xmax_10_lambda_1e6_tir_60_rg_flow.json";
    static constexpr auto label = "Scenario I, N=10";
    static constexpr ScenarioKind scenario = ScenarioKind::I;
    static constexpr double n_flavors = 10.0;
    static constexpr double lambda = scenario_i_lambda;
    static constexpr std::optional<std::size_t> profile_group_index = 1;
    static constexpr bool has_reference_potential = false;
  };

  struct ScenarioI_ON100 : ONFlowDescriptor<ScenarioI_ON100> {
    static constexpr auto relative_path = "data/2108_02504/sc_i_on_1_10_100_n_800_xmax_10_lambda_1e6_tir_60_rg_flow.json";
    static constexpr auto label = "Scenario I, N=100";
    static constexpr ScenarioKind scenario = ScenarioKind::I;
    static constexpr double n_flavors = 100.0;
    static constexpr double lambda = scenario_i_lambda;
    static constexpr std::optional<std::size_t> profile_group_index = 2;
    static constexpr bool has_reference_potential = false;
  };

  struct ScenarioII_ON4 : ONFlowDescriptor<ScenarioII_ON4> {
    static constexpr auto relative_path = "data/2108_02504/sc_ii_n_on_4_n_800_xmax_10_lambda_1e12_tir_60_rg_flow.json";
    static constexpr auto label = "Scenario II, N=4";
    static constexpr ScenarioKind scenario = ScenarioKind::II;
    static constexpr double n_flavors = 4.0;
    static constexpr double lambda = scenario_ii_lambda;
    static constexpr std::optional<std::size_t> profile_group_index = std::nullopt;
    static constexpr bool has_reference_potential = true;
  };

  struct ScenarioIII_ON4 : ONFlowDescriptor<ScenarioIII_ON4> {
    static constexpr auto relative_path = "data/2108_02504/sc_iii_on_4_n_800_xmax_10_lambda_1e12_tir_60_rg_flow.json";
    static constexpr auto label = "Scenario III, N=4";
    static constexpr ScenarioKind scenario = ScenarioKind::III;
    static constexpr double n_flavors = 4.0;
    static constexpr double lambda = scenario_iii_lambda;
    static constexpr std::optional<std::size_t> profile_group_index = std::nullopt;
    static constexpr bool has_reference_potential = true;
  };

  struct ScenarioIV_ON3 : ONFlowDescriptor<ScenarioIV_ON3> {
    static constexpr auto relative_path = "data/2108_02504/sc_iv_on_3_n_800_xmax_10_lambda_1e8_tir_60_rg_flow.json";
    static constexpr auto label = "Scenario IV, N=3";
    static constexpr ScenarioKind scenario = ScenarioKind::IV;
    static constexpr double n_flavors = 3.0;
    static constexpr double lambda = scenario_iv_lambda;
    static constexpr std::optional<std::size_t> profile_group_index = std::nullopt;
    static constexpr bool has_reference_potential = true;
  };

  // Realistic physical-scale linear sigma model. Lambda = 0.65 (no UV warm-up),
  // N = 2, initial double-well from m2 = -0.1, lambda_quartic = 71.6, T = 0.05
  // matches Examples/ONfiniteT_KT/parameter.json.
  constexpr double lsm_physical_lambda = 0.65;
  struct LSMPhysical_ON2 : ONFlowDescriptor<LSMPhysical_ON2> {
    static constexpr auto relative_path = "";  // no reference data; smoke test only
    static constexpr auto label = "LSM physical-scale, N=2";
    static constexpr ScenarioKind scenario = ScenarioKind::LinearSigmaModelPhysical;
    static constexpr double n_flavors = 2.0;
    static constexpr double lambda = lsm_physical_lambda;
    static constexpr std::optional<std::size_t> profile_group_index = std::nullopt;
    static constexpr bool has_reference_potential = false;
  };

  ProfileData parse_profile(const json::array &pairs)
  {
    ProfileData result;
    result.x.reserve(pairs.size());
    result.y.reserve(pairs.size());
    for (const auto &entry : pairs) {
      const auto &pair = entry.as_array();
      if (pair.size() != 2) throw std::runtime_error("Expected [x, y] profile entries.");
      result.x.push_back(kt_regression::json_number_to_double(pair[0]));
      result.y.push_back(kt_regression::json_number_to_double(pair[1]));
    }
    return result;
  }

  double scenario_potential(const ScenarioKind scenario, const double sigma)
  {
    const double abs_sigma = std::abs(sigma);
    switch (scenario) {
      case ScenarioKind::I:
        if (abs_sigma <= 2.0) return -0.5 * abs_sigma * abs_sigma;
        if (abs_sigma <= 3.0) return -2.0;
        return 0.5 * (abs_sigma * abs_sigma - 13.0);
      case ScenarioKind::II:
        return -0.5 * abs_sigma * abs_sigma + std::pow(abs_sigma, 4) / 24.0;
      case ScenarioKind::III:
        return 0.5 * abs_sigma * abs_sigma - std::pow(abs_sigma, 4) / 20.0 + std::pow(abs_sigma, 6) / 720.0;
      case ScenarioKind::IV:
        if (abs_sigma <= std::sqrt(8.0)) return -std::pow(abs_sigma * abs_sigma, 1.0 / 3.0);
        return 0.5 * abs_sigma * abs_sigma - 6.0;
      case ScenarioKind::LinearSigmaModelPhysical:
        // V_0(sigma) = (m2/2) sigma^2 + (lambda/8) sigma^4. Smooth, double-well
        // when m2 < 0. m2 and lambda fixed to match the
        // Examples/ONfiniteT_KT/parameter.json values.
        return 0.5 * LSMPhysicalParameters::m2 * sigma * sigma +
               (LSMPhysicalParameters::lambda_quartic / 8.0) * std::pow(sigma, 4);
    }
    throw std::runtime_error("Unknown O(N) test scenario.");
  }

  template <typename Derived, template <typename> typename BoundaryStrategy,
            template <typename> typename ConstraintStrategy>
  class ONKTModelCommon : public def::AbstractModel<Derived, Components>,
                          public def::fRG,
                          public BoundaryStrategy<Derived>,
                          public ConstraintStrategy<Derived>,
                          public def::AD<Derived>
  {
  public:
    ONKTModelCommon(const JSONValue &json, const FlowCase &flow_case, const GridSettings &grid_settings)
        : def::fRG(json), flow_case(flow_case), cell_width(grid_spacing(grid_settings))
    {}

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double sigma = x[0];
      const double right = scenario_potential(flow_case.scenario, sigma + 0.5 * cell_width);
      const double left = scenario_potential(flow_case.scenario, sigma - 0.5 * cell_width);
      values[0] = (right - left) / cell_width;
    }

    template <int spatial_dim, typename FluxNumberType, typename Solutions, std::size_t n_fe_functions>
    void KurganovTadmor_advection_flux(
        std::array<Tensor<1, spatial_dim, FluxNumberType>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
        const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "ONKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "ONKTModel expects a single FE function.");

      const auto &fe_functions = get<0>(sol);
      const FluxNumberType u = fe_functions[0];
      const double sigma = x[0];
      const double r = k;
      const double regulator_insertion = 0.5 * r;

      const FluxNumberType u_over_sigma =
          std::abs(sigma) < origin_tol ? FluxNumberType(0.0) : u / FluxNumberType(sigma);
      static_cast<const Derived *>(this)->note_advection_denominator(sigma, u);
      F_i[0][0] = FluxNumberType(regulator_insertion * (flow_case.n_flavors - 1.0)) /
                  (FluxNumberType(r) + u_over_sigma);
    }

    template <int spatial_dim, typename FluxNumberType, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, FluxNumberType>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "ONKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "ONKTModel expects a single FE function.");

      const auto &fe_derivatives = get<1>(sol);
      const FluxNumberType du_dsigma = fe_derivatives[0][0];
      const double r = k;
      const double half_dr_dt = -0.5 * r;

      (void)x;
      static_cast<const Derived *>(this)->note_diffusion_denominator(x[0], du_dsigma);
      F_i[0][0] = FluxNumberType(half_dr_dt) / (FluxNumberType(r) + du_dsigma);
    }

  protected:
    FlowCase flow_case;
    double cell_width;
  };

  template <template <typename> typename BoundaryStrategy,
            template <typename> typename ConstraintStrategy = def::NoAffineConstraints>
  class ONKTModelBase : public ONKTModelCommon<ONKTModelBase<BoundaryStrategy, ConstraintStrategy>,
                                               BoundaryStrategy, ConstraintStrategy>
  {
  public:
    using Base = ONKTModelCommon<ONKTModelBase<BoundaryStrategy, ConstraintStrategy>, BoundaryStrategy,
                                 ConstraintStrategy>;
    using Base::Base;

    template <typename FluxNumberType>
    void note_advection_denominator(const double, const FluxNumberType &) const
    {}

    template <typename FluxNumberType>
    void note_diffusion_denominator(const double, const FluxNumberType &) const
    {}
  };

  template <template <typename> typename BoundaryStrategy,
            template <typename> typename ConstraintStrategy = def::NoAffineConstraints>
  class ONKTDiagnosticModel : public ONKTModelCommon<ONKTDiagnosticModel<BoundaryStrategy, ConstraintStrategy>,
                                                     BoundaryStrategy, ConstraintStrategy>
  {
  public:
    using Base = ONKTModelCommon<ONKTDiagnosticModel<BoundaryStrategy, ConstraintStrategy>, BoundaryStrategy,
                                 ConstraintStrategy>;
    using Base::Base;

    void begin_diagnostics() const { diagnostics = Diagnostics{}; }

    void diagnose_state(std::ostream &out, const std::vector<Point<dim>> &support_points, const VectorType &solution,
                        const VectorType &, const VectorType &) const
    {
      out << "KT diagnostic: u/sigma denominator from assembled flux";
      if (diagnostics.advection_seen)
        out << " value=" << diagnostics.advection_denominator << " abs=" << diagnostics.min_abs_advection_denominator
            << " sigma=" << diagnostics.advection_sigma << " u=" << diagnostics.advection_u;
      else
        out << " unavailable";
      out << '\n';

      out << "KT diagnostic: r+du/dsigma denominator from assembled flux";
      if (diagnostics.diffusion_seen)
        out << " value=" << diagnostics.diffusion_denominator << " abs=" << diagnostics.min_abs_diffusion_denominator
            << " sigma=" << diagnostics.diffusion_sigma << " du_dsigma=" << diagnostics.diffusion_du_dsigma;
      else
        out << " unavailable";
      out << '\n';

      if (solution.size() == support_points.size() && !support_points.empty()) {
        std::size_t min_u_over_sigma_index = 0;
        std::size_t min_du_index = 0;
        double min_u_over_sigma_denominator = std::numeric_limits<double>::infinity();
        double min_du_denominator = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < support_points.size(); ++i) {
          const double sigma = support_points[i][0];
          const double u = solution[i];
          const double u_over_sigma = std::abs(sigma) < origin_tol ? 0.0 : u / sigma;
          const double advection_denominator = this->k + u_over_sigma;
          if (std::abs(advection_denominator) < std::abs(min_u_over_sigma_denominator)) {
            min_u_over_sigma_denominator = advection_denominator;
            min_u_over_sigma_index = i;
          }

          double du_dsigma = 0.0;
          if (support_points.size() > 1) {
            if (i == 0)
              du_dsigma = (solution[1] - solution[0]) / (support_points[1][0] - support_points[0][0]);
            else if (i + 1 == support_points.size())
              du_dsigma = (solution[i] - solution[i - 1]) / (support_points[i][0] - support_points[i - 1][0]);
            else
              du_dsigma =
                  (solution[i + 1] - solution[i - 1]) / (support_points[i + 1][0] - support_points[i - 1][0]);
          }
          const double diffusion_denominator = this->k + du_dsigma;
          if (std::abs(diffusion_denominator) < std::abs(min_du_denominator)) {
            min_du_denominator = diffusion_denominator;
            min_du_index = i;
          }
        }

        out << "KT diagnostic: cellwise min(r + u/sigma) dof=" << min_u_over_sigma_index
            << " sigma=" << support_points[min_u_over_sigma_index][0] << " value=" << min_u_over_sigma_denominator
            << " u=" << solution[min_u_over_sigma_index] << '\n';
        out << "KT diagnostic: cellwise min(r + estimated du/dsigma) dof=" << min_du_index
            << " sigma=" << support_points[min_du_index][0] << " value=" << min_du_denominator
            << " u=" << solution[min_du_index] << '\n';
      }
    }

    struct Diagnostics {
      double min_abs_advection_denominator = std::numeric_limits<double>::infinity();
      double advection_denominator = std::numeric_limits<double>::quiet_NaN();
      double advection_sigma = std::numeric_limits<double>::quiet_NaN();
      double advection_u = std::numeric_limits<double>::quiet_NaN();
      bool advection_seen = false;

      double min_abs_diffusion_denominator = std::numeric_limits<double>::infinity();
      double diffusion_denominator = std::numeric_limits<double>::quiet_NaN();
      double diffusion_sigma = std::numeric_limits<double>::quiet_NaN();
      double diffusion_du_dsigma = std::numeric_limits<double>::quiet_NaN();
      bool diffusion_seen = false;
    };

    template <typename FluxNumberType> void note_advection_denominator(const double sigma, const FluxNumberType &u) const
    {
      const double u_value = scalar_value(u);
      const double u_over_sigma = std::abs(sigma) < origin_tol ? 0.0 : u_value / sigma;
      const double denominator = this->k + u_over_sigma;
      if (!diagnostics.advection_seen || std::abs(denominator) < diagnostics.min_abs_advection_denominator) {
        diagnostics.min_abs_advection_denominator = std::abs(denominator);
        diagnostics.advection_denominator = denominator;
        diagnostics.advection_sigma = sigma;
        diagnostics.advection_u = u_value;
        diagnostics.advection_seen = true;
      }
    }

    template <typename FluxNumberType>
    void note_diffusion_denominator(const double sigma, const FluxNumberType &du_dsigma) const
    {
      const double du_dsigma_value = scalar_value(du_dsigma);
      const double denominator = this->k + du_dsigma_value;
      if (!diagnostics.diffusion_seen || std::abs(denominator) < diagnostics.min_abs_diffusion_denominator) {
        diagnostics.min_abs_diffusion_denominator = std::abs(denominator);
        diagnostics.diffusion_denominator = denominator;
        diagnostics.diffusion_sigma = sigma;
        diagnostics.diffusion_du_dsigma = du_dsigma_value;
        diagnostics.diffusion_seen = true;
      }
    }

  private:
    mutable Diagnostics diagnostics;
  };

  using ONKTModel = ONKTModelBase<def::OriginOddLinearExtrapolationBoundaries, ConstrainUAtOrigin>;
  using ONKTDiagnosticModelOriginConstrained =
      ONKTDiagnosticModel<def::OriginOddLinearExtrapolationBoundaries, ConstrainUAtOrigin>;
  using ONSymmetricDefaultBoundaryDiagnosticModel =
      ONKTDiagnosticModel<def::FVDefaultBoundaries, ConstrainUAtOrigin>;

  // ===========================================================================
  // Integrator-based per-mode loop kernels (pion piece and sigma piece of the
  // finite-T LPA flow with a PolynomialExp regulator). These mirror the
  // kernels at Examples/ONfiniteT/flows/V_{pion,sigma}/kernel.hh that the
  // user-facing example uses.
  //
  // The smoke test below reproduces exactly that integrator-based setup so we
  // can probe whether the KT + IDA failure observed in
  // Examples/ONfiniteT_KT/ is reproducible from inside the regression suite.
  // ===========================================================================

  template <typename _Regulator> class V_pion_kernel_integrator
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto & /*N*/,
                                                   const auto &T, const auto &m2Pi)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const auto _interp1 = Regulator::RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = CothFiniteT(sqrt(powr<2>(l1) + m2Pi + _interp1), T);
      const auto _interp3 = Regulator::RBdot(powr<2>(k), powr<2>(l1));
      const auto _cse1 = powr<2>(l1);
      return (0.25) * (_interp2) * (_interp3) * (sqrt(powr<-1>(_cse1 + _interp1 + m2Pi)));
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto & /*k*/, const auto & /*N*/, const auto & /*T*/,
                                                     const auto & /*m2Pi*/)
    {
      return 0.;
    }
  };

  template <typename _Regulator> class V_sigma_kernel_integrator
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto & /*N*/,
                                                   const auto &T, const auto &m2Sigma)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const auto _interp1 = Regulator::RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = Regulator::RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = CothFiniteT(sqrt(powr<2>(l1) + m2Sigma + _interp1), T);
      const auto _cse1 = powr<2>(l1);
      return (0.25) * (_interp3) * (_interp4) * (sqrt(powr<-1>(_cse1 + _interp1 + m2Sigma)));
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto & /*k*/, const auto & /*N*/, const auto & /*T*/,
                                                     const auto & /*m2Sigma*/)
    {
      return 0.;
    }
  };

  // Wrapper that owns the per-NumberType integrator instances. The Real<2, double>
  // variant is required because KT's compute_flux_jacobian_and_hessian seeds
  // second-order AD for the advection-flux Hessian.
  template <typename Kernel> class IntegratorWrapper
  {
  public:
    using Regulator = DiFfRG::PolynomialExpRegulator<>;

    IntegratorWrapper(DiFfRG::QuadratureProvider &qp, const DiFfRG::JSONValue &json)
        : integrator(qp, json), integrator_AD(qp, json), integrator_AD2(qp, json)
    {
    }

    DiFfRG::Integrator_p2<3, double, Kernel, DiFfRG::TBB_exec> integrator;
    DiFfRG::Integrator_p2<3, autodiff::real, Kernel, DiFfRG::TBB_exec> integrator_AD;
    DiFfRG::Integrator_p2<3, autodiff::Real<2, double>, Kernel, DiFfRG::TBB_exec> integrator_AD2;

    void get(double &dest, const double &k, const double &N, const double &T, const double &m2)
    {
      integrator.get(dest, k, N, T, m2);
    }
    void get(autodiff::real &dest, const double &k, const double &N, const double &T, const autodiff::real &m2)
    {
      integrator_AD.get(dest, k, N, T, m2);
    }
    void get(autodiff::Real<2, double> &dest, const double &k, const double &N, const double &T,
             const autodiff::Real<2, double> &m2)
    {
      integrator_AD2.get(dest, k, N, T, m2);
    }
  };

  using V_pion_integrator_t = IntegratorWrapper<V_pion_kernel_integrator<DiFfRG::PolynomialExpRegulator<>>>;
  using V_sigma_integrator_t = IntegratorWrapper<V_sigma_kernel_integrator<DiFfRG::PolynomialExpRegulator<>>>;

  // T=0 / Litim variants of the same per-mode kernels — used to compare the
  // integrator+AD path against the analytical closed-form on identical
  // physics (Litim regulator, T=0). The CothFiniteT thermal factor reduces
  // to +1 at T=0 and is therefore omitted. Everything else (the
  // `0.25 · RBdot · sqrt(1/(l1² + RB + m²))` form) matches the PolynomialExp
  // counterpart so we can A/B-test regulator+T choice.
  template <typename _Regulator> class V_pion_kernel_litim_T0
  {
  public:
    using Regulator = _Regulator;
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto & /*N*/,
                                                   const auto & /*T*/, const auto &m2Pi)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const auto _interp1 = Regulator::RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = Regulator::RBdot(powr<2>(k), powr<2>(l1));
      const auto _cse1 = powr<2>(l1);
      return (0.25) * (_interp3) * (sqrt(powr<-1>(_cse1 + _interp1 + m2Pi)));
    }
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto & /*k*/, const auto & /*N*/, const auto & /*T*/,
                                                     const auto & /*m2Pi*/) { return 0.; }
  };

  template <typename _Regulator> class V_sigma_kernel_litim_T0
  {
  public:
    using Regulator = _Regulator;
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const auto &k, const auto & /*N*/,
                                                   const auto & /*T*/, const auto &m2Sigma)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const auto _interp1 = Regulator::RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = Regulator::RBdot(powr<2>(k), powr<2>(l1));
      const auto _cse1 = powr<2>(l1);
      return (0.25) * (_interp3) * (sqrt(powr<-1>(_cse1 + _interp1 + m2Sigma)));
    }
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto & /*k*/, const auto & /*N*/, const auto & /*T*/,
                                                     const auto & /*m2Sigma*/) { return 0.; }
  };

  using V_pion_litim_integrator_t = IntegratorWrapper<V_pion_kernel_litim_T0<DiFfRG::LitimRegulator<>>>;
  using V_sigma_litim_integrator_t = IntegratorWrapper<V_sigma_kernel_litim_T0<DiFfRG::LitimRegulator<>>>;

  class ONIntegratorLitimFlows
  {
  public:
    ONIntegratorLitimFlows(const DiFfRG::JSONValue &json)
        : quadrature_provider(json), V_pion(quadrature_provider, json), V_sigma(quadrature_provider, json)
    {
    }
    void set_k(double k)
    {
      DiFfRG::all_set_k(V_pion, k);
      DiFfRG::all_set_k(V_sigma, k);
    }
    void set_T(double T)
    {
      DiFfRG::all_set_T(V_pion, T);
      DiFfRG::all_set_T(V_sigma, T);
    }

    DiFfRG::QuadratureProvider quadrature_provider;
    V_pion_litim_integrator_t V_pion;
    V_sigma_litim_integrator_t V_sigma;
  };

  // Container for the flow integrators. Owns its own QuadratureProvider.
  class ONIntegratorFlows
  {
  public:
    ONIntegratorFlows(const DiFfRG::JSONValue &json)
        : quadrature_provider(json), V_pion(quadrature_provider, json), V_sigma(quadrature_provider, json)
    {
    }
    void set_k(double k)
    {
      DiFfRG::all_set_k(V_pion, k);
      DiFfRG::all_set_k(V_sigma, k);
    }
    void set_T(double T)
    {
      DiFfRG::all_set_T(V_pion, T);
      DiFfRG::all_set_T(V_sigma, T);
    }

    DiFfRG::QuadratureProvider quadrature_provider;
    V_pion_integrator_t V_pion;
    V_sigma_integrator_t V_sigma;
  };

  // Integrator-based KT O(N) model in sigma-coordinates. Mirrors
  // Examples/ONfiniteT_KT/model_sigma.hh:
  //   F_adv  = (N-1) * V_pion(m^2_Pi = u/sigma)
  //   F_diff = V_sigma(m^2_Sigma = du/dsigma)
  // with the same OriginOdd / ConstrainUAtOrigin / def::AD inheritance as
  // ONKTModel, but with the analytic flux replaced by the integrator.
  class ONIntegratorKTModel : public def::AbstractModel<ONIntegratorKTModel, Components>,
                              public def::fRG,
                              public def::OriginOddLinearExtrapolationBoundaries<ONIntegratorKTModel>,
                              public ConstrainUAtOrigin<ONIntegratorKTModel>,
                              public def::AD<ONIntegratorKTModel>
  {
  public:
    ONIntegratorKTModel(const JSONValue &json, const FlowCase &flow_case, const GridSettings &grid_settings)
        : def::fRG(json), flow_case_(flow_case), cell_width_(grid_spacing(grid_settings)), flows_(json)
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(LSMPhysicalParameters::T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double sigma = x[0];
      const double right = scenario_potential(flow_case_.scenario, sigma + 0.5 * cell_width_);
      const double left = scenario_potential(flow_case_.scenario, sigma - 0.5 * cell_width_);
      values[0] = (right - left) / cell_width_;
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * Lambda;
      flows_.set_k(k);
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void KurganovTadmor_advection_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i,
                                       const Point<spatial_dim> &x, const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "ONIntegratorKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "ONIntegratorKTModel expects a single FE function.");

      const auto &fe_functions = get<0>(sol);
      const NT u = fe_functions[0];
      const double sigma = x[0];
      const NT m2Pi = (std::abs(sigma) < origin_tol) ? NT(0.0) : (u / NT(sigma));

      NT pion_loop;
      flows_.V_pion.get(pion_loop, k, flow_case_.n_flavors, LSMPhysicalParameters::T, m2Pi);
      F_i[0][0] = (flow_case_.n_flavors - 1.0) * pion_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> & /*x*/,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "ONIntegratorKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "ONIntegratorKTModel expects a single FE function.");

      const auto &fe_derivatives = get<1>(sol);
      const NT m2Sigma = fe_derivatives[0][0];

      NT sigma_loop;
      flows_.V_sigma.get(sigma_loop, k, flow_case_.n_flavors, LSMPhysicalParameters::T, m2Sigma);
      F_i[0][0] = sigma_loop;
    }

  private:
    FlowCase flow_case_;
    double cell_width_;
    mutable ONIntegratorFlows flows_;
  };

  // Same model as ONIntegratorKTModel but with Litim regulator at T=0,
  // matching the analytic ONKTModel's physics exactly. Used to isolate
  // whether KT+IDA failure is in the integrator/AD path itself (rather than
  // in the PolynomialExp+finite-T choice).
  class ONIntegratorLitimKTModel : public def::AbstractModel<ONIntegratorLitimKTModel, Components>,
                                   public def::fRG,
                                   public def::OriginOddLinearExtrapolationBoundaries<ONIntegratorLitimKTModel>,
                                   public ConstrainUAtOrigin<ONIntegratorLitimKTModel>,
                                   public def::AD<ONIntegratorLitimKTModel>
  {
  public:
    ONIntegratorLitimKTModel(const JSONValue &json, const FlowCase &flow_case, const GridSettings &grid_settings)
        : def::fRG(json), flow_case_(flow_case), cell_width_(grid_spacing(grid_settings)), flows_(json)
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(0.0);  // Litim closed-form is T=0
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double sigma = x[0];
      const double right = scenario_potential(flow_case_.scenario, sigma + 0.5 * cell_width_);
      const double left = scenario_potential(flow_case_.scenario, sigma - 0.5 * cell_width_);
      values[0] = (right - left) / cell_width_;
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * Lambda;
      flows_.set_k(k);
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void KurganovTadmor_advection_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i,
                                       const Point<spatial_dim> &x, const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "ONIntegratorLitimKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "ONIntegratorLitimKTModel expects a single FE function.");

      const auto &fe_functions = get<0>(sol);
      const NT u = fe_functions[0];
      const double sigma = x[0];
      const NT m2Pi = (std::abs(sigma) < origin_tol) ? NT(0.0) : (u / NT(sigma));

      NT pion_loop;
      flows_.V_pion.get(pion_loop, k, flow_case_.n_flavors, 0.0, m2Pi);
      F_i[0][0] = (flow_case_.n_flavors - 1.0) * pion_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> & /*x*/,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "ONIntegratorLitimKTModel is one-dimensional.");
      static_assert(n_fe_functions == 1, "ONIntegratorLitimKTModel expects a single FE function.");

      const auto &fe_derivatives = get<1>(sol);
      const NT m2Sigma = fe_derivatives[0][0];

      NT sigma_loop;
      flows_.V_sigma.get(sigma_loop, k, flow_case_.n_flavors, 0.0, m2Sigma);
      F_i[0][0] = sigma_loop;
    }

  private:
    FlowCase flow_case_;
    double cell_width_;
    mutable ONIntegratorLitimFlows flows_;
  };

  // ===========================================================================
  // CG comparison model — same physics as ONIntegratorKTModel (PolynomialExp
  // T=0.05 integrator-based flux, same V_pion / V_sigma kernels, same LSM
  // initial potential, same Lambda), but using a CG discretisation rather
  // than KT.
  //
  // The point of the comparison: KT's Jacobian path seeds Real<2, double>
  // through the integrator (for the wave-speed Hessian), CG only seeds
  // Real<1, double>. If KT fails and CG passes on identical physics, the
  // second-order-AD-through-quadrature is the precision bottleneck.
  //
  // CG only calls `flux()`, so the advection and diffusion pieces are
  // combined into a single flux here: F = (N-1)·V_pion(u/σ) + V_sigma(∂u/∂σ).
  // ===========================================================================
  class ON_CG_LSM_Model : public def::AbstractModel<ON_CG_LSM_Model, Components>,
                          public def::fRG,
                          public def::LLFFlux<ON_CG_LSM_Model>,
                          public def::FlowBoundaries<ON_CG_LSM_Model>,
                          public ConstrainUAtOrigin<ON_CG_LSM_Model>,
                          public def::AD<ON_CG_LSM_Model>
  {
  public:
    ON_CG_LSM_Model(const JSONValue &json, const FlowCase &flow_case, const GridSettings &grid_settings)
        : def::fRG(json), flow_case_(flow_case), cell_width_(grid_spacing(grid_settings)), flows_(json)
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(LSMPhysicalParameters::T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      const double sigma = x[0];
      const double right = scenario_potential(flow_case_.scenario, sigma + 0.5 * cell_width_);
      const double left = scenario_potential(flow_case_.scenario, sigma - 0.5 * cell_width_);
      values[0] = (right - left) / cell_width_;
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * Lambda;
      flows_.set_k(k);
    }

    // Combined CG flux: F = (N-1)·V_pion(u/σ) + V_sigma(∂u/∂σ).
    // Same origin guard as ONIntegratorKTModel for the u/σ singularity at σ=0.
    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F, const Point<dim> &x,
              const Solution &sol) const
    {
      const auto &fe_functions = get<"fe_functions">(sol);
      const auto &fe_derivatives = get<"fe_derivatives">(sol);

      const NT u = fe_functions[0];
      const NT du_dsigma = fe_derivatives[0][0];
      const double sigma = x[0];
      const NT m2Pi = (std::abs(sigma) < origin_tol) ? NT(0.0) : (u / NT(sigma));
      const NT m2Sigma = du_dsigma;

      NT pion_loop, sigma_loop;
      flows_.V_pion.get(pion_loop, k, flow_case_.n_flavors, LSMPhysicalParameters::T, m2Pi);
      flows_.V_sigma.get(sigma_loop, k, flow_case_.n_flavors, LSMPhysicalParameters::T, m2Sigma);
      F[0][0] = (flow_case_.n_flavors - 1.0) * pion_loop + sigma_loop;
    }

  private:
    FlowCase flow_case_;
    double cell_width_;
    mutable ONIntegratorFlows flows_;
  };

  // ===========================================================================
  // KT ρ-coordinate variant of the same physics. FE variable is now
  //   m²(ρ) = ∂V/∂ρ
  // (with ρ = σ²/2). The LSM initial potential V_init(σ) = (m²/2)σ² + (λ/8)σ⁴
  // becomes V_init(ρ) = m²·ρ + (λ/2)·ρ², so the initial profile is
  //   m²_init(ρ) = m² + λ·ρ
  // (linear, hence cell-averaged value equals point value at cell centre).
  //
  // KT split in ρ-form:
  //   F_adv  = (N-1) · V_pion(m²_Pi = m²)            — depends on u only
  //   F_diff = V_sigma(m²_Sigma = m² + 2ρ · ∂m²/∂ρ)  — depends on u and grad u
  //
  // Note the *2ρ* amplifier in F_diff at large ρ — this is the dual of the
  // 1/σ amplifier the σ-form KT test exhibits at small σ.
  // ===========================================================================
  class ON_KT_RHO_LSM_Model : public def::AbstractModel<ON_KT_RHO_LSM_Model, Components>,
                              public def::fRG,
                              public def::FVDefaultBoundaries<ON_KT_RHO_LSM_Model>,
                              public def::AD<ON_KT_RHO_LSM_Model>
  {
  public:
    ON_KT_RHO_LSM_Model(const JSONValue &json, const FlowCase &flow_case, const GridSettings & /*grid_settings*/)
        : def::fRG(json), flow_case_(flow_case), flows_(json)
    {
      flows_.set_k(this->Lambda);
      flows_.set_T(LSMPhysicalParameters::T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      // m²(ρ) = m² + λ·ρ. Linear in ρ ⇒ cell-averaged value == point value.
      const double rho = x[0];
      values[0] = LSMPhysicalParameters::m2 + LSMPhysicalParameters::lambda_quartic * rho;
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * Lambda;
      flows_.set_k(k);
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void KurganovTadmor_advection_flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i,
                                       const Point<spatial_dim> & /*x*/, const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "ON_KT_RHO_LSM_Model is one-dimensional.");
      static_assert(n_fe_functions == 1, "ON_KT_RHO_LSM_Model expects a single FE function.");

      const auto &fe_functions = get<0>(sol);
      const NT m2Pi = fe_functions[0];  // m²_Pi = m² in ρ-form (no 1/σ)

      NT pion_loop;
      flows_.V_pion.get(pion_loop, k, flow_case_.n_flavors, LSMPhysicalParameters::T, m2Pi);
      F_i[0][0] = (flow_case_.n_flavors - 1.0) * pion_loop;
    }

    template <int spatial_dim, typename NT, typename Solutions, std::size_t n_fe_functions>
    void flux(std::array<Tensor<1, spatial_dim, NT>, n_fe_functions> &F_i, const Point<spatial_dim> &x,
              const Solutions &sol) const
    {
      static_assert(spatial_dim == dim, "ON_KT_RHO_LSM_Model is one-dimensional.");
      static_assert(n_fe_functions == 1, "ON_KT_RHO_LSM_Model expects a single FE function.");

      const auto &fe_functions = get<0>(sol);
      const auto &fe_derivatives = get<1>(sol);
      const double rho = x[0];

      // m²_Sigma = m² + 2ρ·∂m²/∂ρ. The 2ρ factor is the geometric amplifier.
      const NT m2Sigma = fe_functions[0] + NT(2.0 * rho) * fe_derivatives[0][0];

      NT sigma_loop;
      flows_.V_sigma.get(sigma_loop, k, flow_case_.n_flavors, LSMPhysicalParameters::T, m2Sigma);
      F_i[0][0] = sigma_loop;
    }

  private:
    FlowCase flow_case_;
    mutable ONIntegratorFlows flows_;
  };

  JSONValue make_json(const FlowCase &flow_case, const int threads = 8, const int fe_order = 0)
  {
    return json::value(
        {{"physical", {{"Lambda", flow_case.lambda}}},
         // /integration block is required by the integrator-based smoke test below
         // (Integrator_p2 reads x_order and x_extent_tolerance). The existing
         // analytic-flux scenarios ignore these keys.
         {"integration",
          {{"x_order", 32},
           {"x_extent_tolerance", 1.0e-3},
           {"jacobian_quadrature_factor", 0.5}}},
         {"discretization",
          // fe_order=0 for FV cells; pass fe_order>0 for the CG comparison test.
          {{"fe_order", fe_order},
           {"threads", threads},
           {"batch_size", 64},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"output_buffer_size", 1},
           {"EoM_abs_tol", 1e-10},
           {"EoM_max_iter", 0}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", 2.0},
           {"explicit",
            {{"dt", 1e-8}, {"minimal_dt", 1e-12}, {"maximal_dt", 1.0}, {"abs_tol", 1e-12}, {"rel_tol", 1e-10}}},
           {"implicit",
            {{"dt", 1e-8},
             {"minimal_dt", 1e-14},
             {"maximal_dt", 1.0},
             {"abs_tol", 1e-7},
             {"rel_tol", 1e-7},
             {"max_steps", 5000000}}}}},
         {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", false}}}});
  }

  std::filesystem::path local_fixture_path(const std::string_view relative_path)
  {
    return std::filesystem::path(__FILE__).parent_path() / std::filesystem::path(relative_path);
  }

  std::vector<double> reference_times()
  {
    std::vector<double> times;
    for (std::size_t i = 0; i <= 30; ++i)
      times.push_back(2.0 * static_cast<double>(i));
    return times;
  }

  ReferenceFlowData load_reference_flow(const FlowCase &flow_case)
  {
    const auto fixture = local_fixture_path(flow_case.relative_path);
    REQUIRE(std::filesystem::exists(fixture));

    const JSONValue fixture_json(fixture.string());
    json::value fixture_value = static_cast<json::value>(fixture_json);
    const auto &root = fixture_value.as_array();

    ReferenceFlowData data;
    const auto times = reference_times();

    if (flow_case.profile_group_index.has_value()) {
      REQUIRE(root.size() >= 3);
      const auto &profiles = root.at(*flow_case.profile_group_index).as_array();
      REQUIRE(profiles.size() == times.size());

      data.snapshots.reserve(times.size());
      for (std::size_t i = 0; i < times.size(); ++i)
        data.snapshots.push_back(ReferenceSnapshot{times[i], std::nullopt, parse_profile(profiles[i].as_array())});
      return data;
    }

    REQUIRE(root.size() == 2);
    const auto &U_profiles = root.at(0).as_array();
    const auto &u_profiles = root.at(1).as_array();
    REQUIRE(U_profiles.size() == times.size());
    REQUIRE(u_profiles.size() == times.size());

    data.snapshots.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
      data.snapshots.push_back(
          ReferenceSnapshot{times[i], parse_profile(U_profiles[i].as_array()), parse_profile(u_profiles[i].as_array())});
    return data;
  }

  std::vector<double> reconstruct_potential_from_u(const std::vector<double> &x_values,
                                                   const std::vector<double> &u_values)
  {
    REQUIRE(x_values.size() == u_values.size());

    std::vector<double> U_values(u_values.size(), 0.0);
    for (std::size_t i = 1; i < u_values.size(); ++i) {
      const double dx = x_values[i] - x_values[i - 1];
      U_values[i] = U_values[i - 1] + 0.5 * dx * (u_values[i - 1] + u_values[i]);
    }
    return U_values;
  }

  struct SimulationResult {
    std::vector<double> x;
    std::vector<double> u;
    std::vector<double> U;
  };

  Config::ConfigurationMesh<1> make_mesh_config(const GridSettings &grid_settings = default_grid_settings())
  {
    const double delta = grid_spacing(grid_settings);
    const Config::GridAxis sigma_axis(grid_settings.sigma_min - 0.5 * delta, delta, grid_settings.sigma_max + 0.5 * delta);
    return Config::ConfigurationMesh<1>(0u, std::vector<Config::GridAxis>{sigma_axis});
  }

  void initialize_exact_cell_averages(FV::FlowingVariables<Discretization> &state, const std::vector<Point<dim>> &support_points,
                                      const FlowCase &flow_case, const GridSettings &grid_settings)
  {
    auto &u = state.spatial_data();
    REQUIRE(u.size() == support_points.size());
    const double delta = grid_spacing(grid_settings);
    for (unsigned int i = 0; i < u.size(); ++i) {
      const double sigma = support_points[i][0];
      const double right = scenario_potential(flow_case.scenario, sigma + 0.5 * delta);
      const double left = scenario_potential(flow_case.scenario, sigma - 0.5 * delta);
      u[i] = (right - left) / delta;
    }
  }

  SimulationResult sample_state(const FV::FlowingVariables<Discretization> &state, const Discretization &discretization)
  {
    const SampledProfile sampled_u = kt_regression::sample_sorted_profile(state, discretization, grid_tol);
    SimulationResult result;
    result.x = sampled_u.x;
    result.u = sampled_u.y;
    result.U = reconstruct_potential_from_u(result.x, result.u);
    return result;
  }

  SimulationResult restrict_to_nonnegative_half(const SimulationResult &full_domain_result)
  {
    SimulationResult result;
    for (std::size_t i = 0; i < full_domain_result.x.size(); ++i) {
      if (full_domain_result.x[i] < -grid_tol) continue;
      result.x.push_back(full_domain_result.x[i]);
      result.u.push_back(full_domain_result.u[i]);
    }
    if (!result.x.empty() && std::abs(result.x.front()) < grid_tol) result.u.front() = 0.0;
    result.U = reconstruct_potential_from_u(result.x, result.u);
    return result;
  }

  template <typename ModelType = ONKTModel>
  std::vector<SimulationResult> run_flow_snapshots(const FlowCase &flow_case, const std::vector<double> &times_to_sample,
                                                   const GridSettings &grid_settings = default_grid_settings(),
                                                   const int threads = 8)
  {
    const JSONValue json = make_json(flow_case, threads);
    ModelType model(json, flow_case, grid_settings);
    Mesh mesh(make_mesh_config(grid_settings));
    Discretization discretization(mesh, json);
    Assembler<ModelType> assembler(discretization, model, json);

    kt_regression::TemporaryDirectory tmp_dir("on_model_kt_regression");
    DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), "on_model_kt_regression", "output", json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == grid_settings.cells);

    std::vector<double> initial_support_points;
    initial_support_points.reserve(support_points.size());
    for (const auto &point : support_points)
      initial_support_points.push_back(point[0]);
    std::sort(initial_support_points.begin(), initial_support_points.end());

    const double delta = grid_spacing(grid_settings);
    for (std::size_t i = 0; i < initial_support_points.size(); ++i)
      REQUIRE(initial_support_points[i] ==
              Catch::Approx(grid_settings.sigma_min + static_cast<double>(i) * delta).margin(grid_tol));

    initialize_exact_cell_averages(state, support_points, flow_case, grid_settings);

    std::vector<SimulationResult> snapshots;
    snapshots.reserve(times_to_sample.size());
    double current_time = 0.0;
    for (const double target_time : times_to_sample) {
      REQUIRE(target_time >= current_time - 1e-12);
      if (target_time > current_time + 1e-12) {
        try {
          time_stepper.run(&state, current_time, target_time);
        } catch (const std::exception &exception) {
          FAIL(std::string("O(N) KT regression solve failed at t=") + std::to_string(target_time) + ": " +
               exception.what());
        }
        current_time = target_time;
      }
      snapshots.push_back(sample_state(state, discretization));
    }

    return snapshots;
  }

  std::vector<std::size_t> selected_snapshot_indices()
  {
    return {30};
  }

  template <typename ModelType = ONKTModel>
  void run_flow_to_time(const FlowCase &flow_case, const double target_time,
                        const GridSettings &grid_settings = default_grid_settings(),
                        const int threads = 8)
  {
    const JSONValue json = make_json(flow_case, threads);
    ModelType model(json, flow_case, grid_settings);
    Mesh mesh(make_mesh_config(grid_settings));
    Discretization discretization(mesh, json);
    Assembler<ModelType> assembler(discretization, model, json);

    kt_regression::TemporaryDirectory tmp_dir("on_model_kt_regression");
    DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), "on_model_kt_regression", "output", json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == grid_settings.cells);
    initialize_exact_cell_averages(state, support_points, flow_case, grid_settings);

    time_stepper.run(&state, 0.0, target_time);
  }

  // KT counterpart of run_flow_to_time but explicitly parameterised on the
  // WaveSpeedStrategy (the existing run_flow_to_time hard-codes ZeroDeriv via the
  // file-level `Assembler` alias). Lets test cases compare
  // MaxEigenvalueWaveSpeed (Real<2> AD H) vs MaxEigenvalueWaveSpeedFD (Real<1>
  // AD + central FD H) vs MaxEigenvalueWaveSpeedZeroDeriv (no H) on identical
  // physics.
  template <typename ModelType, typename WaveSpeedStrategy>
  void run_flow_to_time_ws(const FlowCase &flow_case, const double target_time,
                           const GridSettings &grid_settings = default_grid_settings(),
                           const int threads = 8)
  {
    using AssemblerWS = FV::KurganovTadmor::Assembler<Discretization, ModelType, Reconstructor, WaveSpeedStrategy>;

    const JSONValue json = make_json(flow_case, threads);
    ModelType model(json, flow_case, grid_settings);
    Mesh mesh(make_mesh_config(grid_settings));
    Discretization discretization(mesh, json);
    AssemblerWS assembler(discretization, model, json);

    kt_regression::TemporaryDirectory tmp_dir("on_model_kt_regression");
    DataOutput<dim, VectorType> data_out(tmp_dir.path.string(), "on_model_kt_regression", "output", json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == grid_settings.cells);
    initialize_exact_cell_averages(state, support_points, flow_case, grid_settings);

    time_stepper.run(&state, 0.0, target_time);
  }

  // CG counterpart of run_flow_to_time. Same lifetime/structure but instantiates
  // CG::Discretization and CG::Assembler, and lets the model.initial_condition
  // do the cell-averaged interpolation (no FV-style exact cell-average override
  // — CG support points are vertices/internal nodes, not cell centres).
  template <typename ModelType>
  void run_flow_to_time_cg(const FlowCase &flow_case, const double target_time,
                           const GridSettings &grid_settings = default_grid_settings(),
                           const int threads = 8, const int fe_order = 4)
  {
    using CGDiscretization = CG::Discretization<Components, NumberType, Mesh>;
    using CGAssembler = CG::Assembler<CGDiscretization, ModelType>;
    using CGTimeStepper =
        TimeStepperSUNDIALS_IDA<typename CGDiscretization::VectorType,
                                typename CGDiscretization::SparseMatrixType, dim, UMFPack>;

    const JSONValue json = make_json(flow_case, threads, fe_order);
    ModelType model(json, flow_case, grid_settings);
    Mesh mesh(make_mesh_config(grid_settings));
    CGDiscretization discretization(mesh, json);
    CGAssembler assembler(discretization, model, json);

    kt_regression::TemporaryDirectory tmp_dir("on_model_kt_regression_cg");
    DataOutput<dim, typename CGDiscretization::VectorType> data_out(
        tmp_dir.path.string(), "on_model_kt_regression_cg", "output", json);
    auto adaptor = std::make_unique<NoAdaptivity<typename CGDiscretization::VectorType>>();
    CGTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FE::FlowingVariables state(discretization);
    state.interpolate(model);

    time_stepper.run(&state, 0.0, target_time);
  }

  template <typename FlowParam>
  struct ONFlowFixture {
    static FlowCase flow_case() { return FlowParam::flow_case(); }

    void run_regression()
    {
      const FlowCase current_flow = flow_case();
      INFO(current_flow.label << ", Lambda=" << current_flow.lambda);

      const auto reference = load_reference_flow(current_flow);
      const auto indices = selected_snapshot_indices();

      std::vector<double> times_to_sample;
      times_to_sample.reserve(indices.size());
      for (const auto index : indices) {
        REQUIRE(index < reference.snapshots.size());
        times_to_sample.push_back(reference.snapshots[index].time);
      }

      const auto snapshots = run_flow_snapshots(current_flow, times_to_sample);
      REQUIRE(snapshots.size() == indices.size());

      for (std::size_t snapshot_index = 0; snapshot_index < indices.size(); ++snapshot_index) {
        const auto &reference_snapshot = reference.snapshots[indices[snapshot_index]];
        const auto &simulation = snapshots[snapshot_index];

        INFO("t=" << reference_snapshot.time);

        for (std::size_t i = 0; i < simulation.x.size(); ++i)
          REQUIRE(simulation.x[i] == Catch::Approx(reference_snapshot.u.x[i]).margin(grid_tol));

        if (const auto u_mismatch = kt_regression::compare_profiles(
                "u(sigma)", simulation.x, simulation.u, reference_snapshot.u.y, current_flow.profile_abs_tol,
                current_flow.profile_rel_tol))
          FAIL(*u_mismatch);

        if (current_flow.has_reference_potential) {
          REQUIRE(reference_snapshot.U.has_value());
          if (const auto U_mismatch = kt_regression::compare_profiles(
                  "U(sigma)", simulation.x, simulation.U, reference_snapshot.U->y, current_flow.profile_abs_tol,
                  current_flow.profile_rel_tol))
            FAIL(*U_mismatch);
        }
      }
    }
  };

} // namespace

TEMPLATE_TEST_CASE_METHOD(ONFlowFixture, "KT O(N) regressions match arXiv 2108.02504 flow snapshots",
                          "[FV][KT][ON][regression]", ScenarioI_ON1, ScenarioI_ON3, ScenarioI_ON10, ScenarioI_ON100,
                          ScenarioII_ON4, ScenarioIII_ON4, ScenarioIV_ON3)
{
  kt_regression::ensure_logger();

  this->run_regression();
}

TEST_CASE("KT O(N) symmetric default-boundary full-domain run matches the reference on sigma >= 0 - ScenarioI_ON3",
          "[FV][KT][ON][regression][symmetric-default-boundary][diagnostic]")
{
  kt_regression::ensure_logger();

  const auto flow_case = ScenarioI_ON3::flow_case();
  INFO(flow_case.label << ", symmetric/default-boundary diagnostic, Lambda=" << flow_case.lambda);

  const auto reference = load_reference_flow(flow_case);
  const auto indices = selected_snapshot_indices();

  std::vector<double> times_to_sample;
  times_to_sample.reserve(indices.size());
  for (const auto index : indices) {
    REQUIRE(index < reference.snapshots.size());
    times_to_sample.push_back(reference.snapshots[index].time);
  }

  GridSettings symmetric_grid{2 * n_cells - 1, -sigma_max, sigma_max};
  const auto full_domain_snapshots =
      run_flow_snapshots<ONSymmetricDefaultBoundaryDiagnosticModel>(flow_case, times_to_sample, symmetric_grid,
                                                                   diagnostic_threads);
  REQUIRE(full_domain_snapshots.size() == indices.size());

  for (std::size_t snapshot_index = 0; snapshot_index < indices.size(); ++snapshot_index) {
    const auto &reference_snapshot = reference.snapshots[indices[snapshot_index]];
    const auto full_domain_positive_half = restrict_to_nonnegative_half(full_domain_snapshots[snapshot_index]);

    INFO("t=" << reference_snapshot.time);

    REQUIRE(full_domain_positive_half.x.size() == reference_snapshot.u.x.size());
    for (std::size_t i = 0; i < full_domain_positive_half.x.size(); ++i)
      REQUIRE(full_domain_positive_half.x[i] == Catch::Approx(reference_snapshot.u.x[i]).margin(grid_tol));

    if (const auto u_mismatch = kt_regression::compare_profiles(
            "u_full(sigma >= 0)", full_domain_positive_half.x, full_domain_positive_half.u, reference_snapshot.u.y,
            flow_case.profile_abs_tol, flow_case.profile_rel_tol))
      FAIL(*u_mismatch);

    if (flow_case.has_reference_potential) {
      REQUIRE(reference_snapshot.U.has_value());
      if (const auto U_mismatch = kt_regression::compare_profiles(
              "U_full(sigma >= 0)", full_domain_positive_half.x, full_domain_positive_half.U, reference_snapshot.U->y,
              flow_case.profile_abs_tol, flow_case.profile_rel_tol))
        FAIL(*U_mismatch);
    }
  }
}

TEST_CASE("KT O(N) half-domain origin-constrained diagnostic solve reaches final time - ScenarioI_ON3",
          "[FV][KT][ON][regression][origin-constrained][diagnostic]")
{
  kt_regression::ensure_logger();

  const auto flow_case = ScenarioI_ON3::flow_case();
  INFO(flow_case.label << ", half-domain origin-constrained diagnostic, Lambda=" << flow_case.lambda);

  run_flow_to_time<ONKTDiagnosticModelOriginConstrained>(flow_case, final_time, default_grid_settings(),
                                                        diagnostic_threads);
}

TEST_CASE("KT O(N) half-domain run matches the symmetric full-domain run on sigma >= 0 - ScenarioI_ON3",
          "[FV][KT][ON][regression][symmetric-default-boundary][diagnostic]")
{
  kt_regression::ensure_logger();

  const auto flow_case = ScenarioI_ON3::flow_case();
  INFO(flow_case.label << ", half-vs-full-domain comparison, Lambda=" << flow_case.lambda);

  const std::vector<double> times_to_sample{0.0, 5.0, 10.0, 12.0, 15.0, 20.0};

  const auto half_domain_snapshots =
      run_flow_snapshots<ONKTDiagnosticModelOriginConstrained>(flow_case, times_to_sample, default_grid_settings(),
                                                              diagnostic_threads);
  GridSettings symmetric_grid{2 * n_cells - 1, -sigma_max, sigma_max};
  const auto full_domain_snapshots =
      run_flow_snapshots<ONSymmetricDefaultBoundaryDiagnosticModel>(flow_case, times_to_sample, symmetric_grid,
                                                                   diagnostic_threads);

  REQUIRE(half_domain_snapshots.size() == times_to_sample.size());
  REQUIRE(full_domain_snapshots.size() == times_to_sample.size());

  for (std::size_t snapshot_index = 0; snapshot_index < times_to_sample.size(); ++snapshot_index) {
    const auto &half_domain_snapshot = half_domain_snapshots[snapshot_index];
    const auto full_domain_positive_half = restrict_to_nonnegative_half(full_domain_snapshots[snapshot_index]);

    INFO("t=" << times_to_sample[snapshot_index]);

    REQUIRE(half_domain_snapshot.x.size() == full_domain_positive_half.x.size());
    for (std::size_t i = 0; i < half_domain_snapshot.x.size(); ++i)
      REQUIRE(half_domain_snapshot.x[i] == Catch::Approx(full_domain_positive_half.x[i]).margin(grid_tol));

    if (const auto u_mismatch = kt_regression::compare_profiles(
            "u_half(sigma) vs u_full(sigma)", half_domain_snapshot.x, half_domain_snapshot.u,
            full_domain_positive_half.u, flow_case.profile_abs_tol, flow_case.profile_rel_tol))
      FAIL(*u_mismatch);
  }
}

// Physical-scale smoke test:
// Runs the analytical KT O(N) Litim flow with IDA in the same parameter regime
// as Examples/ONfiniteT_KT/parameter.json (Lambda = 0.65, m2 = -0.1, lambda_quartic = 71.6,
// N = 2). The integrator-based finite-T variant of this physics fails IDA with
// a checkerboard instability (see /IDA_failure_diagnosis.md at the repo root).
// This test isolates whether the analytical-flux KT + IDA can handle the
// physical-scale regime at all. There is no reference profile to compare
// against; the test only asserts that IDA reaches `target_time` without
// throwing.
TEST_CASE("KT O(N) physical-scale LSM smoke test - Lambda=0.65, m2=-0.1, lambda=71.6, N=2 reaches final time",
          "[FV][KT][ON][smoke][physical-scale]")
{
  kt_regression::ensure_logger();

  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", physical-scale smoke test, Lambda=" << flow_case.lambda);

  constexpr double target_time = 4.0;  // matches Examples/ONfiniteT_KT/parameter.json
  run_flow_to_time(flow_case, target_time, default_grid_settings(), diagnostic_threads);
}

// Integrator-based counterpart of the previous smoke test. Same model, same
// initial potential, same Lambda; only the flux changes — from the analytic
// closed-form Litim expression used by ONKTModel above to the finite-T
// PolynomialExp quadrature-based flux used by the user-facing example
// Examples/ONfiniteT_KT/model_sigma.hh. If this test fails while the analytic
// one passes, that pins the KT + IDA failure on the integrator/AD interaction,
// not on the SSB physics or the Lambda regime.
TEST_CASE("KT O(N) physical-scale integrator-based smoke test - same physics, finite-T PolynomialExp quadrature flux",
          "[FV][KT][ON][smoke][integrator][physical-scale]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();  // required for QuadratureProvider construction

  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", integrator-based smoke test, Lambda=" << flow_case.lambda
                       << ", T=" << LSMPhysicalParameters::T);

  constexpr double target_time = 4.0;
  run_flow_to_time<ONIntegratorKTModel>(flow_case, target_time, default_grid_settings(), diagnostic_threads);
}

// Same integrator-based model and physics as above, but run on the SAME mesh
// used by Examples/ONfiniteT_KT/parameter_sigma.json (151 narrow cells from
// ~0 to ~0.174, dx ≈ 1.16e-3). The previous test uses the regression suite's
// default grid (800 wide cells over [0, 10], dx ≈ 0.0125) and succeeds. If
// this test fails where the previous one passed, the difference is the mesh
// — specifically that the 10× finer cells push the lowest cell centres so
// close to sigma = 0 that the 1/sigma geometric amplifier in the pion
// advection flux becomes O(several hundred) instead of O(40).
TEST_CASE("KT O(N) physical-scale integrator-based smoke test - example-style narrow mesh",
          "[FV][KT][ON][smoke][integrator][physical-scale][narrow-mesh]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", integrator-based smoke test on narrow mesh, Lambda=" << flow_case.lambda
                       << ", T=" << LSMPhysicalParameters::T);

  constexpr double target_time = 4.0;
  // Matches Examples/ONfiniteT_KT/parameter_sigma.json (after the cell-0-at-origin
  // grid shift): 151 cells of width ~1.16e-3, mesh centres at 0, dx, 2dx, ..., sigma_max.
  GridSettings narrow_grid{151, 0.0, 0.174};
  run_flow_to_time<ONIntegratorKTModel>(flow_case, target_time, narrow_grid, diagnostic_threads);
}

// Matched-physics quadrature counterpart of the analytic smoke test:
// Litim regulator, T=0, same LSM initial potential and Lambda. If this
// completes while the PolynomialExp+finite-T variants fail, the trigger is
// in the regulator/temperature choice (not the integrator/AD path).
TEST_CASE("KT O(N) physical-scale integrator-based smoke test - Litim regulator at T=0",
          "[FV][KT][ON][smoke][integrator][physical-scale][litim-T0]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", integrator-based Litim T=0 smoke test, Lambda=" << flow_case.lambda);

  constexpr double target_time = 4.0;
  run_flow_to_time<ONIntegratorLitimKTModel>(flow_case, target_time, default_grid_settings(), diagnostic_threads);
}

TEST_CASE("KT O(N) physical-scale integrator-based smoke test - Litim regulator at T=0, narrow mesh",
          "[FV][KT][ON][smoke][integrator][physical-scale][litim-T0][narrow-mesh]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", integrator-based Litim T=0 smoke test on narrow mesh, Lambda="
                       << flow_case.lambda);

  constexpr double target_time = 4.0;
  GridSettings narrow_grid{151, 0.0, 0.174};
  run_flow_to_time<ONIntegratorLitimKTModel>(flow_case, target_time, narrow_grid, diagnostic_threads);
}

// KT in ρ-coordinates for direct comparison with KT-σ and CG-ρ-combined.
// Same LSM physics, PolyExp T=0.05 integrator-based flux, V_pion / V_sigma
// split, but with m² = ∂V/∂ρ as the FE variable. The geometric amplifier
// is now 2ρ at large ρ (in m²_Sigma) instead of 1/σ at small σ.
TEST_CASE("KT O(N) physical-scale integrator-based smoke test - rho-coords",
          "[FV][KT][ON][smoke][integrator][physical-scale][kt-rho]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", KT rho-coords smoke test, Lambda=" << flow_case.lambda
                       << ", T=" << LSMPhysicalParameters::T);

  constexpr double target_time = 4.0;
  // rho_max chosen so that sqrt(2 * rho_max) ≈ sigma_max = 10 ⇒ rho_max = 50.
  // 800 cells keeps the cell width comparable to the sigma-form default grid.
  GridSettings rho_grid{800, 0.0, 50.0};
  run_flow_to_time<ON_KT_RHO_LSM_Model>(flow_case, target_time, rho_grid, diagnostic_threads);
}

// Same KT ρ-coords model, but on the *fine* mesh that the user-facing
// example Examples/ONfiniteT_KT/parameter.json uses (150 cells over
// [0, 0.015], dx ≈ 1e-4). Diagnostic: isolates whether the example's
// far-earlier failure (~t = 0.008) vs the wide-mesh test's t ≈ 2.14
// is due to mesh resolution rather than something orchestration-related.
TEST_CASE("KT O(N) physical-scale integrator-based smoke test - rho-coords, fine mesh",
          "[FV][KT][ON][smoke][integrator][physical-scale][kt-rho][kt-rho-fine]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", KT rho-coords smoke test on fine mesh, Lambda=" << flow_case.lambda
                       << ", T=" << LSMPhysicalParameters::T);

  constexpr double target_time = 4.0;
  // 150 cells over [0, 0.015]: dx ≈ 1e-4 — matches the user-facing example.
  GridSettings rho_grid{150, 0.0, 0.015};
  run_flow_to_time<ON_KT_RHO_LSM_Model>(flow_case, target_time, rho_grid, diagnostic_threads);
}

// ============================================================================
// Wave-speed-strategy comparison tests
// ----------------------------------------------------------------------------
// All other KT integrator tests above use MaxEigenvalueWaveSpeedZeroDeriv (the
// file-level `Assembler` alias hard-codes it). The tests below sweep all three
// strategies on the same physics so we can quantify the per-strategy
// failure-time difference cleanly. Tags:
//   [ws-zerod] — MaxEigenvalueWaveSpeedZeroDeriv (Real<1>, H=0)
//   [ws-ad]    — MaxEigenvalueWaveSpeed (Real<2> AD for H)
//   [ws-fd]    — MaxEigenvalueWaveSpeedFD (Real<1> + central FD for H)
// ============================================================================

TEST_CASE("KT wavespeed comparison - sigma-coords PolyExp T=0.05, MaxEigenvalueWaveSpeedZeroDeriv",
          "[FV][KT][ON][smoke][integrator][physical-scale][ws-comparison][ws-zerod][ws-sigma]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", wavespeed strategy = ZeroDeriv");
  run_flow_to_time_ws<ONIntegratorKTModel, FV::KurganovTadmor::MaxEigenvalueWaveSpeedZeroDeriv>(
      flow_case, 4.0, default_grid_settings(), diagnostic_threads);
}

TEST_CASE("KT wavespeed comparison - sigma-coords PolyExp T=0.05, MaxEigenvalueWaveSpeed",
          "[FV][KT][ON][smoke][integrator][physical-scale][ws-comparison][ws-ad][ws-sigma]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", wavespeed strategy = MaxEigenvalueWaveSpeed (Real<2> AD H)");
  run_flow_to_time_ws<ONIntegratorKTModel, FV::KurganovTadmor::MaxEigenvalueWaveSpeed>(
      flow_case, 4.0, default_grid_settings(), diagnostic_threads);
}

TEST_CASE("KT wavespeed comparison - sigma-coords PolyExp T=0.05, MaxEigenvalueWaveSpeedFD",
          "[FV][KT][ON][smoke][integrator][physical-scale][ws-comparison][ws-fd][ws-sigma]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", wavespeed strategy = MaxEigenvalueWaveSpeedFD (Real<1> + central-FD H)");
  run_flow_to_time_ws<ONIntegratorKTModel, FV::KurganovTadmor::MaxEigenvalueWaveSpeedFD>(
      flow_case, 4.0, default_grid_settings(), diagnostic_threads);
}

TEST_CASE("KT wavespeed comparison - rho-coords PolyExp T=0.05, MaxEigenvalueWaveSpeedZeroDeriv",
          "[FV][KT][ON][smoke][integrator][physical-scale][ws-comparison][ws-zerod][ws-rho]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", rho-coords, wavespeed strategy = ZeroDeriv");
  GridSettings rho_grid{800, 0.0, 50.0};
  run_flow_to_time_ws<ON_KT_RHO_LSM_Model, FV::KurganovTadmor::MaxEigenvalueWaveSpeedZeroDeriv>(
      flow_case, 4.0, rho_grid, diagnostic_threads);
}

TEST_CASE("KT wavespeed comparison - rho-coords PolyExp T=0.05, MaxEigenvalueWaveSpeed",
          "[FV][KT][ON][smoke][integrator][physical-scale][ws-comparison][ws-ad][ws-rho]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", rho-coords, wavespeed strategy = MaxEigenvalueWaveSpeed (Real<2> AD H)");
  GridSettings rho_grid{800, 0.0, 50.0};
  run_flow_to_time_ws<ON_KT_RHO_LSM_Model, FV::KurganovTadmor::MaxEigenvalueWaveSpeed>(
      flow_case, 4.0, rho_grid, diagnostic_threads);
}

TEST_CASE("KT wavespeed comparison - rho-coords PolyExp T=0.05, MaxEigenvalueWaveSpeedFD",
          "[FV][KT][ON][smoke][integrator][physical-scale][ws-comparison][ws-fd][ws-rho]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", rho-coords, wavespeed strategy = MaxEigenvalueWaveSpeedFD (Real<1> + central-FD H)");
  GridSettings rho_grid{800, 0.0, 50.0};
  run_flow_to_time_ws<ON_KT_RHO_LSM_Model, FV::KurganovTadmor::MaxEigenvalueWaveSpeedFD>(
      flow_case, 4.0, rho_grid, diagnostic_threads);
}

// CG-vs-KT comparison test: same physics as the failing
// [smoke][integrator][physical-scale] case (same V_pion/V_sigma kernels,
// same LSM initial potential, same Lambda, same PolyExp T=0.05 regulator),
// but the KT FV discretisation is replaced with CG order-4. If this test
// passes while the KT counterpart fails, the integrator+AD path is fine and
// the KT-specific Real<2, double> wave-speed Hessian path is the precision
// bottleneck.
TEST_CASE("CG O(N) physical-scale integrator-based comparison test - same physics as KT, PolyExp T=0.05",
          "[CG][ON][smoke][integrator][physical-scale][cg-vs-kt]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  const auto flow_case = LSMPhysical_ON2::flow_case();
  INFO(flow_case.label << ", CG comparison test, Lambda=" << flow_case.lambda
                       << ", T=" << LSMPhysicalParameters::T);

  constexpr double target_time = 4.0;
  run_flow_to_time_cg<ON_CG_LSM_Model>(flow_case, target_time, default_grid_settings(), diagnostic_threads);
}
