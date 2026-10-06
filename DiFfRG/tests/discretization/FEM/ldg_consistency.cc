#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/model/model.hh>

#include <spdlog/sinks/stdout_color_sinks.h>

#include <cmath>

using namespace dealii;
using namespace DiFfRG;

namespace
{
  template <bool extractors>
  using LDGComponents =
      ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>, VariableDescriptor<>,
                          std::conditional_t<extractors, ExtractorDescriptor<Scalar<"e">>, ExtractorDescriptor<>>,
                          FEFunctionDescriptor<Scalar<"lu">, Scalar<"lv">>>;

  template <typename Model>
  using UpDownFluxes =
      def::LDGUpDownFluxes<Model, def::UpDownFlux<def::FlowDirections<0, 0>, def::UpDown<def::from_right, def::from_left>>>;

  /**
   * Two FE functions u, v and one LDG level (lu, lv) = (u', v'). The main flux and source depend nonlinearly on
   * both levels and (optionally) on an extractor, which also reaches the LLF numerical flux. With `constant`, the
   * model declares the level jacobian constant, so the assembler builds the level from the solution by a
   * precomputed matrix instead of assembling it.
   */
  template <bool constant, bool extractors>
  class ModelLDG : public def::AbstractModel<ModelLDG<constant, extractors>, LDGComponents<extractors>>,
                   public def::Time,
                   public def::LLFFlux<ModelLDG<constant, extractors>>,
                   public UpDownFluxes<ModelLDG<constant, extractors>>,
                   public def::FlowBoundaries<ModelLDG<constant, extractors>>,
                   public def::AD<ModelLDG<constant, extractors>>
  {
  public:
    static constexpr uint dim = 1;
    using Components = LDGComponents<extractors>;

    ModelLDG()
    {
      this->components().add_dependency(1, 0, 0, 0);
      this->components().add_dependency(1, 1, 0, 1);
      if constexpr (constant) this->components().set_jacobian_constant(1, 0);
    }

    template <typename Vector> void initial_condition(const Point<dim> &x, Vector &values) const
    {
      values[0] = 1. + 0.3 * std::sin(2. * x[0]) + 0.2 * x[0] * x[0];
      values[1] = 0.5 + 0.4 * std::cos(3. * x[0]);
    }

    template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector &) const
    {
      return {{x[0] - 0.37}};
    }

    template <typename NT, typename Solution>
    void extract(std::array<NT, Components::count_extractors()> &e, const Point<dim> &, const Solution &sol) const
    {
      if constexpr (extractors) e[0] = get<"fe_functions">(sol)[0] * get<"fe_functions">(sol)[1];
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, 2> &F, const Point<dim> &x, const Solution &sol) const
    {
      const auto &u = get<"fe_functions">(sol);
      const auto &l = get<"LDG1">(sol);
      const auto e = extractor(sol);
      F[0][0] = u[0] * u[1] + 0.3 * l[0] * u[0] + 0.1 * e * u[0] * u[0] + 0.05 * x[0];
      F[1][0] = 0.5 * u[1] * u[1] + 0.2 * l[1] * u[0];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, 2> &S, const Point<dim> &, const Solution &sol) const
    {
      using std::exp;
      const auto &u = get<"fe_functions">(sol);
      const auto &l = get<"LDG1">(sol);
      const auto e = extractor(sol);
      S[0] = 0.1 * u[0] * u[0] * u[0] - e * u[1] + l[0] * l[1];
      S[1] = exp(0.1 * u[0]) * u[1];
    }

    template <uint submodel, typename NT, typename Vector>
    void ldg_flux(std::array<Tensor<1, dim, NT>, 2> &F, const Point<dim> &, const Vector &u) const
    {
      F[0][0] = u[0];
      F[1][0] = u[1];
    }

  private:
    template <typename Solution> static auto extractor(const Solution &sol)
    {
      if constexpr (extractors)
        return get<"extractors">(sol)[0];
      else
        return 0.;
    }
  };

  ConfigTree make_config(const int fe_order)
  {
    return ConfigTree(json::value({{"physical", {}},
                                   {"discretization",
                                    {{"fe_order", fe_order},
                                     {"overintegration", 0},
                                     {"output_subdivisions", 1},
                                     {"EoM_abs_tol", 1e-10},
                                     {"EoM_max_iter", 100},
                                     {"grid", {{"x_grid", "0:0.1:1"}, {"refine", 0}}},
                                     {"adaptivity",
                                      {{"start_adapt_at", 0.},
                                       {"adapt_dt", 1e-1},
                                       {"level", 0},
                                       {"refine_percent", 1e-1},
                                       {"coarsen_percent", 5e-2}}}}},
                                   {"output", {{"live_plot", false}, {"verbosity", 0}}}}));
  }

  void ensure_logger()
  {
    try {
      spdlog::stdout_color_mt("log");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  constexpr double weight = 1.3, weight_mass = 0.7;

  template <bool constant, bool extractors> struct Setup {
    using Model = ModelLDG<constant, extractors>;
    using Discretization = LDG::Discretization<Model, RectangularMeshSerial<1>>;
    using VectorType = typename Discretization::VectorType;
    using SparseMatrixType = typename Discretization::SparseMatrixType;

    Setup(const int fe_order)
        : config(make_config(fe_order)), mesh((ensure_logger(), Config::ConfigurationMesh<1>(config))),
          discretization(mesh, config), assembler(discretization, model, config), state(discretization)
    {
      state.interpolate(model);
      u = state.spatial_data();
      u_dot = u;
      for (uint i = 0; i < u_dot.size(); ++i)
        u_dot[i] = std::sin(1. + i);
    }

    VectorType residual(const VectorType &at)
    {
      VectorType r(at);
      r = 0;
      assembler.residual(r, at, weight, u_dot, weight_mass);
      return r;
    }

    /// d(residual)/du: the mass term enters through beta = weight_mass, u_dot is held fixed.
    SparseMatrixType &jacobian()
    {
      J.reinit(assembler.get_sparsity_pattern_jacobian());
      assembler.jacobian(J, u, weight, u_dot, 0., weight_mass);
      return J;
    }

    ConfigTree config;
    Model model;
    RectangularMeshSerial<1> mesh;
    Discretization discretization;
    LDG::Assembler<Discretization> assembler;
    FE::FlowingVariables<Discretization> state;
    VectorType u, u_dot;
    SparseMatrixType J;
  };

  template <typename V> double max_abs(const V &v)
  {
    double m = 0.;
    for (uint i = 0; i < v.size(); ++i)
      m = std::max(m, std::abs(v[i]));
    return m;
  }

  /// Central finite differences of the residual, column by column, against the assembled jacobian.
  template <typename S> void require_jacobian_matches_finite_differences(S &s)
  {
    const auto &J = s.jacobian();
    double worst = 0., scale = 0.;
    for (uint j = 0; j < s.u.size(); ++j) {
      const double h = 1e-6 * std::max(1., std::abs(s.u[j]));
      auto plus = s.u, minus = s.u;
      plus[j] += h;
      minus[j] -= h;
      const auto r_plus = s.residual(plus), r_minus = s.residual(minus);
      for (uint i = 0; i < s.u.size(); ++i) {
        const double fd = (r_plus[i] - r_minus[i]) / (2 * h);
        scale = std::max(scale, std::abs(fd));
        worst = std::max(worst, std::abs(fd - J.el(i, j)));
      }
    }
    REQUIRE(scale > 0.);
    INFO("jacobian vs finite differences: worst " << worst << ", scale " << scale);
    REQUIRE(worst <= 1e-6 * scale);
  }
} // namespace

// With a constant level jacobian the level is built as J_gu * u; every FE component has to reach it, not just
// the first. The assembled level (non-constant path) is the reference.
TEST_CASE("LDG level from a constant jacobian matches the assembled level", "[discretization][ldg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2);
  Setup<true, false> constant(fe_order);
  Setup<false, false> assembled(fe_order);
  const auto r_c = constant.residual(constant.u), r_a = assembled.residual(assembled.u);
  auto deviation = r_c;
  deviation -= r_a;
  REQUIRE(max_abs(r_a) > 0.);
  INFO("worst deviation " << max_abs(deviation) << ", scale " << max_abs(r_a));
  REQUIRE(max_abs(deviation) <= 1e-11 * max_abs(r_a));
}

TEST_CASE("LDG jacobian matches finite differences of the residual", "[discretization][ldg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2);
  SECTION("constant level jacobian")
  {
    Setup<true, false> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("assembled level jacobian")
  {
    Setup<false, false> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
  // The extractor enters the volume terms and, through the LLF flux, the face terms; weight != 1.
  SECTION("constant level jacobian, extractor")
  {
    Setup<true, true> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("assembled level jacobian, extractor")
  {
    Setup<false, true> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
}
