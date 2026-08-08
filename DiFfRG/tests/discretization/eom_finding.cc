#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>

#include <boilerplate/models.hh>

#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>

namespace
{
  using namespace dealii;
  using namespace DiFfRG;

  ConfigTree make_json(const int fe_order)
  {
    return json::value(
        {{"physical", {}},
         {"integration",
          {{"x_quadrature_order", 32},
           {"angle_quadrature_order", 8},
           {"x0_quadrature_order", 16},
           {"x0_summands", 8},
           {"q0_quadrature_order", 16},
           {"q0_summands", 8},
           {"x_extent_tolerance", 1e-3},
           {"x0_extent_tolerance", 1e-3},
           {"q0_extent_tolerance", 1e-3},
           {"jacobian_quadrature_factor", 0.5}}},
         {"discretization",
          {{"fe_order", fe_order},
           {"mesh_workers", 1},
           {"batch_size", 16},
           {"overintegration", 0},
           {"output_subdivisions", 2},
           {"EoM_abs_tol", 1e-12},
           {"EoM_max_iter", 200},
           {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 5e-2}}}}},
         {"timestepping",
          {{"final_time", 1.},
           {"output_dt", 1e-1},
           {"explicit",
            {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}},
           {"implicit",
            {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}}}},
         {"output", {{"live_plot", false}, {"verbosity", 0}}}});
  }

  void setup_logger()
  {
    try {
      auto log = spdlog::stdout_color_mt("log");
      log->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
  }

  template <int dim>
  Testing::PhysicalParameters parameters_from_affine(const std::array<double, dim> &x0,
                                                     const std::array<double, dim> &x1)
  {
    Testing::PhysicalParameters prm;
    for (uint d = 0; d < dim; ++d) {
      prm.initial_x0[d] = x0[d];
      prm.initial_x1[d] = x1[d];
    }
    return prm;
  }

  template <int dim> Point<dim> affine_minimum(const Testing::PhysicalParameters &prm)
  {
    Point<dim> minimum;
    for (uint d = 0; d < dim; ++d)
      minimum[d] = prm.initial_x0[d] > 0. ? 0. : -prm.initial_x0[d] / prm.initial_x1[d];
    return minimum;
  }

  template <int dim> Testing::PhysicalParameters parameters_with_minimum(const Point<dim> &minimum)
  {
    Testing::PhysicalParameters prm;
    for (uint d = 0; d < dim; ++d) {
      prm.initial_x1[d] = 2.5 + d;
      prm.initial_x0[d] = -prm.initial_x1[d] * minimum[d];
    }
    return prm;
  }

  Testing::PhysicalParameters parameters_with_boundary_minimum()
  {
    Testing::PhysicalParameters prm;
    for (uint d = 0; d < 3; ++d) {
      prm.initial_x0[d] = 1.;
      prm.initial_x1[d] = 1.;
    }
    return prm;
  }

  template <int dim> double distance(const Point<dim> &a, const Point<dim> &b)
  {
    double out = 0.;
    for (uint d = 0; d < dim; ++d)
      out = std::max(out, std::abs(a[d] - b[d]));
    return out;
  }

  template <int dim>
  class ModelStitchedPolynomial
      : public def::AbstractModel<ModelStitchedPolynomial<dim>, typename Testing::compFactory<dim>::value>,
        public def::Time,
        public def::NoNumFlux<ModelStitchedPolynomial<dim>>,
        public def::FlowBoundaries<ModelStitchedPolynomial<dim>>,
        public def::AD<ModelStitchedPolynomial<dim>>
  {
  public:
    explicit ModelStitchedPolynomial(const Point<dim> &roots) : roots(roots) {}

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      for (uint d = 0; d < dim; ++d)
        values[d] = value(d, pos[d]);
    }

    template <typename Vector> std::array<double, dim> EoM(const Point<dim> &, const Vector &values) const
    {
      std::array<double, dim> out{{}};
      for (uint d = 0; d < dim; ++d)
        out[d] = values[d];
      return out;
    }

  private:
    static constexpr std::array<double, 3> left_linear{{1.25, 1.75, 2.25}};
    static constexpr std::array<double, 3> right_linear{{5.0, 4.0, 3.5}};
    static constexpr std::array<double, 3> left_cubic{{2.0, 2.5, 3.0}};
    static constexpr std::array<double, 3> right_cubic{{10.0, 8.0, 6.0}};

    double value(const uint d, const double x) const
    {
      const double s = x - roots[d];
      const double linear = s < 0. ? left_linear[d] : right_linear[d];
      const double cubic = s < 0. ? left_cubic[d] : right_cubic[d];
      return linear * s + cubic * s * s * s;
    }

    const Point<dim> roots;
  };

  template <int dim, template <typename, typename, typename> typename DiscretizationTemplate, typename Model>
  auto reconstruct_with_potential_with_model(const Model &model, const int fe_order)
  {
    using NumberType = double;
    using Discretization = DiscretizationTemplate<typename Model::Components, NumberType, RectangularMesh<dim>>;
    using VectorType = typename Discretization::VectorType;

    setup_logger();

    auto json = make_json(fe_order);
    RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
    Discretization discretization(mesh, json, DiFfRG::LogPort{});

    FE::FlowingVariables initial_condition(discretization);
    initial_condition.interpolate(model);
    const VectorType &src = initial_condition.spatial_data();

    const auto &dof_handler = discretization.get_dof_handler();
    const auto &mapping = discretization.get_mapping();

    auto EoM_cell = dof_handler.begin_active();
    return get_EoM_point_with_potential(
        EoM_cell, src, dof_handler, mapping, [&](const auto &p, const auto &values) { return model.EoM(p, values); },
        [&](const auto &p, const auto &) { return p; }, json.get_double("/discretization/EoM_abs_tol"),
        json.get_uint("/discretization/EoM_max_iter"));
  }

  template <int dim, template <typename, typename, typename> typename DiscretizationTemplate, typename Model>
  Point<dim> reconstruct_minimum_with_model(const Model &model, const int fe_order)
  {
    return reconstruct_with_potential_with_model<dim, DiscretizationTemplate>(model, fe_order).point;
  }

  template <int dim, template <typename, typename, typename> typename DiscretizationTemplate>
  Point<dim> reconstruct_minimum(const Testing::PhysicalParameters &prm, const int fe_order)
  {
    using Model = Testing::ModelConstant<dim, dim>;
    Model model(prm);
    return reconstruct_minimum_with_model<dim, DiscretizationTemplate>(model, fe_order);
  }
} // namespace

TEST_CASE("CG EoM potential reconstruction finds 1D affine minima", "[discretization][EoM][cg][1d]")
{
  constexpr uint dim = 1;
  const int fe_order = GENERATE(1, 2, 3, 4, 5);
  const double slope = GENERATE(1.25, 2., 2.5, 5., 10.);
  const auto prm = parameters_from_affine<dim>(std::array<double, dim>{{-1.}}, std::array<double, dim>{{slope}});
  const auto expected = affine_minimum<dim>(prm);
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 6e-2);
}

TEST_CASE("CG EoM potential reconstruction finds 2D affine and boundary minima", "[discretization][EoM][cg][2d]")
{
  constexpr uint dim = 2;
  const int fe_order = GENERATE(1, 2, 3, 4, 5);
  const uint case_index = GENERATE(0u, 1u, 2u, 3u);
  const std::array<std::array<double, 4>, 4> cases{{
      {{-1., 2.5, -1., 5.}},
      {{1., 2., -1., 2.5}},
      {{-1., 5., 1., 4.}},
      {{1., 3., 1., 4.}},
  }};
  const auto &c = cases[case_index];
  const auto prm =
      parameters_from_affine<dim>(std::array<double, dim>{{c[0], c[2]}}, std::array<double, dim>{{c[1], c[3]}});
  const auto expected = affine_minimum<dim>(prm);
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 8e-2);
}

TEST_CASE("CG EoM potential reconstruction finds 3D affine minima", "[discretization][EoM][cg][3d]")
{
  constexpr uint dim = 3;
  const int fe_order = GENERATE(1, 2);
  const uint case_index = GENERATE(0u, 1u);
  const std::array<std::array<double, 6>, 2> cases{{
      {{-1., 2.5, -1., 2., -1., 1.6666666666666667}},
      {{-1., 5., -1., 2.5, -1., 1.25}},
  }};
  const auto &c = cases[case_index];
  const auto prm = parameters_from_affine<dim>(std::array<double, dim>{{c[0], c[2], c[4]}},
                                               std::array<double, dim>{{c[1], c[3], c[5]}});
  const auto expected = affine_minimum<dim>(prm);
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(prm, fe_order);

  REQUIRE(distance(EoM, expected) < 1.1e-1);
}

TEST_CASE("Detailed EoM reconstruction owns the scalar potential and its gauge",
          "[discretization][EoM][potential]")
{
  setup_logger();

  const auto check_result = []<int dim>() {
    const Point<dim> expected = []() {
      if constexpr (dim == 1)
        return Point<dim>(0.4);
      else if constexpr (dim == 2)
        return Point<dim>(0.4, 0.6);
      else
        return Point<dim>(0.4, 0.6, 0.8);
    }();

    using Model = Testing::ModelConstant<dim, dim>;
    using Discretization = CG::Discretization<typename Model::Components, double, RectangularMesh<dim>>;

    auto json = make_json(2);
    Model model(parameters_with_minimum(expected));
    RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
    Discretization discretization(mesh, json, DiFfRG::LogPort{});
    FE::FlowingVariables initial_condition(discretization);
    initial_condition.interpolate(model);

    const auto &dof_handler = discretization.get_dof_handler();
    auto EoM_cell = dof_handler.begin_active();
    const auto result = get_EoM_point_with_potential(
        EoM_cell, initial_condition.spatial_data(), dof_handler, discretization.get_mapping(),
        [&](const auto &p, const auto &values) { return model.EoM(p, values); },
        [&](const auto &p, const auto &) { return p; }, 1e-12, 100);

    REQUIRE(distance(result.point, expected) < 1.1e-1);
    REQUIRE(result.potential.has_value());
    const auto &potential = result.potential.value();
    REQUIRE(potential.finite_element != nullptr);
    REQUIRE(potential.dof_handler != nullptr);
    REQUIRE(potential.values.size() == potential.dof_handler->n_dofs());
    REQUIRE(potential.finite_element->n_components() == 1);
    REQUIRE(distance(potential.minimum, result.point) < 1e-12);

    const auto support_points =
        DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), *potential.dof_handler);
    auto gauge_dof = support_points.begin()->first;
    double gauge_distance = support_points.begin()->second.distance(Point<dim>());
    for (const auto &[dof, point] : support_points) {
      const double current_distance = point.distance(Point<dim>());
      if (current_distance < gauge_distance) {
        gauge_distance = current_distance;
        gauge_dof = dof;
      }
    }
    CHECK(potential.values[gauge_dof] == Catch::Approx(0.).margin(1e-12));
  };

  check_result.template operator()<1>();
  check_result.template operator()<2>();
  check_result.template operator()<3>();
}

TEST_CASE("CG EoM potential reconstruction handles a 1D stitched polynomial minimum",
          "[discretization][EoM][cg][stitched][1d]")
{
  constexpr uint dim = 1;
  const int fe_order = GENERATE(1, 2, 3, 4);
  const Point<dim> expected(0.4);
  const ModelStitchedPolynomial<dim> model(expected);
  const auto EoM = reconstruct_minimum_with_model<dim, CG::Discretization>(model, fe_order);

  REQUIRE(distance(EoM, expected) < 6e-2);
}

TEST_CASE("CG EoM potential reconstruction handles a 2D stitched polynomial minimum",
          "[discretization][EoM][cg][stitched][2d]")
{
  constexpr uint dim = 2;
  const int fe_order = GENERATE(1, 2, 3);
  const Point<dim> expected(0.4, 0.6);
  const ModelStitchedPolynomial<dim> model(expected);
  const auto EoM = reconstruct_minimum_with_model<dim, CG::Discretization>(model, fe_order);

  REQUIRE(distance(EoM, expected) < 8e-2);
}

TEST_CASE("DG EoM potential reconstruction supports DG0", "[discretization][EoM][dg][p0]")
{
  constexpr uint dim = 1;
  const Point<dim> expected(0.4);
  const auto EoM = reconstruct_minimum<dim, DG::Discretization>(parameters_with_minimum(expected), 0);

  REQUIRE(distance(EoM, expected) < 1.6e-1);
}

TEST_CASE("DG EoM potential reconstruction supports higher-order DG", "[discretization][EoM][dg]")
{
  constexpr uint dim = 2;
  const int fe_order = GENERATE(1, 2);
  const Point<dim> expected(0.4, 0.6);
  const auto EoM = reconstruct_minimum<dim, DG::Discretization>(parameters_with_minimum(expected), fe_order);

  REQUIRE(distance(EoM, expected) < 1.1e-1);
}

TEST_CASE("EoM potential reconstruction can return a boundary minimum", "[discretization][EoM][boundary]")
{
  constexpr uint dim = 2;
  const Point<dim> expected(0., 0.);
  const auto EoM = reconstruct_minimum<dim, CG::Discretization>(parameters_with_boundary_minimum(), 2);

  REQUIRE(distance(EoM, expected) < 1e-12);
}

TEST_CASE("EoM max-iteration zero keeps the origin bypass", "[discretization][EoM][origin]")
{
  constexpr uint dim = 2;
  using Model = Testing::ModelConstant<dim, dim>;
  using Discretization = CG::Discretization<typename Model::Components, double, RectangularMesh<dim>>;

  setup_logger();

  auto json = make_json(1);
  Model model(parameters_with_boundary_minimum());
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json, DiFfRG::LogPort{});
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  auto EoM_cell = discretization.get_dof_handler().begin_active();
  const auto result = get_EoM_point_with_potential(
      EoM_cell, initial_condition.spatial_data(), discretization.get_dof_handler(), discretization.get_mapping(),
      [&](const auto &p, const auto &values) { return model.EoM(p, values); },
      [&](const auto &p, const auto &) { return p; }, 1e-12, 0);

  REQUIRE(distance(result.point, Point<dim>(0., 0.)) < 1e-12);
  REQUIRE_FALSE(result.potential.has_value());
}
