#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <DiFfRG/model/fv_boundaries.hh>

namespace
{
  struct DummyModelDefault : public DiFfRG::def::FVDefaultBoundaries<DummyModelDefault>
  {
  };

  struct DummyModelOriginOddLinearExtrapolation
      : public DiFfRG::def::OriginOddLinearExtrapolationBoundaries<DummyModelOriginOddLinearExtrapolation>
  {
  };

  double central_difference_gradient(const std::array<dealii::Point<1>, 3> &x, const std::array<double, 3> &u)
  {
    return (u[2] - u[0]) / (x[2][0] - x[0][0]);
  }
} // namespace

TEST_CASE("FVDefaultBoundaries copies the interior ghost gradient", "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;
  using namespace dealii;

  DummyModelDefault model;

  std::array<Tensor<1, 1, double>, 2> ghost_gradient{};
  std::array<Tensor<1, 1, double>, 2> cell_gradient{};
  std::array<types::boundary_id, 2> boundary_ids{{0, numbers::invalid_boundary_id}};
  std::array<std::array<Tensor<1, 1, double>, 2>, 2> neighboring_gradients{};
  cell_gradient[0][0] = 1.25;
  cell_gradient[1][0] = -0.75;

  model.boundary_ghost_gradient(ghost_gradient, 0, Tensor<1, 1>{}, Point<1>{}, std::array<double, 2>{},
                                Point<1>{}, std::array<double, 2>{}, Point<1>{}, cell_gradient, boundary_ids,
                                neighboring_gradients);

  CHECK_THAT(ghost_gradient[0][0], Catch::Matchers::WithinAbs(1.25, 1e-12));
  CHECK_THAT(ghost_gradient[1][0], Catch::Matchers::WithinAbs(-0.75, 1e-12));
}

TEST_CASE("OriginOddLinearExtrapolationBoundaries uses odd reflection at the origin", "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;
  using namespace dealii;

  DummyModelOriginOddLinearExtrapolation model;

  std::array<std::array<double, 1>, 2> u_neighbors{{{{0.0}}, {{3.0}}}};
  std::array<Point<1>, 2> x_neighbors{{Point<1>(-1.0), Point<1>(1.5)}};
  std::array<types::boundary_id, 2> boundary_ids{{0, numbers::invalid_boundary_id}};
  std::array<Point<1>, 2> face_centers{{Point<1>(0.0), Point<1>()}};
  const std::array<double, 1> u_cell{{2.0}};
  const Point<1> x_cell(0.5);

  model.apply_boundary_conditions(u_neighbors, x_neighbors, boundary_ids, face_centers, u_cell, x_cell);

  CHECK_THAT(x_neighbors[0][0], Catch::Matchers::WithinAbs(-1.5, 1e-12));
  CHECK_THAT(u_neighbors[0][0], Catch::Matchers::WithinAbs(-3.0, 1e-12));
  CHECK_THAT(x_neighbors[1][0], Catch::Matchers::WithinAbs(1.5, 1e-12));
  CHECK_THAT(u_neighbors[1][0], Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("OriginOddLinearExtrapolationBoundaries uses linear extrapolation at sigma_max",
          "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;
  using namespace dealii;

  DummyModelOriginOddLinearExtrapolation model;

  std::array<std::array<double, 1>, 2> u_neighbors{{{{1.0}}, {{0.0}}}};
  std::array<Point<1>, 2> x_neighbors{{Point<1>(0.5), Point<1>(0.0)}};
  std::array<types::boundary_id, 2> boundary_ids{{numbers::invalid_boundary_id, 1}};
  std::array<Point<1>, 2> face_centers{{Point<1>(), Point<1>(2.0)}};
  const std::array<double, 1> u_cell{{2.0}};
  const Point<1> x_cell(1.5);

  model.apply_boundary_conditions(u_neighbors, x_neighbors, boundary_ids, face_centers, u_cell, x_cell);

  CHECK_THAT(x_neighbors[1][0], Catch::Matchers::WithinAbs(2.5, 1e-12));
  CHECK_THAT(u_neighbors[1][0], Catch::Matchers::WithinAbs(3.0, 1e-12));
  CHECK_THAT(x_neighbors[0][0], Catch::Matchers::WithinAbs(0.5, 1e-12));
  CHECK_THAT(u_neighbors[0][0], Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("OriginOddLinearExtrapolationBoundaries selects the opposite interior neighbor gradient",
          "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;
  using namespace dealii;

  DummyModelOriginOddLinearExtrapolation model;

  std::array<Tensor<1, 1, double>, 1> ghost_gradient{};
  std::array<Tensor<1, 1, double>, 1> cell_gradient{};
  std::array<types::boundary_id, 2> boundary_ids{{0, numbers::invalid_boundary_id}};
  std::array<std::array<Tensor<1, 1, double>, 1>, 2> neighboring_gradients{};

  const std::array<Point<1>, 3> lattice_points{{Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
  const std::array<double, 3> lattice_values{{0.0, 1.0, 3.0}};

  cell_gradient[0][0] = 0.0;
  neighboring_gradients[0][0][0] = -2.0;
  neighboring_gradients[1][0][0] = central_difference_gradient(lattice_points, lattice_values);

  model.boundary_ghost_gradient(ghost_gradient, 0, Tensor<1, 1>{}, Point<1>(0.0), std::array<double, 1>{{-0.2}},
                                Point<1>(-0.2), std::array<double, 1>{{0.0}}, Point<1>(0.0), cell_gradient,
                                boundary_ids, neighboring_gradients);

  CHECK_THAT(ghost_gradient[0][0], Catch::Matchers::WithinAbs(1.5, 1e-12));
}
