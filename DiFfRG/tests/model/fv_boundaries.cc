#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <DiFfRG/model/fv_boundaries.hh>

namespace
{
  using namespace dealii;

  constexpr std::size_t left_face = 0;
  constexpr std::size_t right_face = GeometryInfo<1>::opposite_face[left_face];
  constexpr double tolerance = 1e-12;

  template <typename T> using FaceArray = std::array<T, GeometryInfo<1>::faces_per_cell>;

  struct DummyModelDefault : public DiFfRG::def::FVDefaultBoundaries<DummyModelDefault>
  {
  };

  struct DummyModelOriginOddLinearExtrapolation
      : public DiFfRG::def::OriginOddLinearExtrapolationBoundaries<DummyModelOriginOddLinearExtrapolation>
  {
  };

  FaceArray<types::boundary_id> left_boundary_ids()
  {
    return FaceArray<types::boundary_id>{{0, numbers::invalid_boundary_id}};
  }

  FaceArray<types::boundary_id> right_boundary_ids()
  {
    return FaceArray<types::boundary_id>{{numbers::invalid_boundary_id, 1}};
  }

  void check_face_state(const FaceArray<Point<1>> &x_neighbors, const FaceArray<std::array<double, 1>> &u_neighbors,
                        const std::size_t face, const double expected_x, const double expected_u)
  {
    CHECK_THAT(x_neighbors[face][0], Catch::Matchers::WithinAbs(expected_x, tolerance));
    CHECK_THAT(u_neighbors[face][0], Catch::Matchers::WithinAbs(expected_u, tolerance));
  }

  double central_difference_gradient(const std::array<Point<1>, 3> &x, const std::array<double, 3> &u)
  {
    return (u[2] - u[0]) / (x[2][0] - x[0][0]);
  }
} // namespace

TEST_CASE("FVDefaultBoundaries copies the interior ghost gradient", "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelDefault model;

  std::array<Tensor<1, 1, double>, 2> ghost_gradient{};
  std::array<Tensor<1, 1, double>, 2> cell_gradient{};
  const auto boundary_ids = left_boundary_ids();
  std::array<std::array<Tensor<1, 1, double>, 2>, 2> neighboring_gradients{};
  cell_gradient[0][0] = 1.25;
  cell_gradient[1][0] = -0.75;

  model.boundary_ghost_gradient(ghost_gradient, 0, Tensor<1, 1>{}, Point<1>{}, std::array<double, 2>{},
                                Point<1>{}, std::array<double, 2>{}, Point<1>{}, cell_gradient, boundary_ids,
                                neighboring_gradients);

  CHECK_THAT(ghost_gradient[0][0], Catch::Matchers::WithinAbs(1.25, tolerance));
  CHECK_THAT(ghost_gradient[1][0], Catch::Matchers::WithinAbs(-0.75, tolerance));
}

TEST_CASE("OriginOddLinearExtrapolationBoundaries applies the matching ghost-state rule", "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelOriginOddLinearExtrapolation model;

  SECTION("origin boundary reflects the interior state with odd parity")
  {
    FaceArray<std::array<double, 1>> u_neighbors{{{{0.0}}, {{3.0}}}};
    FaceArray<Point<1>> x_neighbors{{Point<1>(-1.0), Point<1>(1.5)}};
    const auto boundary_ids = left_boundary_ids();
    FaceArray<Point<1>> face_centers{{Point<1>(0.0), Point<1>()}};
    const std::array<double, 1> u_cell{{2.0}};
    const Point<1> x_cell(0.5);

    model.apply_boundary_conditions(u_neighbors, x_neighbors, boundary_ids, face_centers, u_cell, x_cell);

    check_face_state(x_neighbors, u_neighbors, left_face, -1.5, -3.0);
    check_face_state(x_neighbors, u_neighbors, right_face, 1.5, 3.0);
  }

  SECTION("outer boundary extrapolates from the cell state")
  {
    FaceArray<std::array<double, 1>> u_neighbors{{{{1.0}}, {{0.0}}}};
    FaceArray<Point<1>> x_neighbors{{Point<1>(0.5), Point<1>(0.0)}};
    const auto boundary_ids = right_boundary_ids();
    FaceArray<Point<1>> face_centers{{Point<1>(), Point<1>(2.0)}};
    const std::array<double, 1> u_cell{{2.0}};
    const Point<1> x_cell(1.5);

    model.apply_boundary_conditions(u_neighbors, x_neighbors, boundary_ids, face_centers, u_cell, x_cell);

    check_face_state(x_neighbors, u_neighbors, left_face, 0.5, 1.0);
    check_face_state(x_neighbors, u_neighbors, right_face, 2.5, 3.0);
  }
}

TEST_CASE("OriginOddLinearExtrapolationBoundaries reuses the interior neighbor gradient",
          "[Model][FV][Boundaries]")
{
  using namespace DiFfRG;

  DummyModelOriginOddLinearExtrapolation model;

  std::array<Tensor<1, 1, double>, 1> ghost_gradient{};
  std::array<Tensor<1, 1, double>, 1> cell_gradient{};
  const auto boundary_ids = left_boundary_ids();
  std::array<std::array<Tensor<1, 1, double>, 1>, 2> neighboring_gradients{};

  const std::array<Point<1>, 3> lattice_points{{Point<1>(0.5), Point<1>(1.5), Point<1>(2.5)}};
  const std::array<double, 3> lattice_values{{0.0, 1.0, 3.0}};

  cell_gradient[0][0] = 0.0;
  neighboring_gradients[0][0][0] = -2.0;
  neighboring_gradients[1][0][0] = central_difference_gradient(lattice_points, lattice_values);

  model.boundary_ghost_gradient(ghost_gradient, 0, Tensor<1, 1>{}, Point<1>(0.0), std::array<double, 1>{{-0.2}},
                                Point<1>(-0.2), std::array<double, 1>{{0.0}}, Point<1>(0.0), cell_gradient,
                                boundary_ids, neighboring_gradients);

  CHECK_THAT(ghost_gradient[0][0], Catch::Matchers::WithinAbs(1.5, tolerance));
}
