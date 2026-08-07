#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/combined_coordinates.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/discretization/coordinates/stack_coordinates.hh>

using namespace DiFfRG;

TEST_CASE("Test 1D Logarithmic coordinates", "[1D][coordinates]")
{
  const float p_start = GENERATE(take(3, random(1e-6, 1e-1)));
  const float p_stop = GENERATE(take(3, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(3, random(10, 100)));
  const float p_bias = GENERATE(take(3, random(1., 10.)));

  LogarithmicCoordinates1D<float> coords(p_size, p_start, p_stop, p_bias);

  const auto grid = make_grid(coords);
  CHECK(is_close(grid[0][0], p_start, 1e-8f));
  for (uint i = 1; i < grid.size() - 1; ++i)
    CHECK(grid[i][0] > grid[i - 1][0]);
  CHECK(is_close(grid[grid.size() - 1][0], p_stop, 1e-5));
}

TEST_CASE("Test 1D Linear coordinates", "[1D][coordinates]")
{
  const float p_start = GENERATE(take(3, random(1e-6, 1e-1)));
  const float p_stop = GENERATE(take(3, random(1., 100.))) + p_start;
  const int p_size = GENERATE(take(3, random(10, 100)));

  LinearCoordinates1D<float> coords(p_size, p_start, p_stop);

  const auto grid = make_grid(coords);
  CHECK(is_close(grid[0][0], p_start, 1e-8f));
  for (uint i = 1; i < grid.size() - 1; ++i)
    CHECK(grid[i][0] > grid[i - 1][0]);
  CHECK(is_close(grid[grid.size() - 1][0], p_stop, 1e-5f));
}

TEST_CASE("Test 1D Linear periodic coordinates", "[1D][coordinates][periodic]")
{
  const double p_start = GENERATE(take(3, random(-10., 10.)));
  const double p_period = GENERATE(take(3, random(1., 100.)));
  const int p_size = GENERATE(take(3, random(2, 100)));
  const double p_stop = p_start + p_period;
  const double dx = p_period / p_size;

  LinearPeriodicCoordinates1D<double> coords(p_size, p_start, p_stop);

  const auto grid = make_grid(coords);
  REQUIRE(grid.size() == size_t(p_size));

  // The endpoint is excluded: the grid runs start, start + dx, ..., stop - dx
  CHECK(is_close(grid[0][0], p_start, 1e-10 * p_period));
  for (uint i = 1; i < grid.size(); ++i)
    CHECK(grid[i][0] > grid[i - 1][0]);
  CHECK(is_close(grid[grid.size() - 1][0], p_stop - dx, 1e-10 * p_period));

  // backward is the inverse of forward on the grid points
  for (int i = 0; i < p_size; ++i)
    CHECK(is_close(coords.backward(coords.forward(i)), double(i), 1e-8));

  // and it folds everything else back into [0, p_size). Note that right at the seam the fold may land on either
  // side of it, so compare indices as points on a circle of circumference p_size.
  const auto circular_distance = [&](const double a, const double b) {
    const double d = std::abs(a - b);
    return std::min(d, double(p_size) - d);
  };
  CHECK(circular_distance(coords.backward(p_stop), 0.) < 1e-8);
  CHECK(circular_distance(coords.backward(p_start - dx), double(p_size - 1)) < 1e-8);
  CHECK(circular_distance(coords.backward(p_start + 3 * p_period + 2 * dx), 2.) < 1e-8);
  for (const double y : {p_start - 7 * p_period, p_start + 11 * p_period, p_stop - 1e-12 * p_period}) {
    const double idx = coords.backward(y);
    CHECK(idx >= 0.);
    CHECK(idx < double(p_size));
  }
}

TEST_CASE("Test 2D CoordinatePackND with two LogarithmicCoordinates1D", "[2D][coordinates]")
{
  const float p_start_1 = GENERATE(take(2, random(1e-6, 1e-1)));
  const float p_stop_1 = GENERATE(take(2, random(1., 100.))) + p_start_1;
  const int p_size_1 = GENERATE(take(2, random(10, 100)));
  const float p_bias_1 = GENERATE(take(2, random(1., 10.)));
  LogarithmicCoordinates1D<float> coords1(p_size_1, p_start_1, p_stop_1, p_bias_1);

  const float p_start_2 = GENERATE(take(2, random(1e-6, 1e-1)));
  const float p_stop_2 = GENERATE(take(2, random(1., 100.))) + p_start_2;
  const int p_size_2 = GENERATE(take(2, random(10, 100)));
  const float p_bias_2 = GENERATE(take(2, random(1., 10.)));
  LogarithmicCoordinates1D<float> coords2(p_size_2, p_start_2, p_stop_2, p_bias_2);

  CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>> coords(coords1, coords2);

  const auto grid = make_grid(coords);

  // Check the boundaries
  for (uint i = 0; i < coords2.size(); ++i) {
    // lower boundary
    if (!is_close(grid[i][1], coords2.forward(i), 1e-5))
      std::cout << "grid[i][1]: " << grid[i][1] << " coords2.forward(i): " << coords2.forward(i) << std::endl;
    CHECK(is_close(grid[i][1], coords2.forward(i), 1e-5));

    if (!is_close(grid[i][0], coords1.forward(0), 1e-5))
      std::cout << "grid[i][0]: " << grid[i][0] << " coords1.forward(0): " << coords1.forward(0) << std::endl;
    CHECK(is_close(grid[i][0], coords1.forward(0), 1e-5));

    if (!is_close(p_start_1, coords1.forward(0), 1e-5))
      std::cout << "p_start_1: " << p_start_1 << " coords1.forward(0): " << coords1.forward(0) << std::endl;
    CHECK(is_close(p_start_1, coords1.forward(0), 1e-5));

    // upper boundary
    if (!is_close(grid[(coords1.size() - 1) * coords2.size() + i][1], coords2.forward(i), 1e-5))
      std::cout << "grid[(coords1.size() - 1) * coords2.size() + i][1]: "
                << grid[(coords1.size() - 1) * coords2.size() + i][1] << " coords2.forward(i): " << coords2.forward(i)
                << std::endl;
    CHECK(is_close(grid[(coords1.size() - 1) * coords2.size() + i][1], coords2.forward(i), 1e-5));

    if (!is_close(grid[(coords1.size() - 1) * coords2.size() + i][0], coords1.forward(coords1.size() - 1), 1e-5))
      std::cout << "grid[(coords1.size() - 1) * coords2.size() + i][0]: "
                << grid[(coords1.size() - 1) * coords2.size() + i][0]
                << " coords1.forward(coords1.size() - 1): " << coords1.forward(coords1.size() - 1) << std::endl;
    CHECK(is_close(grid[(coords1.size() - 1) * coords2.size() + i][0], coords1.forward(coords1.size() - 1), 1e-5));

    if (!is_close(p_stop_1, coords1.forward(p_size_1 - 1), 1e-5))
      std::cout << "p_stop_1: " << p_stop_1 << " coords1.forward(p_size_1 - 1): " << coords1.forward(p_size_1 - 1)
                << std::endl;
    CHECK(is_close(p_stop_1, coords1.forward(p_size_1 - 1), 1e-5));
  }
}