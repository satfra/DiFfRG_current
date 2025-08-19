#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/tbb.hh>

using namespace DiFfRG;

// Test TBBReduction up to 9 dimensions with random values
TEMPLATE_TEST_CASE_SIG("Test TBB reducer", "[integration][tbb]", ((int dim), dim), (1), (2), (3), (4), (5), (6), (7),
                       (8), (9))
{
  const auto val = GENERATE(take(10, random(-1000., 1000.)));
  auto functor = [val](const auto &) { return val; };
  device::array<size_t, dim> grid_size;
  for (size_t i = 0; i < dim; ++i) {
    grid_size[i] = GENERATE(take(1, random(1, 10)));
  }
  double res = TBBReduction<dim, double, decltype(functor)>(grid_size, functor);
  double expected = val;
  for (size_t i = 0; i < dim; ++i) {
    expected *= grid_size[i];
  }
  CHECK(is_close(res, expected, 1e-6));
}