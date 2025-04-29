#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/initialize.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation_kokkos/linear_interpolator.hh>

using namespace DiFfRG;

TEMPLATE_TEST_CASE_SIG("Test 2D interpolation", "[float][double][complex][autodiff][interpolation][2D]",
                       ((typename T), T), float, double, complex<double>, complex<float>, autodiff::real)
{
  DiFfRG::Initialize();

  using ctype = typename get_type::ctype<T>;

  using Coordinates1D = LinearCoordinates1D<ctype>;
  using Coordinates2D = CoordinatePackND<Coordinates1D, Coordinates1D>;

  const ctype p1_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p1_stop = GENERATE(take(2, random(1, 100))) + p1_start;
  const int p1_size = GENERATE(take(2, random(10, 100)));

  const ctype p2_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p2_stop = GENERATE(take(2, random(1, 100))) + p1_start;
  const int p2_size = GENERATE(take(2, random(10, 100)));

  std::vector<T> empty_data(p1_size * p2_size, 0.);
  std::vector<T> in_data(p1_size * p2_size, 0.);
  for (int i = 0; i < p1_size; ++i)
    for (int j = 0; j < p2_size; ++j)
      in_data[i * p2_size + j] = j;

  Coordinates2D coords(Coordinates1D(p1_size, p1_start, p1_stop), Coordinates1D(p2_size, p2_start, p2_stop));
  LinearInterpolatorND<CPU_memory, T, Coordinates2D> interpolator(empty_data.data(), coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(2, random(2, 200)));
  const ctype p1_pt = (p1_start + GENERATE(take(3, random(0., 1.))) * (p1_stop - p1_start));
  const ctype p2_pt = (p2_start + GENERATE(take(3, random(0., 1.))) * (p2_stop - p2_start));

  const auto res_host = interpolator(p1_pt, p2_pt) * ctype(n_el);

  auto [p1_idx, p2_idx] = coords.backward(p1_pt, p2_pt);
  p1_idx = std::max((ctype)0, std::min(p1_idx, ctype(p1_size)));
  p2_idx = std::max((ctype)0, std::min(p2_idx, ctype(p2_size)));
  const auto res_local = p2_idx * ctype(n_el);

  if (!is_close(res_host, res_local, 1e-6 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-6 * n_el));
}