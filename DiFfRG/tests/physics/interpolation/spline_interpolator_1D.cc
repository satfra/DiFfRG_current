#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

TEST_CASE("Test template constraints for SplineInterpolator1D", "[interpolator]")
{
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1D<double, LinCoordinates, CPU_memory>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1D<complex<double>, LinCoordinates, CPU_memory>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1D<autodiff::real, LinCoordinates, CPU_memory>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1D<cxreal, LinCoordinates, CPU_memory>>);

  STATIC_REQUIRE(is_interpolator<SplineInterpolator1D<double, LinCoordinates, GPU_memory>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1D<complex<double>, LinCoordinates, GPU_memory>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1D<autodiff::real, LinCoordinates, GPU_memory>>);
  STATIC_REQUIRE(is_interpolator<SplineInterpolator1D<cxreal, LinCoordinates, GPU_memory>>);

  STATIC_REQUIRE(!is_interpolator<int>);
  STATIC_REQUIRE(!is_interpolator<std::array<double, 3>>);
}

TEMPLATE_TEST_CASE("Test 1D spline interpolation", "[float][double][complex][autodiff]", double, complex<double>,
                   autodiff::real, cxreal)
{
  using T = TestType;
  DiFfRG::Init();

  using ctype = typename get_type::ctype<T>;

  const ctype p_start = GENERATE(take(3, random(1e-6, 1e-1)));
  const ctype p_stop = GENERATE(take(3, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(3, random(10, 100)));
  const ctype p_bias = GENERATE(take(3, random(1., 10.)));

  std::vector<T> in_data(p_size);
  for (int j = 0; j < p_size; ++j)
    in_data[j] = j;
  LogarithmicCoordinates1D<ctype> coords(p_size, p_start, p_stop, p_bias);
  SplineInterpolator1D<T, LogarithmicCoordinates1D<ctype>, CPU_memory> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(3, random(2, 200)));
  const ctype p_pt = (p_start + GENERATE(take(10, random(0., 1.))) * (p_stop - p_start));

  const auto res_host = interpolator(p_pt) * ctype(n_el);

  auto p_idx = coords.backward(p_pt);
  p_idx = std::max((ctype)0, std::min(p_idx, ctype(p_size)));
  const auto res_local = (in_data[std::floor(p_idx)] +
                          (p_idx - std::floor(p_idx)) * (in_data[std::ceil(p_idx)] - in_data[std::floor(p_idx)])) *
                         ctype(n_el);
  constexpr ctype expected_precision = std::numeric_limits<ctype>::epsilon() * 1e2;
  if (!is_close(res_host, res_local, expected_precision))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, expected_precision));
}

TEMPLATE_TEST_CASE("Test GPU 1D spline interpolation", "[float][double][complex][autodiff]", double, complex<double>,
                   autodiff::real, cxreal)
{
  using T = TestType;
  DiFfRG::Init();

  using ctype = typename get_type::ctype<T>;

  const ctype p_start = GENERATE(take(3, random(1e-6, 1e-1)));
  const ctype p_stop = GENERATE(take(3, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(3, random(10, 100)));
  const ctype p_bias = GENERATE(take(3, random(1., 10.)));

  std::vector<T> in_data(p_size);
  for (int j = 0; j < p_size; ++j)
    in_data[j] = j;
  LogarithmicCoordinates1D<ctype> coords(p_size, p_start, p_stop, p_bias);
  SplineInterpolator1D<T, LogarithmicCoordinates1D<ctype>, GPU_memory> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(3, random(2, 200)));
  const ctype p_pt = (p_start + GENERATE(take(10, random(0., 1.))) * (p_stop - p_start));

  // Moving back and forth should not affect the data!
  const auto res_host = interpolator.CPU().GPU().CPU()(p_pt) * ctype(n_el);

  auto p_idx = coords.backward(p_pt);
  p_idx = std::max((ctype)0, std::min(p_idx, ctype(p_size)));
  const auto res_local = (in_data[std::floor(p_idx)] +
                          (p_idx - std::floor(p_idx)) * (in_data[std::ceil(p_idx)] - in_data[std::floor(p_idx)])) *
                         ctype(n_el);

  T res_gpu;
  Kokkos::parallel_reduce(
      "Get one point", Kokkos::RangePolicy(0, n_el),
      KOKKOS_LAMBDA(const uint, T &update) { update += interpolator(p_pt); }, res_gpu);

  constexpr ctype expected_precision = std::numeric_limits<ctype>::epsilon() * 1e2;
  if (!is_close(res_host, res_local, expected_precision))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  if (!is_close(res_gpu, res_local, expected_precision))
    std::cout << "gpu: " << res_gpu << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, expected_precision));
  CHECK(is_close(res_gpu, res_local, expected_precision));
}
