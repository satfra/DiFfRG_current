#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/linear_interpolation_1D.hh>
#include <DiFfRG/physics/interpolation/tex_linear_interpolation_1D.hh>

template <typename NT, typename LIN> __global__ void interp_kernel(NT *dest, LIN lin, float at)
{
  uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
  dest[idx_x] = lin(at);
}

using namespace DiFfRG;

TEST_CASE("Test 1D gpu texture interpolation", "[1D][texture interpolation]")
{
  const int p_start = GENERATE(take(1, random(1e-6, 1e-1)));
  const int p_stop = GENERATE(take(1, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(1, random(10, 100)));
  const float p_bias = GENERATE(take(1, random(1., 10.)));

  std::vector<float> empty_data(p_size);
  std::vector<double> double_empty_data(p_size);
  std::vector<float> in_data(p_size);
  for (int j = 0; j < p_size; ++j)
    in_data[j] = j;
  LogarithmicCoordinates1D<float> coords(p_size, p_start, p_stop, p_bias);
  TexLinearInterpolator1D<float, LogarithmicCoordinates1D<float>> interpolator(empty_data, coords);
  interpolator.update(in_data.data());
  LinearInterpolator1D<float, LogarithmicCoordinates1D<float>> lin_interpolator(empty_data, coords);
  lin_interpolator.update(in_data.data());
  LinearInterpolator1D<double, LogarithmicCoordinates1D<double>> double_lin_interpolator(double_empty_data, coords);
  double_lin_interpolator.update(in_data.data());

  const uint n_el = GENERATE(1e+3, 1e+6, 1e+8);
  const float p_pt = (p_start + GENERATE(take(1, random(0., 1.))) * (p_stop - p_start));

  BENCHMARK_ADVANCED("TexInterpolator " + std::to_string(n_el))(Catch::Benchmark::Chronometer meter)
  {
    thrust::device_vector<float> dest(n_el, 0.);
    meter.measure([&] {
      interp_kernel<<<1, n_el>>>(thrust::raw_pointer_cast(dest.data()), interpolator, p_pt);
      const auto res_device = thrust::reduce(dest.begin(), dest.end());
    });
  };
  BENCHMARK_ADVANCED("LinInterpolator " + std::to_string(n_el))(Catch::Benchmark::Chronometer meter)
  {
    thrust::device_vector<float> dest(n_el, 0.);
    meter.measure([&] {
      interp_kernel<<<1, n_el>>>(thrust::raw_pointer_cast(dest.data()), lin_interpolator, p_pt);
      const auto res_device = thrust::reduce(dest.begin(), dest.end());
    });
  };
  BENCHMARK_ADVANCED("DoubleLinInterpolator " + std::to_string(n_el))(Catch::Benchmark::Chronometer meter)
  {
    thrust::device_vector<float> dest(n_el, 0.);
    meter.measure([&] {
      interp_kernel<<<1, n_el>>>(thrust::raw_pointer_cast(dest.data()), double_lin_interpolator, p_pt);
      const auto res_device = thrust::reduce(dest.begin(), dest.end());
    });
  };
}