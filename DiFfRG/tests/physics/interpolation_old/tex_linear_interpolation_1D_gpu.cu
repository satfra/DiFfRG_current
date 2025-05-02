#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/tex_linear_interpolation_1D.hh>

template <typename NT, typename LIN> __global__ void interp_kernel(NT *dest, LIN lin, float at)
{
  uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
  dest[idx_x] = lin(at);
}

using namespace DiFfRG;

TEST_CASE("Test 1D gpu texture interpolation", "[1D][texture interpolation]")
{
  const float p_start = GENERATE(take(3, random(1e-6, 1e-1)));
  const float p_stop = GENERATE(take(3, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(3, random(10, 100)));
  const float p_bias = GENERATE(take(3, random(1., 10.)));

  std::vector<float> empty_data(p_size);
  std::vector<float> in_data(p_size);
  for (int j = 0; j < p_size; ++j)
    in_data[j] = j;
  LogarithmicCoordinates1D<float> coords(p_size, p_start, p_stop, p_bias);
  TexLinearInterpolator1D<float, LogarithmicCoordinates1D<float>> interpolator(empty_data, coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(3, random(2, 200)));
  const float p_pt = (p_start + GENERATE(take(10, random(0., 1.))) * (p_stop - p_start));

  thrust::device_vector<float> dest(n_el, 0.);
  interp_kernel<<<1, n_el>>>(thrust::raw_pointer_cast(dest.data()), interpolator, p_pt);
  check_cuda("interp_kernel");

  const auto res_host = interpolator(p_pt) * float(n_el);
  const auto res_device = thrust::reduce(dest.begin(), dest.end());

  auto p_idx = coords.backward(p_pt);
  p_idx = std::max(0.f, std::min(p_idx, float(p_size)));
  const auto res_local = (in_data[std::floor(p_idx)] +
                          (p_idx - std::floor(p_idx)) * (in_data[std::ceil(p_idx)] - in_data[std::floor(p_idx)])) *
                         float(n_el);

  if (!is_close(res_host, res_local, 1e-10 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-10 * n_el));

  using std::abs;
  if (!is_close(res_device, res_local, 2e-3 * n_el))
    std::cout << "device: " << res_device << " local: " << res_local << std::endl;
  CHECK(is_close(res_device, res_local, 2e-3 * n_el));
}