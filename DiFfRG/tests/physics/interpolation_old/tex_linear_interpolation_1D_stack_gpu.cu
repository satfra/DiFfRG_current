#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/grid/combined_coordinates.hh>
#include <DiFfRG/physics/interpolation/tex_linear_interpolation_1D_stack.hh>

using namespace DiFfRG;

template <typename NT, typename LIN> __global__ void interp_kernel(NT *dest, LIN lin, float m_at, float p_at)
{
  uint idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
  dest[idx_x] = lin(m_at, p_at);
}

TEST_CASE("Test 1D gpu texture interpolation stack", "[1D][texture interpolation stack]")
{
  const int m_start = GENERATE(take(2, random(-50, 5)));
  const int m_size = GENERATE(take(2, random(2, 64)));
  const int m_stop = m_start + m_size;

  const float T = GENERATE(take(2, random(1e-6, 1e-1)));

  const float p_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const float p_stop = GENERATE(take(2, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(2, random(10, 100)));

  const float p_bias = GENERATE(take(2, random(1., 10.)));

  std::vector<float> in_data(m_size * p_size);
  for (int i = 0; i < m_size; ++i)
    for (int j = 0; j < p_size; ++j)
      in_data[i * p_size + j] = i * p_size + j;
  BosonicCoordinates1DFiniteT<int, float> coords(m_start, m_stop, T, p_size, p_start, p_stop, p_bias);
  TexLinearInterpolator1DStack<float, BosonicCoordinates1DFiniteT<int, float>> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(2, random(2, 200)));
  const float m_pt = (m_start + int(GENERATE(take(2, random(0., 1.))) * (m_stop - m_start - 0.5))) * 2. * M_PI * T;
  const float p_pt = (p_start + GENERATE(take(2, random(0., 1.))) * (p_stop - p_start));

  thrust::device_vector<float> dest(n_el, 0.);
  interp_kernel<<<1, n_el>>>(thrust::raw_pointer_cast(dest.data()), interpolator, m_pt, p_pt);
  check_cuda("interp_kernel");

  const auto res_host = interpolator(m_pt, p_pt) * float(n_el);
  const auto res_device = thrust::reduce(dest.begin(), dest.end());

  auto [m_idx, p_idx] = coords.backward(m_pt, p_pt);
  p_idx = std::max(0.f, std::min(p_idx, float(p_size)));
  const auto res_local = (in_data[m_idx * p_size + std::floor(p_idx)] +
                          (p_idx - std::floor(p_idx)) * (in_data[m_idx * p_size + std::ceil(p_idx)] -
                                                         in_data[m_idx * p_size + std::floor(p_idx)])) *
                         float(n_el);

  if (!is_close(res_host, res_local, 1e-7 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-7 * n_el));

  if (!is_close(res_device, res_local, 5e-4 * n_el))
    std::cout << "device: " << res_device << " local: " << res_local << std::endl;
  CHECK(is_close(res_device, res_local, 5e-4 * n_el));
}