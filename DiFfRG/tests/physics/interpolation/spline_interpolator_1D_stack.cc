#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

TEMPLATE_TEST_CASE("Test 1D spline interpolation stack", "[float][double][complex][autodiff]", double, complex<double>,
                   autodiff::real, cxreal)
{
  using Type = TestType;
  DiFfRG::Init();

  using ctype = typename get_type::ctype<Type>;

  const int m_start = GENERATE(take(2, random(-50, 5)));
  const int m_size = GENERATE(take(2, random(2, 64)));
  const int m_stop = m_start + m_size;

  const ctype T = GENERATE(take(2, random(1e-6, 1e-1)));

  const ctype p_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p_stop = GENERATE(take(2, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(2, random(10, 100)));
  const ctype p_bias = GENERATE(take(2, random(1., 10.)));

  std::vector<Type> in_data(m_size * p_size);
  for (int i = 0; i < m_size; ++i)
    for (int j = 0; j < p_size; ++j)
      in_data[i * p_size + j] = i * p_size + j;
  BosonicCoordinates1DFiniteT<int, ctype> coords(m_start, m_stop, T, p_size, p_start, p_stop, p_bias);
  SplineInterpolator1DStack<Type, BosonicCoordinates1DFiniteT<int, ctype>, CPU_memory> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(2, random(2, 200)));
  const ctype m_pt = (m_start + int(GENERATE(take(2, random(0., 1.))) * (m_stop - m_start - 0.5))) * 2. * M_PI * T;
  const ctype p_pt = (p_start + GENERATE(take(2, random(0., 1.))) * (p_stop - p_start));

  const auto res_host = interpolator(m_pt, p_pt) * ctype(n_el);

  auto [m_idx, p_idx] = coords.backward(m_pt, p_pt);

  p_idx = std::max(ctype(0), std::min(p_idx, ctype(p_size)));
  const auto res_local = (in_data[m_idx * p_size + std::floor(p_idx)] +
                          (p_idx - std::floor(p_idx)) * (in_data[m_idx * p_size + std::ceil(p_idx)] -
                                                         in_data[m_idx * p_size + std::floor(p_idx)])) *
                         ctype(n_el);

  if (!is_close(res_host, res_local, 1e-6 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-6 * n_el));
}

TEMPLATE_TEST_CASE("Test 1D spline interpolation stack GPU", "[float][double][complex][autodiff]", double,
                   complex<double>, autodiff::real, cxreal)
{
  using Type = TestType;
  DiFfRG::Init();

  using ctype = typename get_type::ctype<Type>;

  const int m_start = GENERATE(take(2, random(-50, 5)));
  const int m_size = GENERATE(take(2, random(2, 64)));
  const int m_stop = m_start + m_size;

  const ctype T = GENERATE(take(2, random(1e-6, 1e-1)));

  const ctype p_start = GENERATE(take(2, random(1e-6, 1e-1)));
  const ctype p_stop = GENERATE(take(2, random(1, 100))) + p_start;
  const int p_size = GENERATE(take(2, random(10, 100)));
  const ctype p_bias = GENERATE(take(2, random(1., 10.)));

  std::vector<Type> in_data(m_size * p_size);
  for (int i = 0; i < m_size; ++i)
    for (int j = 0; j < p_size; ++j)
      in_data[i * p_size + j] = i * p_size + j;
  BosonicCoordinates1DFiniteT<int, ctype> coords(m_start, m_stop, T, p_size, p_start, p_stop, p_bias);
  SplineInterpolator1DStack<Type, BosonicCoordinates1DFiniteT<int, ctype>, GPU_memory> interpolator(coords);
  interpolator.update(in_data.data());

  const int n_el = GENERATE(take(2, random(2, 200)));
  const ctype m_pt = (m_start + int(GENERATE(take(2, random(0., 1.))) * (m_stop - m_start - 0.5))) * 2. * M_PI * T;
  const ctype p_pt = (p_start + GENERATE(take(2, random(0., 1.))) * (p_stop - p_start));

  const auto res_host = interpolator.CPU()(m_pt, p_pt) * ctype(n_el);

  auto [m_idx, p_idx] = coords.backward(m_pt, p_pt);

  p_idx = std::max(ctype(0), std::min(p_idx, ctype(p_size)));
  const auto res_local = (in_data[m_idx * p_size + std::floor(p_idx)] +
                          (p_idx - std::floor(p_idx)) * (in_data[m_idx * p_size + std::ceil(p_idx)] -
                                                         in_data[m_idx * p_size + std::floor(p_idx)])) *
                         ctype(n_el);

  Type res_gpu;
  Kokkos::parallel_reduce(
      "Get one point", Kokkos::RangePolicy(0, n_el),
      KOKKOS_LAMBDA(const uint, Type &update) { update += interpolator(m_pt, p_pt); }, res_gpu);

  if (!is_close(res_host, res_local, 1e-6 * n_el))
    std::cout << "host: " << res_host << " local: " << res_local << std::endl;
  if (!is_close(res_gpu, res_local, 1e-6 * n_el))
    std::cout << "gpu: " << res_gpu << " local: " << res_local << std::endl;
  CHECK(is_close(res_host, res_local, 1e-6 * n_el));
  CHECK(is_close(res_gpu, res_local, 1e-6 * n_el));
}