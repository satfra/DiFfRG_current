#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/math.hh>

using namespace DiFfRG;

// minimal kernel to test GPU autodiff
__global__ void test_autodiff(const autodiff::real *x, const autodiff::real *y, autodiff::real *z)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  z[i] = x[i] * y[i];
}

// kernel to multiply a complex<autodiff::real> with a real and store the real and imag in two arrays
__global__ void test_autodiff_complex(const cxReal *x, const autodiff::real *y, autodiff::real *z, autodiff::real *w)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // additional operations are to test whether all operators are overloaded correctly
  z[i] = real(2. * x[i] / 2. * real(y[i] * complex<double>(1., 0.)) + 1. - 1.);
  w[i] = imag(2. * x[i] / 2. * real(y[i] * complex<double>(1., 0.)) + 1. - 1.);
}

TEST_CASE("Test autodiff on GPU", "[autodiff][gpu]")
{
  const int N = GENERATE(1 << 10, 1 << 20, 1 << 22);

  SECTION("Basic autodiff")
  {
    thrust::device_vector<autodiff::real> x(N, std::array<double, 2>{{1.0, 2.0}});
    thrust::device_vector<autodiff::real> y(N, std::array<double, 2>{{2.0, 3.0}});
    thrust::device_vector<autodiff::real> z(N);

    const int threads_per_block = 256;
    const int blocks = N / threads_per_block;

    // Launch a kernel on the GPU
    test_autodiff<<<blocks, threads_per_block>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()),
                                                 thrust::raw_pointer_cast(z.data()));
    const autodiff::real result =
        thrust::reduce(z.begin(), z.end(), autodiff::real(0.0), thrust::plus<autodiff::real>());

    // Check for errors (all values should be 2.0)
    if (!is_close(autodiff::val(result), 2.0 * N, std::numeric_limits<double>::epsilon()))
      std::cout << "result: " << autodiff::val(result) << " expected: " << 2.0 * N << std::endl;
    REQUIRE(is_close(autodiff::val(result), 2.0 * N, std::numeric_limits<double>::epsilon()));
    // Check for errors (all derivatives should be 7.0)
    REQUIRE(is_close(autodiff::derivative(result), 7.0 * N, std::numeric_limits<double>::epsilon()));
  }
  SECTION("autodiff with cxReal")
  {
    const cxReal x_val({{complex<double>(1.0, 2.0), complex<double>(2.0, 3.0)}});
    thrust::device_vector<cxReal> x(N, x_val);
    thrust::device_vector<autodiff::real> y(N, std::array<double, 2>{{2.0, 3.0}});
    thrust::device_vector<autodiff::real> z(N);
    thrust::device_vector<autodiff::real> w(N);

    const int threads_per_block = 256;
    const int blocks = N / threads_per_block;

    // Launch a kernel on the GPU
    test_autodiff_complex<<<blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()), thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(w.data()));
    const autodiff::real result_z =
        thrust::reduce(z.begin(), z.end(), autodiff::real(0.0), thrust::plus<autodiff::real>());
    const autodiff::real result_w =
        thrust::reduce(w.begin(), w.end(), autodiff::real(0.0), thrust::plus<autodiff::real>());

    // Check for errors
    REQUIRE(is_close(autodiff::val(result_z), 2.0 * N, std::numeric_limits<double>::epsilon()));
    REQUIRE(is_close(autodiff::val(result_w), 4.0 * N, std::numeric_limits<double>::epsilon()));
    REQUIRE(is_close(autodiff::derivative(result_z), 7.0 * N, std::numeric_limits<double>::epsilon()));
    REQUIRE(is_close(autodiff::derivative(result_w), 12.0 * N, std::numeric_limits<double>::epsilon()));
  }
}