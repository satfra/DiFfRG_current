#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/math.hh>

using namespace DiFfRG;

// minimal kernel to test GPU autodiff
__global__ void test_1(double *z, const double muq)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  z[i] = real((complex<double>(muq, 0.)) * (complex<double>(muq, 0.)));
  // z[i] = cuda::std::real(cuda::std::complex<double>(muq * muq, 0.));
  // z[i] = powr<2>(muq);
  // z[i] = std::real(1. * (powr<2>(std::complex<double>(0., muq)) + 0.5));
}

TEST_CASE("Test complex on GPU", "[complex][gpu]")
{
  const int N = GENERATE(1 << 10, 1 << 20, 1 << 22);

  SECTION("Test 1")
  {
    thrust::device_vector<double> z(N, 0.);

    const int threads_per_block = 256;
    const int blocks = N / threads_per_block;

    // Launch a kernel on the GPU
    test_1<<<blocks, threads_per_block>>>(thrust::raw_pointer_cast(z.data()), 0.5);
    const double result = thrust::reduce(z.begin(), z.end(), double(0.0), thrust::plus<double>());

    // Check for errors
    if (!is_close(double(result), 0.25 * N, std::numeric_limits<double>::epsilon() * 1e+1))
      std::cout << "N: " << N << "result: " << double(result) << " expected: " << 0.25 * N << std::endl;
    REQUIRE(is_close(double(result), 0.25 * N, std::numeric_limits<double>::epsilon() * 1e+1));
  }
}
