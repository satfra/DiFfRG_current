#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/math.hh>

using namespace DiFfRG;

// minimal kernel to test GPU autodiff
__global__ void test_1(autodiff::real *z, const double muq)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  z[i] =
      real(cxReal({{complex<double>(muq), complex<double>(0)}}) * cxReal({{complex<double>(muq), complex<double>(0)}}));
  const auto k = tanh(z[i]);

  // z[i] = cuda::std::real(cuda::std::complex<double>(muq * muq, 0.));
  // z[i] = powr<2>(muq);
  // z[i] = std::real(1. * (powr<2>(std::complex<double>(0., muq)) + 0.5));
}

__device__ bool validate_no_ad(const complex<double> &result, const complex<double> &value, const char *msg)
{
  if (!is_close(result.real(), value.real(), 1e-10) || !is_close(result.imag(), value.imag(), 1e-10)) {
    printf("Error: %s: (%f, %f) is not equal to (%f, %f) (expected)\n", msg, result.real(), result.imag(), value.real(),
           value.imag());
    return false;
  }
  // printf("Success: %s: (%f, %f) is equal to (%f, %f) (expected)\n", msg, result.real(), result.imag(), value.real(),
  // value.imag());
  return true;
}

__device__ bool validate(const cxReal &result, const complex<double> &value, const complex<double> &derivative,
                         const char *msg)
{
  if (!is_close(autodiff::val(result).real(), value.real(), 1e-10) ||
      !is_close(autodiff::val(result).imag(), value.imag(), 1e-10)) {
    printf("Error in value: %s: (%f, %f) is not equal to (%f, %f) (expected)\n", msg, autodiff::val(result).real(),
           autodiff::val(result).imag(), value.real(), value.imag());
    return false;
  }
  if (!is_close(autodiff::derivative(result).real(), derivative.real(), 1e-10) ||
      !is_close(autodiff::derivative(result).imag(), derivative.imag(), 1e-10)) {
    printf("Error in derivative: %s: (%f, %f) is not equal to (%f, %f) (expected)\n", msg,
           autodiff::derivative(result).real(), autodiff::derivative(result).imag(), derivative.real(),
           derivative.imag());
    return false;
  }
  // printf("Success (value): %s: (%f, %f) is equal to (%f, %f) (expected)\n", msg, autodiff::val(result).real(),
  // autodiff::val(result).imag(), value.real(), value.imag()); printf("Success (derivative): %s: (%f, %f) is equal to
  // (%f, %f) (expected)\n", msg, autodiff::derivative(result).real(), autodiff::derivative(result).imag(),
  // derivative.real(), derivative.imag());
  return true;
}

__device__ bool validate(const autodiff::real &result, const double value, const double derivative, const char *msg)
{
  if (!is_close(result[0], value, 1e-10)) {
    printf("Error in value: %s: %f is not equal to %f (expected)\n", msg, result[0], value);
    return false;
  }
  if (!is_close(result[1], derivative, 1e-10)) {
    printf("Error in derivative: %s: %f is not equal to %f (expected)\n", msg, result[1], derivative);
    return false;
  }
  // printf("Success (value): %s: %f is equal to %f (expected)\n", msg, result[0], value);
  // printf("Success (derivative): %s: %f is equal to %f (expected)\n", msg, result[1], derivative);
  return true;
}

__global__ void test_multiplication(bool *result)
{
  const double x = 2.0;
  autodiff::real ad_x;
  ad_x[0] = 3.0;
  ad_x[1] = 5.0;
  const complex<double> c_x(2.0, 3.0);
  cxReal ad_c_x;
  ad_c_x[0] = complex<double>(3.0, 2.0);
  ad_c_x[1] = complex<double>(5.0, -5.0);

  // x, c_x
  result[0] &= validate_no_ad(c_x * x, complex<double>(4.0, 6.0), "c_x * x");
  result[0] &= validate_no_ad(x * c_x, complex<double>(4.0, 6.0), "x * c_x");

  // x, ad_x
  result[0] &= validate(ad_x * x, 6.0, 10.0, "ad_x * x");
  result[0] &= validate(x * ad_x, 6.0, 10.0, "x * ad_x");

  // x, ad_c_x
  result[0] &= validate(ad_c_x * x, complex<double>(6.0, 4.0), complex<double>(10.0, -10.0), "ad_c_x * x");
  result[0] &= validate(x * ad_c_x, complex<double>(6.0, 4.0), complex<double>(10.0, -10.0), "x * ad_c_x");

  // c_x, ad_x
  result[0] &= validate(ad_x * c_x, complex<double>(6.0, 9.0), complex<double>(10.0, 15.0), "ad_x * c_x");
  result[0] &= validate(c_x * ad_x, complex<double>(6.0, 9.0), complex<double>(10.0, 15.0), "c_x * ad_x");

  // c_x, ad_c_x
  result[0] &= validate(ad_c_x * c_x, complex<double>(0.0, 13.0), complex<double>(25.0, 5.0), "ad_c_x * c_x");
  result[0] &= validate(c_x * ad_c_x, complex<double>(0.0, 13.0), complex<double>(25.0, 5.0), "c_x * ad_c_x");

  // ad_x, ad_c_x
  result[0] &= validate(ad_x * ad_c_x, complex<double>(9.0, 6.0), complex<double>(30.0, -5.0), "ad_x * ad_c_x");
  result[0] &= validate(ad_c_x * ad_x, complex<double>(9.0, 6.0), complex<double>(30.0, -5.0), "ad_c_x * ad_x");

  // ad_c_x, ad_c_x
  result[0] &= validate(ad_c_x * ad_c_x, complex<double>(5.0, 12.0), complex<double>(50.0, -10.0), "ad_c_x * ad_c_x");
}

__global__ void test_addition(bool *result)
{
  const double x = 2.0;
  autodiff::real ad_x;
  ad_x[0] = 3.0;
  ad_x[1] = 5.0;
  const complex<double> c_x(2.0, 3.0);
  cxReal ad_c_x;
  ad_c_x[0] = complex<double>(3.0, 2.0);
  ad_c_x[1] = complex<double>(5.0, -5.0);

  // x, c_x
  result[0] &= validate_no_ad(c_x + x, complex<double>(4.0, 3.0), "c_x + x");
  result[0] &= validate_no_ad(x + c_x, complex<double>(4.0, 3.0), "x + c_x");

  // x, ad_x
  result[0] &= validate(ad_x + x, 5.0, 5.0, "ad_x + x");
  result[0] &= validate(x + ad_x, 5.0, 5.0, "x + ad_x");

  // x, ad_c_x
  result[0] &= validate(ad_c_x + x, complex<double>(5.0, 2.0), complex<double>(5.0, -5.0), "ad_c_x + x");
  result[0] &= validate(x + ad_c_x, complex<double>(5.0, 2.0), complex<double>(5.0, -5.0), "x + ad_c_x");

  // c_x, ad_x
  result[0] &= validate(ad_x + c_x, complex<double>(5.0, 3.0), complex<double>(5.0, 0.0), "ad_x + c_x");
  result[0] &= validate(c_x + ad_x, complex<double>(5.0, 3.0), complex<double>(5.0, 0.0), "c_x + ad_x");

  // c_x, ad_c_x
  result[0] &= validate(ad_c_x + c_x, complex<double>(5.0, 5.0), complex<double>(5.0, -5.0), "ad_c_x + c_x");
  result[0] &= validate(c_x + ad_c_x, complex<double>(5.0, 5.0), complex<double>(5.0, -5.0), "c_x + ad_c_x");

  // ad_x, ad_c_x
  result[0] &= validate(ad_x + ad_c_x, complex<double>(6.0, 2.0), complex<double>(10.0, -5.0), "ad_x + ad_c_x");
  result[0] &= validate(ad_c_x + ad_x, complex<double>(6.0, 2.0), complex<double>(10.0, -5.0), "ad_c_x + ad_x");

  // ad_c_x, ad_c_x
  result[0] &= validate(ad_c_x + ad_c_x, complex<double>(6.0, 4.0), complex<double>(10.0, -10.0), "ad_c_x + ad_c_x");
}

__global__ void test_subtraction(bool *result)
{
  const double x = 2.0;
  autodiff::real ad_x;
  ad_x[0] = 3.0;
  ad_x[1] = 5.0;
  const complex<double> c_x(2.0, 3.0);
  cxReal ad_c_x;
  ad_c_x[0] = complex<double>(3.0, 2.0);
  ad_c_x[1] = complex<double>(5.0, -5.0);

  // x, c_x
  result[0] &= validate_no_ad(c_x - x, complex<double>(0.0, 3.0), "c_x - x");
  result[0] &= validate_no_ad(x - c_x, complex<double>(0.0, -3.0), "x - c_x");

  // x, ad_x
  result[0] &= validate(ad_x - x, 1.0, 5.0, "ad_x - x");
  result[0] &= validate(x - ad_x, -1.0, -5.0, "x - ad_x");

  // x, ad_c_x
  result[0] &= validate(ad_c_x - x, complex<double>(1.0, 2.0), complex<double>(5.0, -5.0), "ad_c_x - x");
  result[0] &= validate(x - ad_c_x, complex<double>(-1.0, -2.0), complex<double>(-5.0, 5.0), "x - ad_c_x");

  // c_x, ad_x
  result[0] &= validate(ad_x - c_x, complex<double>(1.0, -3.0), complex<double>(5.0, 0.0), "ad_x - c_x");
  result[0] &= validate(c_x - ad_x, complex<double>(-1.0, 3.0), complex<double>(-5.0, 0.0), "c_x - ad_x");

  // c_x, ad_c_x
  result[0] &= validate(ad_c_x - c_x, complex<double>(1.0, -1.0), complex<double>(5.0, -5.0), "ad_c_x - c_x");
  result[0] &= validate(c_x - ad_c_x, complex<double>(-1.0, 1.0), complex<double>(-5.0, 5.0), "c_x - ad_c_x");

  // ad_x, ad_c_x
  result[0] &= validate(ad_x - ad_c_x, complex<double>(0.0, -2.0), complex<double>(0.0, 5.0), "ad_x - ad_c_x");
  result[0] &= validate(ad_c_x - ad_x, complex<double>(0.0, 2.0), complex<double>(0.0, -5.0), "ad_c_x - ad_x");

  // ad_c_x, ad_c_x
  result[0] &= validate(ad_c_x - ad_c_x, complex<double>(0., 0.), complex<double>(0., 0.), "ad_c_x - ad_c_x");
}

__global__ void test_division(bool *result)
{
  const double x = 2.0;
  autodiff::real ad_x;
  ad_x[0] = 3.0;
  ad_x[1] = 5.0;
  const complex<double> c_x(2.0, 3.0);
  cxReal ad_c_x;
  ad_c_x[0] = complex<double>(3.0, 2.0);
  ad_c_x[1] = complex<double>(5.0, -5.0);

  // x, c_x
  result[0] &= validate_no_ad(c_x / x, complex<double>(1.0, 1.5), "c_x / x");
  result[0] &= validate_no_ad(x / c_x, complex<double>(4. / 13., -6. / 13.), "x / c_x");

  // x, ad_x
  result[0] &= validate(ad_x / x, 1.5, 5. / 2., "ad_x / x");
  result[0] &= validate(x / ad_x, 2. / 3., -10. / 9., "x / ad_x");

  // x, ad_c_x
  result[0] &= validate(ad_c_x / x, complex<double>(3. / 2., 1.), complex<double>(5. / 2., -5. / 2.), "ad_c_x / x");
  result[0] &= validate(x / ad_c_x, complex<double>(6. / 13., -4.0 / 13.), complex<double>(70. / 169., 170. / 169.),
                        "x / ad_c_x");

  // c_x, ad_x
  result[0] &=
      validate(ad_x / c_x, complex<double>(6. / 13., -9. / 13.), complex<double>(10. / 13., -15. / 13.), "ad_x / c_x");
  result[0] &= validate(c_x / ad_x, complex<double>(2. / 3., 1.), complex<double>(-10. / 9., -5. / 3.), "c_x / ad_x");

  // c_x, ad_c_x
  result[0] &= validate(ad_c_x / c_x, complex<double>(12. / 13., -5. / 13.), complex<double>(-5. / 13., -25. / 13.),
                        "ad_c_x / c_x");
  result[0] &= validate(c_x / ad_c_x, complex<double>(12. / 13., 5. / 13.), complex<double>(-185. / 169., 275. / 169.),
                        "c_x / adc_x");

  // ad_x, ad_c_x
  result[0] &= validate(ad_x / ad_c_x, complex<double>(9. / 13., -6. / 13.), complex<double>(300. / 169., 125. / 169.),
                        "ad_x / ad_c_x");
  result[0] &= validate(ad_c_x / ad_x, complex<double>(1., 2. / 3.), complex<double>(0.0, -25. / 9.), "ad_c_x / ad_x");

  // ad_c_x, ad_c_x
  result[0] &= validate(ad_c_x / ad_c_x, complex<double>(1., 0.), complex<double>(0., 0.), "ad_c_x / ad_c_x");
}

TEST_CASE("Test complex on GPU", "[autodiff][complex][gpu]")
{
  const int N = GENERATE(1 << 10, 1 << 20, 1 << 22);

  SECTION("Test 1")
  {
    thrust::device_vector<autodiff::real> z(N, 0.);

    const int threads_per_block = 256;
    const int blocks = N / threads_per_block;

    // Launch a kernel on the GPU
    test_1<<<blocks, threads_per_block>>>(thrust::raw_pointer_cast(z.data()), 0.5);
    const autodiff::real result =
        thrust::reduce(z.begin(), z.end(), autodiff::real(0.0), thrust::plus<autodiff::real>());

    // Check for errors
    if (!is_close(double(result[0]), 0.25 * N, std::numeric_limits<double>::epsilon() * 1e+1))
      std::cout << "N: " << N << "result: " << double(result) << " expected: " << 0.25 * N << std::endl;
    REQUIRE(is_close(double(result[0]), 0.25 * N, std::numeric_limits<double>::epsilon() * 1e+1));
  }
}

TEST_CASE("Test complex autodiff on GPU", "[autodiff][complex][gpu]")
{
  SECTION("Test Multiplication")
  {
    thrust::device_vector<bool> result(1, true);

    test_multiplication<<<1, 1>>>(thrust::raw_pointer_cast(result.data()));

    bool success = thrust::reduce(result.begin(), result.end(), true, thrust::logical_and<bool>());
    REQUIRE(success);

    if (success)
      std::cout << "Success: Multiplication" << std::endl;
    else
      std::cout << "Error: Multiplication" << std::endl;
  }

  SECTION("Test Addition")
  {
    thrust::device_vector<bool> result(1, true);

    test_addition<<<1, 1>>>(thrust::raw_pointer_cast(result.data()));

    bool success = thrust::reduce(result.begin(), result.end(), true, thrust::logical_and<bool>());
    REQUIRE(success);

    if (success)
      std::cout << "Success: Addition" << std::endl;
    else
      std::cout << "Error: Addition" << std::endl;
  }

  SECTION("Test Subtraction")
  {
    thrust::device_vector<bool> result(1, true);

    test_subtraction<<<1, 1>>>(thrust::raw_pointer_cast(result.data()));

    bool success = thrust::reduce(result.begin(), result.end(), true, thrust::logical_and<bool>());
    REQUIRE(success);

    if (success)
      std::cout << "Success: Subtraction" << std::endl;
    else
      std::cout << "Error: Subtraction" << std::endl;
  }

  SECTION("Test Division")
  {
    thrust::device_vector<bool> result(1, true);

    test_division<<<1, 1>>>(thrust::raw_pointer_cast(result.data()));

    bool success = thrust::reduce(result.begin(), result.end(), true, thrust::logical_and<bool>());
    REQUIRE(success);

    if (success)
      std::cout << "Success: Division" << std::endl;
    else
      std::cout << "Error: Division" << std::endl;
  }
}
