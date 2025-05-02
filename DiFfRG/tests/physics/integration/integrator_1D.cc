#include <Kokkos_Core.hpp>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/initialize.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/integrator_1D.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static KOKKOS_INLINE_FUNCTION auto kernel(const double q, const double /*c*/, const double x0, const double x1,
                                            const double x2, const double x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q));
  }

  static KOKKOS_INLINE_FUNCTION auto constant(const double c, const double /*x0*/, const double /*x1*/,
                                              const double /*x2*/, const double /*x3*/)
  {
    return c;
  }
};

TEMPLATE_TEST_CASE_SIG("Test 1D integrals on device", "[double][float][complex][autodiff]", ((typename T), T), double,
                       float, complex<double>, complex<float>, autodiff::real)
{
  DiFfRG::Initialize();

  using ctype = typename get_type::ctype<T>;

  const ctype x_min = GENERATE(take(2, random(-2., -1.)));
  const ctype x_max = GENERATE(take(2, random(1., 2.)));
  const uint size = GENERATE(16, 32, 64, 128, 256);

  QuadratureProvider quadrature_provider;
  Integrator1D<T, PolyIntegrand, GPU_exec> integrator(quadrature_provider, {size}, {x_min}, {x_max});

  SECTION("Volume integral")
  {
    const ctype reference_integral = (x_max - x_min);

    T integral{};
    integrator.get(integral, 0., 1., 0., 0., 0.);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }

    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}

TEMPLATE_TEST_CASE_SIG("Test 1D integrals on host", "[double][float][complex][autodiff]", ((typename T), T), double,
                       float, complex<double>, complex<float>, autodiff::real)
{
  DiFfRG::Initialize();

  using ctype = typename get_type::ctype<T>;

  const ctype x_min = GENERATE(take(2, random(-2., -1.)));
  const ctype x_max = GENERATE(take(2, random(1., 2.)));
  const uint size = GENERATE(16, 32, 64, 128, 256);

  QuadratureProvider quadrature_provider;
  Integrator1D<T, PolyIntegrand, CPU_exec> integrator(quadrature_provider, {size}, {x_min}, {x_max});

  SECTION("Volume integral")
  {
    const ctype reference_integral = (x_max - x_min);

    T integral{};
    integrator.get(integral, 0., 1., 0., 0., 0.);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }

    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}

TEST_CASE("Test 1D device integrals", "[double][gpu][integration][quadrature]")
{
  DiFfRG::Initialize();

  const double x_min = GENERATE(take(2, random(-2., -1.)));
  const double x_max = GENERATE(take(2, random(1., 2.)));
  const uint size = GENERATE(16, 32, 64, 128, 256);

  QuadratureProvider quadrature_provider;
  Integrator1D<double, PolyIntegrand, GPU_exec> integrator(quadrature_provider, {size}, {x_min}, {x_max});

  SECTION("Volume integral")
  {
    const double reference_integral = 2. * (x_max - x_min);

    double integral{};
    integrator.get(integral, 0., 2., 0., 0., 0.);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-9)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }

    constexpr double expected_precision = 1e-9;
    CHECK(integral == Catch::Approx(reference_integral).epsilon(expected_precision));
  }

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;
    const auto x_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(take_n, random(-1., 1.))), // x2
        GENERATE(take(take_n, random(-1., 1.))), // x3
    });

    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(x_min, x_max);

    double integral{};
    integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3]);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-9)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }

    constexpr double expected_precision = 1e-9;
    CHECK(integral == Catch::Approx(reference_integral).epsilon(expected_precision));
  }
}

TEST_CASE("Test 1D host integrals", "[double][cpu][integration][quadrature]")
{
  DiFfRG::Initialize();

  const double x_min = GENERATE(take(2, random(-2., -1.)));
  const double x_max = GENERATE(take(2, random(1., 2.)));
  const uint size = GENERATE(16, 32, 64, 128, 256);

  QuadratureProvider quadrature_provider;
  Integrator1D<double, PolyIntegrand, CPU_exec> integrator(quadrature_provider, {size}, {x_min}, {x_max});

  SECTION("Volume integral")
  {
    const double reference_integral = (x_max - x_min);

    double integral{};
    integrator.get(integral, 0., 1., 0., 0., 0.);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-9)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }

    constexpr double expected_precision = 1e-9;
    CHECK(integral == Catch::Approx(reference_integral).epsilon(expected_precision));
  }

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;
    const auto x_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(take_n, random(-1., 1.))), // x2
        GENERATE(take(take_n, random(-1., 1.))), // x3
    });

    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(x_min, x_max);

    double integral{};
    integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3]);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}