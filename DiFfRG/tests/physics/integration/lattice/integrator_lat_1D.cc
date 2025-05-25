#include <Kokkos_Core.hpp>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "../boilerplate/poly_integrand.hh"
#include <DiFfRG/common/initialize.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat_1D.hh>

using namespace DiFfRG;

TEMPLATE_TEST_CASE("Test 1D lattice integrals on host", "[lattice][double][float][complex][autodiff]", double, float,
                   complex<double>, complex<float>, autodiff::real)
{
  using T = TestType;

  DiFfRG::Initialize();

  using ctype = typename get_type::ctype<T>;

  const ctype a = GENERATE(take(2, random(0.01, 1.)));
  const uint size = GENERATE(16, 32, 64, 128, 256);
  const bool q0_symmetric = GENERATE(false, true);

  IntegratorLat1D<T, PolyIntegrand<1, T>, CPU_exec> integrator({{size}}, {{a}}, q0_symmetric);

  SECTION("Volume integral")
  {
    const ctype reference_integral = 1 / a;

    T integral{};
    integrator.get(integral, 0., 1., 0., 0., 0.);

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }

    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}
/*
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
}*/