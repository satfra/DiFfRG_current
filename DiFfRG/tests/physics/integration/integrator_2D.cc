#include "DiFfRG/common/initialize.hh"
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration/integrator_2D.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static KOKKOS_INLINE_FUNCTION auto kernel(const double qx, const double qy, const double /*c*/, const double x0,
                                            const double x1, const double x2, const double x3, const double y0,
                                            const double y1, const double y2, const double y3)
  {
    return (x0 + x1 * powr<1>(qx) + x2 * powr<2>(qx) + x3 * powr<3>(qx)) *
           (y0 + y1 * powr<1>(qy) + y2 * powr<2>(qy) + y3 * powr<3>(qy));
  }

  static KOKKOS_INLINE_FUNCTION auto constant(const double c, const double /*x0*/, const double /*x1*/,
                                              const double /*x2*/, const double /*x3*/, const double /*y0*/,
                                              const double /*y1*/, const double /*y2*/, const double /*y3*/)
  {
    return c;
  }
};

TEST_CASE("Test 2D host integrals", "[double][cpu][integration][quadrature]")
{
  DiFfRG::Initialize();

  const double x_min = GENERATE(take(2, random(-2., -1.)));
  const double y_min = GENERATE(take(2, random(-2., -1.)));
  const double x_max = GENERATE(take(2, random(1., 2.)));
  const double y_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator2D<double, PolyIntegrand, CPU_exec> integrator(quadrature_provider, {32, 32}, {x_min, y_min},
                                                           {x_max, y_max});

  SECTION("Volume integral")
  {
    const double reference_integral = (x_max - x_min) * (y_max - y_min);

    double integral{};
    integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0.);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-9)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
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
        GENERATE(take(1, random(-1., 1.))),      // x3
    });
    const auto y_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(take_n, random(-1., 1.))), // y1
        GENERATE(take(take_n, random(-1., 1.))), // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });

    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(x_min, x_max) * y_poly.integral(y_min, y_max);

    double integral{};
    integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], y_poly[0], y_poly[1], y_poly[2],
                   y_poly[3]);
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

TEST_CASE("Test 2D device integrals", "[double][cpu][integration][quadrature]")
{
  DiFfRG::Initialize();

  const double x_min = GENERATE(take(2, random(-2., -1.)));
  const double y_min = GENERATE(take(2, random(-2., -1.)));
  const double x_max = GENERATE(take(2, random(1., 2.)));
  const double y_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator2D<double, PolyIntegrand, GPU_exec> integrator(quadrature_provider, {32, 32}, {x_min, y_min},
                                                           {x_max, y_max});

  SECTION("Volume integral")
  {
    const double reference_integral = (x_max - x_min) * (y_max - y_min);

    double integral{};
    integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0.);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-9)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
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
        GENERATE(take(1, random(-1., 1.))),      // x3
    });
    const auto y_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(take_n, random(-1., 1.))), // y1
        GENERATE(take(take_n, random(-1., 1.))), // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });

    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(x_min, x_max) * y_poly.integral(y_min, y_max);

    double integral{};
    integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], y_poly[0], y_poly[1], y_poly[2],
                   y_poly[3]);
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