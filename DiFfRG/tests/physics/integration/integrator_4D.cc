#include <Kokkos_Macros.hpp>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/initialize.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/integrator_4D.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double qx, const double qy, const double qz, const double qw,
                                                 const double /*c*/, const double x0, const double x1, const double x2,
                                                 const double x3, const double y0, const double y1, const double y2,
                                                 const double y3, const double z0, const double z1, const double z2,
                                                 const double z3, const double w0, const double w1, const double w2,
                                                 const double w3)
  {
    return (x0 + x1 * powr<1>(qx) + x2 * powr<2>(qx) + x3 * powr<3>(qx)) *
           (y0 + y1 * powr<1>(qy) + y2 * powr<2>(qy) + y3 * powr<3>(qy)) *
           (z0 + z1 * powr<1>(qz) + z2 * powr<2>(qz) + z3 * powr<3>(qz)) *
           (w0 + w1 * powr<1>(qw) + w2 * powr<2>(qw) + w3 * powr<3>(qw));
  }

  static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double c, const double /*x0*/, const double /*x1*/,
                                                   const double /*x2*/, const double /*x3*/, const double /*y0*/,
                                                   const double /*y1*/, const double /*y2*/, const double /*y3*/,
                                                   const double /*z0*/, const double /*z1*/, const double /*z2*/,
                                                   const double /*z3*/, const double /*w0*/, const double /*w1*/,
                                                   const double /*w2*/, const double /*w3*/)
  {
    return c;
  }
};

TEST_CASE("Test 4D host integrals", "[double][cpu][integration][quadrature]")
{
  DiFfRG::Initialize();

  const double x_min = GENERATE(take(2, random(-2., -1.)));
  const double y_min = GENERATE(take(2, random(-2., -1.)));
  const double z_min = GENERATE(take(2, random(-2., -1.)));
  const double w_min = GENERATE(take(2, random(-2., -1.)));

  const double x_max = GENERATE(take(2, random(1., 2.)));
  const double y_max = GENERATE(take(2, random(1., 2.)));
  const double z_max = GENERATE(take(2, random(1., 2.)));
  const double w_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator4D<double, PolyIntegrand, CPU_exec> integrator(quadrature_provider, {32, 8, 8, 8},
                                                           {x_min, y_min, z_min, w_min}, {x_max, y_max, z_max, w_max});

  SECTION("Volume integral")
  {
    const double reference_integral = (x_max - x_min) * (y_max - y_min) * (z_max - z_min) * (w_max - w_min);

    double integral{};
    integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-9)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }

    constexpr double expected_precision = 1e-9;
    CHECK(integral == Catch::Approx(reference_integral).epsilon(expected_precision));
  }

  SECTION("Volume map")
  {
    const uint rsize = GENERATE(16, 32);
    std::vector<double> integral_view(rsize);
    LinearCoordinates1D<double> coordinates(rsize, 0., 0.);

    integrator.map(integral_view.data(), coordinates, 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.)
        .fence();

    for (uint i = 0; i < rsize; ++i)
      CHECK(is_close(integral_view[i],
                     coordinates.forward(i) + (x_max - x_min) * (y_max - y_min) * (z_max - z_min) * (w_max - w_min),
                     1e-6));
  };

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;
    const auto x_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(1, random(-1., 1.))),      // x1
        GENERATE(take(1, random(-1., 1.))),      // x2
        GENERATE(take(1, random(-1., 1.))),      // x3
    });
    const auto y_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(1, random(-1., 1.))),      // y1
        GENERATE(take(1, random(-1., 1.))),      // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });
    const auto z_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(1, random(-1., 1.))),      // y1
        GENERATE(take(1, random(-1., 1.))),      // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });
    const auto w_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(1, random(-1., 1.))),      // y1
        GENERATE(take(1, random(-1., 1.))),      // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });

    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(x_min, x_max) * y_poly.integral(y_min, y_max) *
                                                     z_poly.integral(z_min, z_max) * w_poly.integral(w_min, w_max);

    double integral{};
    integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], y_poly[0], y_poly[1], y_poly[2],
                   y_poly[3], z_poly[0], z_poly[1], z_poly[2], z_poly[3], w_poly[0], w_poly[1], w_poly[2], w_poly[3]);
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

TEST_CASE("Test 4D device integrals", "[double][cpu][integration][quadrature]")
{
  DiFfRG::Initialize();

  const double x_min = GENERATE(take(2, random(-2., -1.)));
  const double y_min = GENERATE(take(2, random(-2., -1.)));
  const double z_min = GENERATE(take(2, random(-2., -1.)));
  const double w_min = GENERATE(take(2, random(-2., -1.)));

  const double x_max = GENERATE(take(2, random(1., 2.)));
  const double y_max = GENERATE(take(2, random(1., 2.)));
  const double z_max = GENERATE(take(2, random(1., 2.)));
  const double w_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator4D<double, PolyIntegrand, CPU_exec> integrator(quadrature_provider, {32, 16, 8, 4},
                                                           {x_min, y_min, z_min, w_min}, {x_max, y_max, z_max, w_max});

  SECTION("Volume integral")
  {
    const double reference_integral = (x_max - x_min) * (y_max - y_min) * (z_max - z_min) * (w_max - w_min);

    double integral{};
    integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.);
    Kokkos::fence();

    if (!is_close(reference_integral, integral, 1e-9)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }

    constexpr double expected_precision = 1e-9;
    CHECK(integral == Catch::Approx(reference_integral).epsilon(expected_precision));
  }

  SECTION("Volume map")
  {
    const uint rsize = GENERATE(16, 32);
    std::vector<double> integral_view(rsize);
    LinearCoordinates1D<double> coordinates(rsize, 0., 0.);

    integrator.map(integral_view.data(), coordinates, 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.)
        .fence();

    for (uint i = 0; i < rsize; ++i)
      CHECK(is_close(integral_view[i],
                     coordinates.forward(i) + (x_max - x_min) * (y_max - y_min) * (z_max - z_min) * (w_max - w_min),
                     1e-6));
  };

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;
    const auto x_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(1, random(-1., 1.))),      // x1
        GENERATE(take(1, random(-1., 1.))),      // x2
        GENERATE(take(1, random(-1., 1.))),      // x3
    });
    const auto y_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(1, random(-1., 1.))),      // y1
        GENERATE(take(1, random(-1., 1.))),      // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });
    const auto z_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(1, random(-1., 1.))),      // y1
        GENERATE(take(1, random(-1., 1.))),      // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });
    const auto w_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(1, random(-1., 1.))),      // y1
        GENERATE(take(1, random(-1., 1.))),      // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });

    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(x_min, x_max) * y_poly.integral(y_min, y_max) *
                                                     z_poly.integral(z_min, z_max) * w_poly.integral(w_min, w_max);

    double integral{};
    integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], y_poly[0], y_poly[1], y_poly[2],
                   y_poly[3], z_poly[0], z_poly[1], z_poly[2], z_poly[3], w_poly[0], w_poly[1], w_poly[2], w_poly[3]);
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