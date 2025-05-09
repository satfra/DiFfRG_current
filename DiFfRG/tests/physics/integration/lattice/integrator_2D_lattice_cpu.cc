#include "autodiff/forward/real/real.hpp"
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration_lattice/integrator_2D_lattice_cpu.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto kernel(const double ix, const double iy, const double /*k*/,
                                                         const double /*c*/, const double x0, const double x1,
                                                         const double x2, const double x3, const double y0,
                                                         const double y1, const double y2, const double y3)

  {
    return (x0 + x1 * powr<1>(ix) + x2 * powr<2>(ix) + x3 * powr<3>(ix)) +
           (y0 + y1 * powr<1>(iy) + y2 * powr<2>(iy) + y3 * powr<3>(iy));
  }

  static __forceinline__ __host__ __device__ auto
  constant(const double /*k*/, const double c, const double /*x0*/, const double /*x1*/, const double /*x2*/,
           const double /*x3*/, const double /*y0*/, const double /*y1*/, const double /*y2*/, const double /*y3*/)

  {
    return c;
  }
};

TEST_CASE("Test 2d cartesian cpu momentum integrals", "[double][cpu][integration][quadrature]")
{
  QuadratureProvider quadrature_provider;
  const uint N = GENERATE(8, 16, 32, 64, 128);

  Integrator2DCartesianTBB<double, PolyIntegrand> integrator(quadrature_provider, {{N, N}});

  SECTION("Volume integral")
  {
    const double reference_integral = powr<2>(N);

    const double integral = integrator.request(1, 0., 1., 0., 0., 0., 0., 0., 0., 0.).get();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
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
    const auto y_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(take_n, random(-1., 1.))), // x2
        GENERATE(take(take_n, random(-1., 1.))), // x3
    });

    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral =
        constant + powr<2>(N) *
                       (12. * x_poly[0] + 6. * (-1. + N) * x_poly[1] +
                        2. * (x_poly[2] + 6. * y_poly[0] - 3. * y_poly[1] + y_poly[2]) +
                        N * ((-6. + 4. * N) * x_poly[2] + 3. * powr<2>(-1. + N) * x_poly[3] + 6. * y_poly[1] -
                             6. * y_poly[2] + 4. * N * y_poly[2] + 3. * powr<2>(-1. + N) * y_poly[3])) /
                       12.;

    const double integral = integrator.get(1, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], y_poly[0],
                                           y_poly[1], y_poly[2], y_poly[3]);

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}