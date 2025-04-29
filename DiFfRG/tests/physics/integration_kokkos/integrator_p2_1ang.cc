#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/initialize.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration_kokkos/integrator_p2_1ang.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static KOKKOS_INLINE_FUNCTION auto kernel(const double q, const double cos, const double /*c*/, const double x0,
                                            const double x1, const double x2, const double x3, const double x4,
                                            const double x5, const double cos_x0, const double cos_x1,
                                            const double cos_x2, const double cos_x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q)) *
           (cos_x0 + cos_x1 * powr<1>(cos) + cos_x2 * powr<2>(cos) + cos_x3 * powr<3>(cos));
  }

  static KOKKOS_INLINE_FUNCTION auto constant(const double c, const double /*x0*/, const double /*x1*/,
                                              const double /*x2*/, const double /*x3*/, const double /*x4*/,
                                              const double /*x5*/, const double /*cos_x0*/, const double /*cos_x1*/,
                                              const double /*cos_x2*/, const double /*cos_x3*/)
  {
    return c;
  }
};

TEMPLATE_TEST_CASE_SIG("Test momentum + angle integrals on host", "[double][cpu][integration][quadrature]",
                       ((int dim), dim), (2), (3), (4), (5))
{
  DiFfRG::Initialize();

  const double x_extent = GENERATE(take(1, random(1., 2.)));
  const uint size = GENERATE(16, 32, 64);

  QuadratureProvider quadrature_provider;
  Integrator_p2_1ang<dim, double, PolyIntegrand, CPU_exec> integrator(quadrature_provider, {size, 8}, x_extent);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double reference_integral = V_d(dim, q_extent) / powr<dim>(2. * M_PI);

    integrator.set_k(k);

    double integral{};
    integrator.get(integral, 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.);
    Kokkos::fence();

    const double expected_precision =
        max(1e-14,                                                          // machine precision
            1e-6                                                            // precision for worst quadrature
                / pow((double)size, 1.5)                                    // adjust for quadrature size
                * (dim % 2 == 1 ? pow(1e7, 1. / sqrt((double)dim)) : 1e-14) // adjust for odd dimensions
        );
    const double rel_err = abs((integral - reference_integral) / reference_integral);
    if (rel_err >= expected_precision) {
      std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                << " | expected precision: " << std::setw(12) << expected_precision << " | order: " << std::setw(5)
                << size << " | dim: " << std::setw(3) << dim << std::endl;
    }
    CHECK(rel_err < expected_precision);
  }
  SECTION("Random polynomials")
  {
    const auto x_poly = Polynomial({
        dim == 1 ? 0. : GENERATE(take(1, random(-1., 1.))), // x0
        GENERATE(take(2, random(-1., 1.))),                 // x1
        GENERATE(take(2, random(-1., 1.))),                 // x2
        GENERATE(take(2, random(-1., 1.))),                 // x3
        GENERATE(take(1, random(-1., 1.))),                 // x4
        GENERATE(take(1, random(-1., 1.))),                 // x5
    });
    const auto cos_poly = Polynomial({
        GENERATE(take(2, random(-1., 1.))), // x0
        GENERATE(take(2, random(-1., 1.))), // x1
        GENERATE(take(1, random(-1., 1.))), // x2
        GENERATE(take(1, random(-1., 1.)))  // x3
    });

    const double k = GENERATE(take(2, random(0., 1.)));
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double constant = GENERATE(take(2, random(-1., 1.)));

    auto int_poly = x_poly;
    std::vector<double> coeff_integrand(dim, 0.);
    coeff_integrand[dim - 1] = 1.;
    int_poly *= Polynomial(coeff_integrand);
    const double reference_integral =
        constant + S_d(dim) / 2. * int_poly.integral(0., q_extent) / powr<dim>(2. * M_PI) * cos_poly.integral(-1., 1.);

    integrator.set_k(k);

    double integral{};
    integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], x_poly[4], x_poly[5], cos_poly[0],
                   cos_poly[1], cos_poly[2], cos_poly[3]);
    Kokkos::fence();

    const double expected_precision = max(1e-14, 1e-1 / pow((double)size, 2));
    const double rel_err = abs((integral - reference_integral) / reference_integral);
    if (rel_err >= expected_precision) {
      std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                << " | expected precision: " << std::setw(12) << expected_precision << " | order: " << std::setw(5)
                << size << " | dim: " << std::setw(3) << dim << std::endl;
    }
    CHECK(rel_err < expected_precision);
  }
}

TEMPLATE_TEST_CASE_SIG("Test momentum + angle integrals on device", "[double][cpu][integration][quadrature]",
                       ((int dim), dim), (2), (3), (4), (5))
{
  DiFfRG::Initialize();

  const double x_extent = GENERATE(take(1, random(1., 2.)));
  const uint size = GENERATE(16, 32, 64);

  QuadratureProvider quadrature_provider;
  Integrator_p2_1ang<dim, double, PolyIntegrand, GPU_exec> integrator(quadrature_provider, {size, 8}, x_extent);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double reference_integral = V_d(dim, q_extent) / powr<dim>(2. * M_PI);

    integrator.set_k(k);

    double integral{};
    integrator.get(integral, 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.);
    Kokkos::fence();

    const double expected_precision =
        max(1e-14,                                                          // machine precision
            1e-6                                                            // precision for worst quadrature
                / pow((double)size, 1.5)                                    // adjust for quadrature size
                * (dim % 2 == 1 ? pow(1e7, 1. / sqrt((double)dim)) : 1e-14) // adjust for odd dimensions
        );
    const double rel_err = abs((integral - reference_integral) / reference_integral);
    if (rel_err >= expected_precision) {
      std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                << " | expected precision: " << std::setw(12) << expected_precision << " | order: " << std::setw(5)
                << size << " | dim: " << std::setw(3) << dim << std::endl;
    }
    CHECK(rel_err < expected_precision);
  }
  SECTION("Random polynomials")
  {
    const auto x_poly = Polynomial({
        dim == 1 ? 0. : GENERATE(take(1, random(-1., 1.))), // x0
        GENERATE(take(2, random(-1., 1.))),                 // x1
        GENERATE(take(2, random(-1., 1.))),                 // x2
        GENERATE(take(2, random(-1., 1.))),                 // x3
        GENERATE(take(1, random(-1., 1.))),                 // x4
        GENERATE(take(1, random(-1., 1.))),                 // x5
    });
    const auto cos_poly = Polynomial({
        GENERATE(take(2, random(-1., 1.))), // x0
        GENERATE(take(2, random(-1., 1.))), // x1
        GENERATE(take(1, random(-1., 1.))), // x2
        GENERATE(take(1, random(-1., 1.)))  // x3
    });

    const double k = GENERATE(take(2, random(0., 1.)));
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double constant = GENERATE(take(2, random(-1., 1.)));

    auto int_poly = x_poly;
    std::vector<double> coeff_integrand(dim, 0.);
    coeff_integrand[dim - 1] = 1.;
    int_poly *= Polynomial(coeff_integrand);
    const double reference_integral =
        constant + S_d(dim) / 2. * int_poly.integral(0., q_extent) / powr<dim>(2. * M_PI) * cos_poly.integral(-1., 1.);

    integrator.set_k(k);

    double integral{};
    integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], x_poly[4], x_poly[5], cos_poly[0],
                   cos_poly[1], cos_poly[2], cos_poly[3]);
    Kokkos::fence();

    const double expected_precision = max(1e-14, 1e-1 / pow((double)size, 2));
    const double rel_err = abs((integral - reference_integral) / reference_integral);
    if (rel_err >= expected_precision) {
      std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                << " | expected precision: " << std::setw(12) << expected_precision << " | order: " << std::setw(5)
                << size << " | dim: " << std::setw(3) << dim << std::endl;
    }
    CHECK(rel_err < expected_precision);
  }
}