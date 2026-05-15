#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2.hh>

#include "../boilerplate/poly_integrand.hh"

using namespace DiFfRG;

namespace
{
  void check_complex_approx(const complex<double> actual, const complex<double> expected)
  {
    CHECK(real(actual) == Catch::Approx(real(expected)));
    CHECK(imag(actual) == Catch::Approx(imag(expected)));
  }
} // namespace

TEMPLATE_TEST_CASE_SIG("Test 1D momentum integrals", "[integration][quadrature]", ((int dim), dim), (2), (3), (4), (5))
{
  DiFfRG::Init();

  auto check = [](auto execution_space, auto type) {
    using NT = std::decay_t<decltype(type)>;
    using ctype = typename get_type::ctype<NT>;
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;

    using Kokkos::abs;
    auto t_abs = [](const auto val) {
      using type = std::decay_t<decltype(val)>;
      using Kokkos::abs;
      if constexpr (std::is_same_v<type, autodiff::real>)
        return abs(autodiff::val(val)) + abs(autodiff::grad(val));
      else if constexpr (std::is_same_v<type, cxreal>)
        return abs(autodiff::val(val)) + abs(autodiff::grad(val));
      else
        return abs(val);
    };

    const ctype x_extent = GENERATE(take(1, random(1., 2.)));
    const uint size = GENERATE(32, 64, 128, 256);

    QuadratureProvider quadrature_provider;
    Integrator_p2<dim, NT, PolyIntegrand<1, NT>, ExecutionSpace> integrator(quadrature_provider, {size}, x_extent);

    SECTION("Volume integral")
    {
      const ctype k = GENERATE(take(1, random(0., 1.)));
      const ctype q_extent = std::sqrt(x_extent * powr<2>(k));
      const NT reference_integral = V_d(dim, q_extent) / powr<dim>(2. * M_PI);

      integrator.set_k(k);

      NT integral{};
      integrator.get(integral, 0., 1., 0., 0., 0.);

      constexpr ctype expected_precision = 1e-14;
      const ctype rel_err = t_abs((integral - reference_integral) / reference_integral);
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
          dim == 1 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
          GENERATE(take(2, random(-1., 1.))),                 // x1
          GENERATE(take(2, random(-1., 1.))),                 // x2
          GENERATE(take(2, random(-1., 1.)))                  // x3
      });

      const ctype k = GENERATE(take(2, random(0., 1.)));
      const ctype q_extent = std::sqrt(x_extent * powr<2>(k));
      const NT constant = GENERATE(take(2, random(-1., 1.)));

      auto int_poly = x_poly;
      std::vector<ctype> coeff_integrand(dim, 0.);
      coeff_integrand[dim - 1] = 1.;
      int_poly *= Polynomial(coeff_integrand);
      const NT reference_integral = constant + S_d(dim) * int_poly.integral(0., q_extent) / powr<dim>(2. * M_PI);

      integrator.set_k(k);

      NT integral{};
      integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3]);

      constexpr ctype expected_precision = 1e-13;
      const ctype rel_err = t_abs((integral - reference_integral) / reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                  << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                  << " | expected precision: " << std::setw(12) << expected_precision << " | order: " << std::setw(5)
                  << size << " | dim: " << std::setw(3) << dim << std::endl;
      }
      CHECK(rel_err < expected_precision);
    }
  };

  // Check on TBB
  SECTION("TBB") { check(TBB_exec(), (double)0); }
  // Check on Threads
  SECTION("Threads") { check(Threads_exec(), (double)0); }
  // Check on GPU
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}

TEST_CASE("Integrator_p2 propagates second-order complex autodiff through TBB", "[integration][quadrature][autodiff]")
{
  DiFfRG::Init();

  using NT = cxReal<2, double>;
  using ctype = typename get_type::ctype<NT>;
  static_assert(std::is_same_v<ctype, double>);

  QuadratureProvider quadrature_provider;
  Integrator_p2<3, NT, ParameterizedPolynomialIntegrand<NT>, TBB_exec> integrator(quadrature_provider, {96}, 1.);
  Integrator_p2<3, NT, PolyIntegrand<1, NT>, TBB_exec> reference_integrator(quadrature_provider, {96}, 1.);
  integrator.set_k(1.);
  reference_integrator.set_k(1.);

  const complex<double> m0(0.7, 0.2);
  const NT m(std::array<complex<double>, 3>{m0, complex<double>(1., 0.), complex<double>(0., 0.)});

  NT integral{};
  integrator.get(integral, m);

  NT poly_integral{};
  reference_integrator.get(poly_integral, NT{}, m, m * m, m * m * m, NT{});

  const ctype prefactor = S_d_prec<ctype>(3) / powr<3>(2. * M_PI);
  const complex<double> expected_value = prefactor * (m0 / 3. + m0 * m0 / 4. + m0 * m0 * m0 / 5.);
  const complex<double> expected_first_derivative = prefactor * (1. / 3. + m0 / 2. + 3. * m0 * m0 / 5.);
  const complex<double> expected_second_derivative = prefactor * (1. / 2. + 6. * m0 / 5.);

  check_complex_approx(autodiff::val(integral), autodiff::val(poly_integral));
  check_complex_approx(autodiff::derivative<1>(integral), autodiff::derivative<1>(poly_integral));
  check_complex_approx(autodiff::derivative<2>(integral), autodiff::derivative<2>(poly_integral));

  check_complex_approx(autodiff::val(integral), expected_value);
  check_complex_approx(autodiff::derivative<1>(integral), expected_first_derivative);
  check_complex_approx(autodiff::derivative<2>(integral), expected_second_derivative);
}
