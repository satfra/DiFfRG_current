#include <Kokkos_Core.hpp>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "../boilerplate/poly_integrand.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat_2D.hh>

using namespace DiFfRG;

TEMPLATE_TEST_CASE("Test 2D lattice integrals on host", "[lattice][double][float][complex][autodiff]", double, float,
                   complex<double>, complex<float>, autodiff::real)
{
  using T = TestType;

  DiFfRG::Init();

  using ctype = typename get_type::ctype<T>;

  const ctype a0 = GENERATE(take(2, random(0.01, 1.)));
  const ctype a1 = GENERATE(take(2, random(0.01, 1.)));
  const uint size0 = GENERATE(16, 32, 64, 128);
  const uint size1 = GENERATE(16, 32, 64, 128);
  const bool q0_symmetric = GENERATE(false, true);

  IntegratorLat2D<T, PolyIntegrand<2, T>, Threads_exec> integrator({{size0, size1}}, {{a0, a1}}, q0_symmetric);

  SECTION("Volume integral")
  {
    const ctype reference_integral = 1. / a0 / a1;

    T integral{};
    integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0.);

    const ctype expected_precision = std::numeric_limits<ctype>::epsilon() * size0 * size1;
    if (!is_close(reference_integral, integral, expected_precision)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, expected_precision));
  }
}