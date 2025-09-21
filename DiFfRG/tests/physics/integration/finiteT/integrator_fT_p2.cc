#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/physics/regulators.hh>

using namespace DiFfRG;

#include "../boilerplate/poly_integrand.hh"

//--------------------------------------------
// Quadrature integration
TEMPLATE_TEST_CASE_SIG("Test finite T momentum integrals", "[integration][quadrature]", ((int dim), dim), (3), (4))
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

    const ctype T = GENERATE(1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1., 5., 10.);
    const ctype x_extent = GENERATE(take(1, random(1., 2.)));
    const uint size = GENERATE(32, 64);

    QuadratureProvider quadrature_provider;
    Integrator_fT_p2<dim, NT, PolyIntegrand<2, NT, -1>, ExecutionSpace> integrator(quadrature_provider, {size},
                                                                                   x_extent);

    const ctype k = GENERATE(take(1, random(0., 1.)));
    const ctype q_extent = std::sqrt(x_extent * powr<2>(k));

    SECTION("Volume integral (bosonic)")
    {
      const ctype val = GENERATE(1e-4, 1e-3, 1e-2, 1e-1, 1., 10.);
      integrator.set_T(T);
      integrator.set_k(k);
      integrator.set_typical_E(val);

      const NT reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                    / (std::tanh(val / (2. * T)) * 2. * val);         // sum

      NT integral{};
      integrator.get(integral, 0., 1., 0., 0., 0., powr<2>(val), 0., 1., 0.);

      constexpr ctype expected_precision = 5e-6;
      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "Failure for T = " << T << ", k = " << k << ", val = " << val << ", dim = " << dim << "\n"
                  << "reference: " << reference_integral << "| integral: " << integral
                  << "| relative error: " << rel_err << std::endl;
      }
      CHECK(rel_err < expected_precision);
    }
    SECTION("Volume map")
    {
      const ctype val = GENERATE(1e-3, 1e-1, 10.);
      integrator.set_T(T);
      integrator.set_k(k);
      integrator.set_typical_E(val);

      const NT reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                    / (std::tanh(val / (2. * T)) * 2. * val);         // sum

      const uint rsize = GENERATE(32, 64);
      std::vector<NT> integral_view(rsize);
      LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

      integrator.map(integral_view.data(), coordinates, 1., 0., 0., 0., powr<2>(val), 0., 1., 0.).fence();

      constexpr ctype expected_precision = 5e-6;
      for (uint i = 0; i < rsize; ++i) {
        const ctype rel_err = t_abs(coordinates.forward(i) + reference_integral - integral_view[i]) /
                              t_abs(coordinates.forward(i) + reference_integral);
        if (rel_err >= expected_precision) {
          std::cout << "reference: " << coordinates.forward(i) + reference_integral
                    << "| integral: " << integral_view[i] << "| relative error: "
                    << abs(coordinates.forward(i) + reference_integral - integral_view[i]) /
                           abs(coordinates.forward(i) + reference_integral)
                    << "| type: " << typeid(type).name() << "| i: " << i << std::endl;
        }
        CHECK(rel_err < expected_precision);
      }
    };
  };

  // Check on TBB
  SECTION("TBB") { check(TBB_exec(), (double)0); }
  // Check on Threads
  SECTION("Threads") { check(Threads_exec(), (double)0); }
  // Check on GPU
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}

template <typename NT> class BosonicIntegrand_an
{
public:
  using ctype = typename DiFfRG::get_type::ctype<NT>;
  static KOKKOS_INLINE_FUNCTION auto kernel(const ctype p, const NT T, const NT k)
  {
    using DiFfRG::powr;

    double val = powr<2>(p) + powr<2>(k) * exp(-p / k);
    return 1. / (std::tanh(val / (2. * T)) * 2. * val); // sum
  }

  static KOKKOS_INLINE_FUNCTION auto constant(const NT, const NT) { return 0; }
};

template <typename NT> class BosonicIntegrand
{
public:
  using ctype = typename DiFfRG::get_type::ctype<NT>;
  static KOKKOS_INLINE_FUNCTION auto kernel(const ctype p, const ctype p0, const NT T, const NT k)
  {
    using DiFfRG::powr;

    double val = powr<2>(p) + powr<2>(k) * exp(-p / k);
    return 1. / (powr<2>(val) + powr<2>(p0));
  }

  static KOKKOS_INLINE_FUNCTION auto constant(const NT, const NT) { return 0; }
};

TEST_CASE("Test against numerical matsubara sum", "[integration][quadrature]")
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

    const ctype T = GENERATE(1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1., 5., 10.);
    const ctype x_extent = GENERATE(take(3, random(1., 2.)));
    const uint size = GENERATE(16, 32);

    QuadratureProvider quadrature_provider;
    Integrator_fT_p2<4, NT, BosonicIntegrand<NT>, ExecutionSpace> integrator(quadrature_provider, {size}, x_extent);
    Integrator_p2<3, NT, BosonicIntegrand_an<NT>, ExecutionSpace> integrator_an(quadrature_provider, {size}, x_extent);

    const ctype k = GENERATE(take(1, random(0., 1.)));
    const ctype q_extent = std::sqrt(x_extent * powr<2>(k));

    SECTION("(bosonic)")
    {
      integrator.set_T(T);
      integrator.set_k(k);
      integrator.set_typical_E(k);
      integrator_an.set_k(k);

      NT integral_an{};
      integrator_an.get(integral_an, T, k);
      NT integral{};
      integrator.get(integral, T, k);

      constexpr ctype expected_precision = 5e-6;
      const ctype rel_err = t_abs(integral_an - integral) / t_abs(integral_an);
      if (rel_err >= expected_precision) {
        std::cerr << "Failure for T = " << T << ", k = " << k << "\n"
                  << "reference: " << integral_an << "| integral: " << integral << "| relative error: " << rel_err
                  << std::endl;
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

/*
TEST_CASE("Test integrator_fT_p2 bug", "[integration][quadrature]")
{
  using NT = double;
  using ctype = typename get_type::ctype<NT>;
  using ExecutionSpace = ExecutionSpaces::TBB_exec_space;
  using Regulator = DiFfRG::PolynomialExpRegulator<>;
  const int dim_fT = 4;
  const int dim = 3;

  DiFfRG::Init();
  const NT T = 1e-2;
  const NT k = 0.65;
  // const double x_extent = GENERATE(take(1, random(1., 2.)));
  const NT x_extent = 2.0;
  // const uint size = GENERATE(64, 128, 256);
  const size_t size = 256;

  QuadratureProvider quadrature_provider;
  Integrator_fT_p2<dim_fT, NT, quark_kernel<Regulator>, ExecutionSpace> integrator_fT(quadrature_provider, {size},
                                                                                      x_extent);
  Integrator_p2<dim, NT, quarkIntegrated_kernel<Regulator>, ExecutionSpace> integrator(quadrature_provider, {size},
                                                                                       x_extent);
  integrator.set_k(k);
  integrator_fT.set_k(k);

  integrator_fT.set_T(T);

  // const double q_extent = std::sqrt(x_extent * powr<2>(k));
  // const double mq2 = GENERATE(0.0, 0.5, 1.0);
  const NT h = 6.2;
  const NT sigma = 0.1;
  const double mq2 = powr<2>(h * sigma);

  NT integral_fT{};
  integrator_fT.get(integral_fT, k, T, mq2);
  NT integralIntegrated{};
  integrator.get(integralIntegrated, k, T, mq2);

  std::cout << "integral analytic matsubara sum: " << integralIntegrated
            << "| integral numeric matsubara sum: " << integral_fT << std::endl;

  constexpr ctype expected_precision = 1e-6;
  const ctype rel_err = abs(integral_fT - integralIntegrated) / abs(integralIntegrated);
  if (rel_err >= expected_precision) {
    std::cerr << "integral analytic matsubara sum: " << integralIntegrated
              << "| integral numeric matsubara sum: " << integral_fT << "| relative error: " << rel_err << std::endl;
  }
  CHECK(rel_err < expected_precision);
}*/