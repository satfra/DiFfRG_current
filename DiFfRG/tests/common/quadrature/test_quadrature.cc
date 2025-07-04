#include "DiFfRG/common/kokkos.hh"
#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/quadrature/gauss_legendre.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>

using namespace DiFfRG;

TEMPLATE_TEST_CASE_SIG("Test correct creation of quadrature rules at example of gauss-legendre", "[double][quadrature]",
                       ((size_t order), order), (4), (8), (16), (32), (64), (128))
{
  const GLQuadrature<order, double> gl;

  std::vector<double> a(order, 0);
  std::vector<double> b(order, 0);

  // Jacobi matrix for Gauss-Legendre quadrature
  const double mu0 = 2.0;
  for (size_t i = 1; i <= order; ++i) {
    double abi = i;
    double abj = 2 * i;
    b[i - 1] = abi * abi / (abj * abj - 1.);
  }

  std::vector<double> x(order, 0);
  std::vector<double> w(order, 0);

  make_quadrature(a, b, mu0, x, w);

  REQUIRE(x[order - 1] == Catch::Approx(2. * gl.x[order - 1] - 1.));
  REQUIRE(w[order - 1] == Catch::Approx(2. * gl.w[order - 1]));
}

TEMPLATE_TEST_CASE_SIG("Test the quadrature class at the example of Gauss-Legendre", "[double][quadrature]",
                       ((size_t order), order), (4), (8), (16), (32), (64), (128))
{
  DiFfRG::Init();

  const Quadrature<double> q(order, QuadratureType::legendre);

  REQUIRE(q.nodes<CPU_memory>().size() == order);
  REQUIRE(q.weights<CPU_memory>().size() == order);

  REQUIRE(q.nodes<GPU_memory>().size() == order);
  REQUIRE(q.weights<GPU_memory>().size() == order);

  const GLQuadrature<order, double> gl;

  for (size_t i = 0; i < order; ++i) {
    REQUIRE(q.nodes<CPU_memory>()[i] == Catch::Approx(gl.x[i]));
    REQUIRE(q.weights<CPU_memory>()[i] == Catch::Approx(gl.w[i]));
  }

  auto gpu_nodes = Kokkos::create_mirror_view(q.nodes<GPU_memory>());
  auto gpu_weights = Kokkos::create_mirror_view(q.weights<GPU_memory>());
  Kokkos::deep_copy(gpu_nodes, q.nodes<GPU_memory>());
  Kokkos::deep_copy(gpu_weights, q.weights<GPU_memory>());

  for (size_t i = 0; i < order; ++i) {
    REQUIRE(gpu_nodes[i] == Catch::Approx(gl.x[i]));
    REQUIRE(gpu_weights[i] == Catch::Approx(gl.w[i]));
  }
}