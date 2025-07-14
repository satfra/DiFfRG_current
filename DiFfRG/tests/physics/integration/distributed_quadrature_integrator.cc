#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "boilerplate/poly_integrand.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

TEST_CASE("Test distributed integration", "[integration][quadrature]")
{
  DiFfRG::Init();

  constexpr int dim = 2;

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
      else if constexpr (std::is_same_v<type, cxReal>)
        return abs(autodiff::val(val)) + abs(autodiff::grad(val));
      else
        return abs(val);
    };

    const int size = GENERATE(16);

    std::array<ctype, dim> ext_min;
    std::array<ctype, dim> ext_max;
    std::array<uint, dim> grid_size;
    std::array<QuadratureType, dim> quad_type;

    for (uint d = 0; d < dim; ++d) {
      ext_min[d] = GENERATE(take(1, random(-2., -1.)));
      ext_max[d] = GENERATE(take(1, random(1., 2.)));

      ext_min[d] = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, ext_min[d]);
      ext_max[d] = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, ext_max[d]);

      grid_size[d] = size;
      quad_type[d] = QuadratureType::legendre;
    }

    QuadratureProvider quadrature_provider;
    QuadratureIntegrator<dim, NT, PolyIntegrand<dim, NT>, ExecutionSpace> integrator(quadrature_provider, grid_size,
                                                                                     ext_min, ext_max, quad_type);
    SECTION("Volume map")
    {
      NT reference_integral = 1;
      for (uint d = 0; d < dim; ++d)
        reference_integral *= (ext_max[d] - ext_min[d]);

      std::array<NT, 4 * dim> coeffs{};
      for (uint d = 0; d < dim; ++d)
        coeffs[4 * d] = 1;

      const uint rsize = GENERATE(32);
      std::vector<NT> integral_view(rsize);
      LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

      if (DiFfRG::MPI::size(MPI_COMM_WORLD) > 1) {
        DiFfRG::NodeDistribution node_distribution(MPI_COMM_WORLD, {rsize / 2, rsize - rsize / 2}, {0, 1});
        DiFfRG::IntegrationLoadBalancer integrator_load_balancer(MPI_COMM_WORLD);

        integrator.set_node_distribution(node_distribution);
        std::apply([&](auto... coeffs) { integrator.map(integral_view.data(), coordinates, coeffs...).fence(); },
                   coeffs);
        DiFfRG::MPI::barrier(MPI_COMM_WORLD);
      } else {
        std::apply([&](auto... coeffs) { integrator.map(integral_view.data(), coordinates, coeffs...).fence(); },
                   coeffs);
      }

      if (DiFfRG::MPI::size(MPI_COMM_WORLD) > 1) {
        // locally ascertain that the values of the other rank are zero
        for (uint i = 0; i < rsize; ++i) {
          if (DiFfRG::MPI::rank(MPI_COMM_WORLD) == 0 && i >= rsize / 2) {
            CHECK(abs(integral_view[i]) < expected_precision<ctype>::value);
          } else if (DiFfRG::MPI::rank(MPI_COMM_WORLD) == 1 && i < rsize / 2) {
            CHECK(abs(integral_view[i]) < expected_precision<ctype>::value);
          }
        }
      }

      // check the values of the other rank
      for (uint i = 0; i < rsize; ++i) {
        integral_view[i] = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, integral_view[i]);
        const ctype rel_err = t_abs(coordinates.forward(i) + reference_integral - integral_view[i]) /
                              t_abs(coordinates.forward(i) + reference_integral);
        if (DiFfRG::MPI::rank(MPI_COMM_WORLD) == 0) {
          if (rel_err >= 10 * expected_precision<ctype>::value) {
            std::cout << "reference: " << coordinates.forward(i) + reference_integral
                      << "| integral: " << integral_view[i] << "| relative error: "
                      << abs(coordinates.forward(i) + reference_integral - integral_view[i]) /
                             abs(coordinates.forward(i) + reference_integral)
                      << "| type: " << typeid(type).name() << "| i: " << i << std::endl;
          }
          CHECK(rel_err < 10 * expected_precision<ctype>::value);
        }
      }
    };
  };
  SECTION("TBB integrals")
  {
    check(TBB_exec(), (double)0);
    check(TBB_exec(), (float)0);
  };
  SECTION("Threads integrals")
  {
    check(Threads_exec(), (double)0);
    check(Threads_exec(), (float)0);
  };
  SECTION("GPU integrals")
  {
    check(GPU_exec(), (double)0);
    check(GPU_exec(), (float)0);
  };
}