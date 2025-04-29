#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/initialize.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration_kokkos/integrator_1D.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

template <typename view_type, typename integrator_type, typename... Args>
void map_integrators(const view_type integral_view, const integrator_type integrator, const Args &...args)
{
  const auto m_args = std::make_tuple(args...);

  using ExecutionSpace = typename integrator_type::execution_space;

  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ExecutionSpace>(integral_view.size(), Kokkos::AUTO), KOKKOS_LAMBDA(const TeamType &team) {
        const uint k = team.league_rank();
        // make subview
        auto subview = Kokkos::subview(integral_view, k);

        std::apply([&](const auto &...iargs) { integrator.get_nested(subview, team, iargs...); }, m_args);
      });
}

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto kernel(const double q, const double /*c*/, const double x0,
                                                         const double x1, const double x2, const double x3,
                                                         const double x4, const double x5)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q));
  }

  static __forceinline__ __host__ __device__ auto constant(const double c, const double /*x0*/, const double /*x1*/,
                                                           const double /*x2*/, const double /*x3*/,
                                                           const double /*x4*/, const double /*x5*/)
  {
    return c;
  }
};

TEST_CASE("Benchmark cpu momentum integrals", "[integration][quadrature integration]")
{
  DiFfRG::Initialize();

  const double x_min = GENERATE(take(1, random(-2., -1.)));
  const double x_max = GENERATE(take(1, random(1., 2.)));

  QuadratureProvider quadrature_provider;

  const double constant = GENERATE(take(1, random(-1., 1.)));
  const auto poly = Polynomial({
      GENERATE(take(1, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))), // x1
      GENERATE(take(1, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.))), // x3
      GENERATE(take(1, random(-1., 1.))), // x4
      GENERATE(take(1, random(-1., 1.))), // x5
  });

  constexpr_for<5, 8, 1>([&](auto j) {
    constexpr_for<6, 16, 3>([&](auto i) {
      constexpr uint rsize = powr<j>(2);
      constexpr uint isize = powr<i>(2);
      {
        Integrator1D<double, PolyIntegrand, CPU_exec> integrator(quadrature_provider, {isize}, {x_min}, {x_max});
        Kokkos::View<double *, CPU_memory> integral_view("cpu_integral_results", rsize);

        BENCHMARK_ADVANCED("host isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure([&] {
            for (uint k = 0; k < rsize; ++k) {
              // make subview
              auto subview = Kokkos::subview(integral_view, k);
              integrator.get(subview, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5]);
            }
            Kokkos::fence();
          });
        };
      }
      {
        Integrator1D<double, PolyIntegrand, GPU_exec> integrator(quadrature_provider, {isize}, {x_min}, {x_max});
        Kokkos::View<double *, GPU_memory> integral_view("cpu_integral_results", rsize);

        BENCHMARK_ADVANCED("device isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure([&] {
            for (uint k = 0; k < rsize; ++k) {
              // make subview
              auto subview = Kokkos::subview(integral_view, k);
              integrator.get(subview, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5]);
            }
            Kokkos::fence();
          });
        };
      }
      {
        Integrator1D<double, PolyIntegrand, CPU_exec> integrator(quadrature_provider, {isize}, {x_min}, {x_max});
        Kokkos::View<double *, CPU_memory> integral_view("cpu_integral_results", rsize);
        auto integral_host = Kokkos::create_mirror_view(integral_view);

        BENCHMARK_ADVANCED("host nested isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure([&] {
            map_integrators(integral_view, integrator, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5]);
            Kokkos::fence();
            Kokkos::deep_copy(integral_host, integral_view);
          });
        };

        // Check the results
        for (uint k = 0; k < rsize; ++k) {
          const double result = integral_host(k);
          const double reference_integral = constant + poly.integral(x_min, x_max);
          if (!is_close(reference_integral, result, 1e-6)) {
            std::cerr << "reference: " << reference_integral << "| integral: " << result
                      << "| relative error: " << abs(reference_integral - result) / abs(reference_integral)
                      << std::endl;
          }
          CHECK(is_close(reference_integral, result, 1e-6));
        }
      }
      {
        Integrator1D<double, PolyIntegrand, GPU_exec> integrator(quadrature_provider, {isize}, {x_min}, {x_max});
        Kokkos::View<double *, GPU_memory> integral_view("cpu_integral_results", rsize);
        auto integral_host = Kokkos::create_mirror_view(integral_view);

        BENCHMARK_ADVANCED("device nested isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure([&] {
            map_integrators(integral_view, integrator, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5]);
            Kokkos::fence();
            Kokkos::deep_copy(integral_host, integral_view);
          });
        };

        // Check the results
        for (uint k = 0; k < rsize; ++k) {
          const double result = integral_host(k);
          const double reference_integral = constant + poly.integral(x_min, x_max);
          if (!is_close(reference_integral, result, 1e-6)) {
            std::cerr << "reference: " << reference_integral << "| integral: " << result
                      << "| relative error: " << abs(reference_integral - result) / abs(reference_integral)
                      << std::endl;
          }
          CHECK(is_close(reference_integral, result, 1e-6));
        }
      }
    });
  });
}