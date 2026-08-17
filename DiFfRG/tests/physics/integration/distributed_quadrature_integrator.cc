#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "boilerplate/poly_integrand.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/map_completion.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <cstring>

using namespace DiFfRG;

/**
 * The contract of the MPI path is *bitwise* equality with a serial run, at any rank count. That is
 * not a stylistic preference: the flow kernels feed a stiff DAE, where a last-bit change in a
 * residual changes the accepted step sequence and therefore the whole trajectory. So this test
 * compares raw bit patterns, never a tolerance.
 *
 * `map()` goes through MapScheduler, `map_dist()` does not, which gives us the serial reference
 * inside the same process and the same execution space.
 */
namespace
{
  template <typename T> bool bitwise_equal(const std::vector<T> &a, const std::vector<T> &b)
  {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
  }
} // namespace

TEST_CASE("Test distributed integration", "[integration][quadrature][mpi]")
{
  DiFfRG::Init();

  constexpr int dim = 2;

  const uint n_ranks = DiFfRG::MPI::size(MPI_COMM_WORLD);
  const uint my_rank = DiFfRG::MPI::rank(MPI_COMM_WORLD);

  auto check = [&](auto execution_space, auto type) {
    using NT = std::decay_t<decltype(type)>;
    using ctype = typename get_type::ctype<NT>;
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;

    constexpr size_t size = 16;

    std::array<ctype, dim> ext_min;
    std::array<ctype, dim> ext_max;
    std::array<size_t, dim> grid_size;
    std::array<QuadratureType, dim> quad_type;

    // Every rank must integrate the *same* thing, so no randomness here: a random extent drawn per
    // rank would make the ranks disagree and the comparison meaningless.
    for (uint d = 0; d < dim; ++d) {
      ext_min[d] = ctype(-1.5);
      ext_max[d] = ctype(1.25);
      grid_size[d] = size;
      quad_type[d] = QuadratureType::legendre;
    }

    QuadratureProvider quadrature_provider;
    QuadratureIntegrator<dim, NT, PolyIntegrand<dim, NT>, ExecutionSpace> integrator(quadrature_provider, grid_size,
                                                                                     ext_min, ext_max, quad_type);

    std::array<NT, 4 * dim> coeffs{};
    for (uint d = 0; d < dim; ++d)
      coeffs[4 * d] = 1;

    constexpr uint rsize = 32;
    LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

    // Serial reference, in the same process and the same execution space: the overloads that take
    // an explicit destination without going through MapScheduler.
    std::vector<NT> reference(rsize, NT(0));
    if constexpr (std::is_same_v<ExecutionSpace, TBB_exec>) {
      auto space = ExecutionSpace();
      std::apply([&](auto... c) { integrator.map(space, reference.data(), coordinates, c...); }, coeffs);
    } else {
      std::apply([&](auto... c) { integrator.map_dist(reference.data(), coordinates, c...).fence(); }, coeffs);
    }
    flush_maps();

    // A quantum this small forces the scheduler to split even this toy grid, so the test exercises
    // the partitioning rather than falling back to "one rank owns everything".
    MapScheduler::instance().set_quantum(1.);

    std::vector<NT> distributed(rsize, NT(0));
    std::apply([&](auto... c) { integrator.map(distributed.data(), coordinates, c...).fence(); }, coeffs);
    flush_maps();

    INFO("rank " << my_rank << " of " << n_ranks);
    CHECK(bitwise_equal(reference, distributed));

    // Sanity: the values are actually the integral, not zeros that happen to match.
    NT volume = 1;
    for (uint d = 0; d < dim; ++d)
      volume *= (ext_max[d] - ext_min[d]);
    for (uint i = 0; i < rsize; ++i) {
      const auto expected = coordinates.forward(i) + volume;
      using Kokkos::abs;
      CHECK(abs(expected - distributed[i]) <= 10 * expected_precision<ctype>::value * abs(expected));
    }
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

TEST_CASE("Test the map schedule itself", "[integration][quadrature][mpi]")
{
  DiFfRG::Init();

  auto &scheduler = MapScheduler::instance();
  const uint n_ranks = scheduler.n_ranks();

  SECTION("A cheap flow is assigned whole, not chopped into slivers")
  {
    scheduler.set_quantum(5e4);

    // G * Q = 64 * 192, well under the quantum: this must land on a single rank.
    std::vector<double> sink(64, 0.);
    const MapSlice slice = scheduler.schedule(1000, sink.data(), sizeof(double), 64, 192, true);
    CHECK((slice.count == 0 || slice.count == 64));

    // Exactly one rank may own it.
    const int owners = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, slice.count > 0 ? 1 : 0);
    CHECK(owners == 1);

    flush_maps(); // completes (and clears) the plan
  }

  SECTION("An expensive flow is spread, and the slices tile the grid exactly")
  {
    scheduler.set_quantum(5e4);

    // G * Q = 64 * 6912 = 442368, i.e. r = 8 before the rank-count clamp.
    std::vector<double> sink(64, 0.);
    const MapSlice slice = scheduler.schedule(1001, sink.data(), sizeof(double), 64, 6912, true);

    const int covered = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, static_cast<int>(slice.count));
    CHECK(covered == 64);

    const int owners = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, slice.count > 0 ? 1 : 0);
    CHECK(owners == static_cast<int>(std::min<uint>(n_ranks, 8)));

    flush_maps();
  }

  SECTION("A multi-dimensional coordinate grid is never split")
  {
    scheduler.set_quantum(1.);

    std::vector<double> sink(64, 0.);
    const MapSlice slice = scheduler.schedule(1002, sink.data(), sizeof(double), 64, 6912, false);
    CHECK((slice.count == 0 || slice.count == 64));

    flush_maps();
  }
}
