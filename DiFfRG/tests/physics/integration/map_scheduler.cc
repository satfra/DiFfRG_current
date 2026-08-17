#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

#include <cstring>
#include <numeric>
#include <vector>

using namespace DiFfRG;

/**
 * Everything MapScheduler does apart from the single MPI_Allgatherv -- choosing the split width,
 * picking owners, the byte accounting, the packing and the landing -- is a pure function of the
 * sequence of map() calls. That makes it testable at any rank count in a single process, which
 * matters: an MPI build needs an MPI deal.II underneath it, so a plain build would otherwise leave
 * all of this code completely unexercised.
 *
 * The subclass below plays the part of the network. Everything it drives (prepare_batch,
 * finish_batch, and the buffers between them) is the production code path, not a re-implementation.
 */
namespace
{
  class SimulatedRank : public MapScheduler
  {
  public:
    SimulatedRank(uint rank, uint n_ranks) : MapScheduler(rank, n_ranks) {}

    using MapScheduler::finish_batch;
    using MapScheduler::prepare_batch;

    const std::vector<char> &send_buffer() const { return m_send; }
    const std::vector<int> &counts() const { return m_counts; }
    const std::vector<int> &displs() const { return m_displs; }

    /// Copy every peer's packed block into this rank's receive buffer, i.e. do by hand what
    /// MPI_Allgatherv would have done.
    void receive_from(const std::vector<SimulatedRank *> &all)
    {
      m_recv.assign(static_cast<size_t>(m_displs.back()) + static_cast<size_t>(m_counts.back()), '\0');
      for (uint p = 0; p < all.size(); ++p)
        if (m_counts[p] > 0)
          std::memcpy(m_recv.data() + m_displs[p], all[p]->send_buffer().data(), static_cast<size_t>(m_counts[p]));
    }
  };

  /// One simulated batch: every rank schedules the same sequence of maps, then they exchange.
  struct Cluster {
    std::vector<std::unique_ptr<SimulatedRank>> ranks;
    std::vector<SimulatedRank *> raw;

    explicit Cluster(uint n)
    {
      for (uint p = 0; p < n; ++p) {
        ranks.push_back(std::make_unique<SimulatedRank>(p, n));
        ranks.back()->set_quantum(5e4);
      }
      for (auto &r : ranks)
        raw.push_back(r.get());
    }

    void exchange()
    {
      for (auto *r : raw)
        REQUIRE(r->prepare_batch());
      for (auto *r : raw)
        r->receive_from(raw);
      for (auto *r : raw)
        r->finish_batch();
    }
  };
} // namespace

TEST_CASE("Slices tile the grid exactly, at every rank count", "[integration][mpi][scheduler]")
{
  const uint n_ranks = GENERATE(1u, 2u, 3u, 4u, 5u, 7u, 8u, 12u, 16u);
  // 64 is the QCD_Nf2 external momentum grid; 3 and 5 are deliberately awkward divisors.
  const size_t G = GENERATE(size_t(1), size_t(3), size_t(5), size_t(64), size_t(100));

  Cluster cluster(n_ranks);
  std::vector<double> sink(G, 0.);

  std::vector<size_t> coverage(G, 0);
  for (uint p = 0; p < n_ranks; ++p) {
    const MapSlice s = cluster.raw[p]->schedule(0, sink.data(), sizeof(double), G, 6912, true);
    REQUIRE(s.offset + s.count <= G);
    for (size_t i = s.offset; i < s.offset + s.count; ++i)
      coverage[i] += 1;
  }

  // Every grid point computed exactly once: no gaps (wrong answers) and no duplicates (wasted GPU
  // time, and two ranks writing the same destination).
  for (size_t i = 0; i < G; ++i)
    CHECK(coverage[i] == 1);
}

TEST_CASE("The split width follows the cost score", "[integration][mpi][scheduler]")
{
  Cluster cluster(16);
  std::vector<double> sink(64, 0.);

  auto owners_of = [&](const size_t Q, const bool splittable) {
    size_t n_owners = 0;
    for (uint p = 0; p < 16; ++p)
      if (cluster.raw[p]->schedule(0, sink.data(), sizeof(double), 64, Q, splittable).count > 0) ++n_owners;
    return n_owners;
  };

  // The three flow classes in QCD_Nf2 at x_order=32, cos*=6, phi=6, against a 5e4 quantum.
  SECTION("p2_1ang (Q = 192, S = 12288) stays on one rank") { CHECK(owners_of(192, true) == 1); }
  SECTION("4D_2ang (Q = 1152, S = 73728) stays on one rank") { CHECK(owners_of(1152, true) == 1); }
  SECTION("4D_3ang (Q = 6912, S = 442368) spreads over 8") { CHECK(owners_of(6912, true) == 8); }
  SECTION("a multi-dimensional grid is never split") { CHECK(owners_of(6912, false) == 1); }
}

TEST_CASE("Cheap flows spread across ranks instead of being chopped up", "[integration][mpi][scheduler]")
{
  // Four cheap flows and four ranks: the point of assigning a cheap flow whole is that they end up
  // on four different ranks, one each, rather than every rank doing a quarter of every flow.
  Cluster cluster(4);
  std::vector<std::vector<double>> sinks(4, std::vector<double>(64, 0.));

  std::vector<size_t> per_rank(4, 0);
  for (size_t flow = 0; flow < 4; ++flow)
    for (uint p = 0; p < 4; ++p)
      if (cluster.raw[p]->schedule(flow, sinks[flow].data(), sizeof(double), 64, 192, true).count > 0) per_rank[p] += 1;

  for (uint p = 0; p < 4; ++p)
    CHECK(per_rank[p] == 1);
}

TEST_CASE("A batch exchange reassembles the full result on every rank", "[integration][mpi][scheduler]")
{
  const uint n_ranks = GENERATE(2u, 3u, 4u, 8u);
  Cluster cluster(n_ranks);

  // Three destinations of different width and cost, as in a real dt_variables() block.
  constexpr size_t n_flows = 3;
  const size_t G[n_flows] = {64, 64, 37};
  const size_t Q[n_flows] = {6912, 192, 6912};

  // The truth each rank should end up holding.
  std::vector<std::vector<double>> truth(n_flows);
  for (size_t f = 0; f < n_flows; ++f) {
    truth[f].resize(G[f]);
    for (size_t i = 0; i < G[f]; ++i)
      truth[f][i] = 1000. * double(f) + double(i);
  }

  // Per rank, per flow: the destination buffer it actually holds.
  std::vector<std::vector<std::vector<double>>> dest(n_ranks);
  for (uint p = 0; p < n_ranks; ++p) {
    dest[p].resize(n_flows);
    for (size_t f = 0; f < n_flows; ++f)
      dest[p][f].assign(G[f], -1.);
  }

  for (size_t f = 0; f < n_flows; ++f)
    for (uint p = 0; p < n_ranks; ++p) {
      const MapSlice s = cluster.raw[p]->schedule(f, dest[p][f].data(), sizeof(double), G[f], Q[f], true);
      // Only the owner computes -- exactly what map() does when it skips a non-owner.
      for (size_t i = s.offset; i < s.offset + s.count; ++i)
        dest[p][f][i] = truth[f][i];
    }

  cluster.exchange();

  for (uint p = 0; p < n_ranks; ++p)
    for (size_t f = 0; f < n_flows; ++f) {
      INFO("rank " << p << ", flow " << f);
      // Bitwise: the whole design rests on the distributed result being the serial result exactly.
      CHECK(std::memcmp(dest[p][f].data(), truth[f].data(), G[f] * sizeof(double)) == 0);
    }
}

TEST_CASE("Every rank derives the same plan", "[integration][mpi][scheduler]")
{
  Cluster cluster(5);
  std::vector<double> sink(64, 0.);

  // A mixed batch, scheduled in the same order on every rank.
  const size_t Qs[] = {6912, 192, 6912, 1152, 192, 6912};
  for (size_t f = 0; f < 6; ++f)
    for (uint p = 0; p < 5; ++p)
      cluster.raw[p]->schedule(f, sink.data(), sizeof(double), 64, Qs[f], true);

  for (auto *r : cluster.raw)
    REQUIRE(r->prepare_batch());

  // The byte accounting is what the Allgatherv is parameterised with, so agreement on it is
  // exactly the condition under which the collective is well formed.
  for (uint p = 1; p < 5; ++p) {
    CHECK(cluster.raw[p]->counts() == cluster.raw[0]->counts());
    CHECK(cluster.raw[p]->displs() == cluster.raw[0]->displs());
  }

  const auto &counts = cluster.raw[0]->counts();
  const int total = std::accumulate(counts.begin(), counts.end(), 0);
  CHECK(total == static_cast<int>(6 * 64 * sizeof(double)));

  for (auto *r : cluster.raw)
    r->receive_from(cluster.raw);
  for (auto *r : cluster.raw)
    r->finish_batch();
}
