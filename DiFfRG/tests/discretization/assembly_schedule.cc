#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/common/assembly_schedule.hh>

#include <limits>

using namespace DiFfRG;

namespace
{
  AssemblySchedule sched(const uint cells, const uint threads, const double cost)
  {
    return make_assembly_schedule(cells, threads, cost);
  }
} // namespace

TEST_CASE("A schedule is always usable by mesh_loop", "[discretization][assembly_schedule]")
{
  // deal.II only guards these with Assert, so in Release a zero would reach tbb::parallel_pipeline
  // as undefined behaviour instead of a diagnostic. Every degenerate input has to be caught here.
  for (const uint cells : {0u, 1u, 2u, 7u, 1000u, 1000000u})
    for (const uint threads : {1u, 2u, 8u, 64u})
      for (const double cost : {1.0, 250.0, 3000.0, 1e9}) {
        const auto s = make_assembly_schedule(cells, threads, cost);
        INFO("cells=" << cells << " threads=" << threads << " cost=" << cost);
        REQUIRE(s.queue_length >= 1);
        REQUIRE(s.chunk_size >= 1);
      }

  // Including through the override path, which is the one place a user-supplied 0 can enter.
  AssemblyScheduleOverrides zeroes;
  zeroes.queue_length = 0;
  zeroes.chunk_size = 0;
  const auto s = make_assembly_schedule(100, 8, 250.0, zeroes);
  REQUIRE(s.queue_length >= 1);
  REQUIRE(s.chunk_size >= 1);
}

TEST_CASE("The pipeline leaves threads free for the nested integrators", "[discretization][assembly_schedule]")
{
  // A cell worker that calls the momentum integrals spawns its own TBB work in the same arena. If
  // the pipeline holds every thread, that nesting has nothing to run on.
  for (const uint threads : {2u, 4u, 8u, 16u, 64u}) {
    const auto s = sched(1000000, threads, assembly_cost::momentum_integral);
    INFO("threads=" << threads);
    REQUIRE(s.queue_length < threads);
  }
  // A single-threaded budget is the one case where "below the thread count" cannot hold.
  REQUIRE(sched(1000000, 1, assembly_cost::momentum_integral).queue_length == 1);
}

TEST_CASE("Chunk size falls as the cell gets more expensive", "[discretization][assembly_schedule]")
{
  // The point of deriving from a cost rather than a class: the answer has to vary smoothly, so
  // that a loop between the reference points gets a schedule between theirs.
  const uint cells = 100000;
  const auto cheap = sched(cells, 8, assembly_cost::local_fe).chunk_size;
  const auto middle = sched(cells, 8, assembly_cost::algebraic).chunk_size;
  const auto integral = sched(cells, 8, assembly_cost::momentum_integral).chunk_size;

  REQUIRE(cheap > middle);
  REQUIRE(middle > integral);
  REQUIRE(integral >= 1);

  // and monotone in between, not just at the three named points
  uint previous = std::numeric_limits<uint>::max();
  for (double cost = 100.; cost < 1e5; cost *= 1.5) {
    const auto chunk = sched(cells, 8, cost).chunk_size;
    REQUIRE(chunk <= previous);
    previous = chunk;
  }
}

TEST_CASE("A mesh too small to fill the pipeline loses workers, not chunk size",
          "[discretization][assembly_schedule]")
{
  // Measured: cheap loops on a small mesh are assembled fastest by a single worker, and only past
  // a few thousand cells does the full budget pay off. Shrinking the chunk instead would put the
  // pipeline overhead back in.
  REQUIRE(sched(8, 8, assembly_cost::local_fe).queue_length == 1);
  REQUIRE(sched(128, 8, assembly_cost::local_fe).queue_length == 1);
  REQUIRE(sched(100000, 8, assembly_cost::local_fe).queue_length > 1);

  uint previous = 0;
  for (const uint cells : {1u, 10u, 100u, 1000u, 10000u, 100000u}) {
    const auto q = sched(cells, 8, assembly_cost::local_fe).queue_length;
    INFO("cells=" << cells);
    REQUIRE(q >= previous);
    previous = q;
  }
}

TEST_CASE("Workers never outnumber the chunks there are to work on", "[discretization][assembly_schedule]")
{
  // Few cells, each enormously expensive: the work would pay for far more workers than the mesh
  // can be split into. Every worker past the chunk count only allocates ScratchData and idles.
  const uint cells = 7;
  const auto s = sched(cells, 64, 1.0e9);
  REQUIRE(s.chunk_size == 1);
  REQUIRE(s.queue_length <= cells);
}

TEST_CASE("Chunks are never allocated past the end of the mesh", "[discretization][assembly_schedule]")
{
  // mesh_loop reserves queue_length * chunk_size CopyData objects up front, whether or not there
  // are cells to put in them.
  for (const uint cells : {1u, 3u, 17u, 64u})
    REQUIRE(sched(cells, 8, assembly_cost::local_fe).chunk_size <= cells);
}

TEST_CASE("An explicit override wins over the derived schedule", "[discretization][assembly_schedule]")
{
  ConfigTree config = json::value({{"discretization", {{"mesh_workers", 13}, {"batch_size", 7}}}});
  const auto overrides = AssemblyScheduleOverrides::from_config(config);

  const auto s = make_assembly_schedule(100000, 8, assembly_cost::local_fe, overrides);
  REQUIRE(s.queue_length == 13);
  REQUIRE(s.chunk_size == 7);

  // Absent keys must not be invented, or every run would look like an override.
  ConfigTree bare = json::value({{"discretization", {{"fe_order", 3}}}});
  const auto none = AssemblyScheduleOverrides::from_config(bare);
  REQUIRE_FALSE(none.queue_length.has_value());
  REQUIRE_FALSE(none.chunk_size.has_value());
}
