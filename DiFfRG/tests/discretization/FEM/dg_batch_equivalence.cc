#define CATCH_CONFIG_MAIN
#include "dg_test_model.hh"

#include <tbb/global_control.h>

#include <optional>

using namespace dg_test;

namespace
{
  template <typename A, typename B> void require_same_assembly(A &a, B &b)
  {
    const auto r_a = a.residual(a.u), r_b = b.residual(b.u);
    auto deviation = r_a;
    deviation -= r_b;
    REQUIRE(max_abs(r_b) > 0.);
    REQUIRE(max_abs(deviation) <= 1e-12 * max_abs(r_b));
    const auto [worst, scale] = matrix_deviation(b.jacobian(), a.jacobian());
    REQUIRE(scale > 0.);
    REQUIRE(worst <= 1e-12 * scale);
  }
} // namespace

TEST_CASE("A model's own evaluate_batch matches the per-point default under dDG", "[discretization][dg]")
{
  DiFfRG::Init();
  Setup<1, true, true, true, true> batched(2, "0:0.1:1");
  Setup<1, true, false, true, true> per_point(2, "0:0.1:1");
  require_same_assembly(batched, per_point);
}

TEST_CASE("The batched LLF numflux matches the per-point LLF numflux", "[discretization][dg]")
{
  DiFfRG::Init();
  Setup<1, true, false, true, true, NumFlux::llf> batched(2, "0:0.1:1");
  Setup<1, true, false, true, true, NumFlux::llf_per_point> per_point(2, "0:0.1:1");
  require_same_assembly(batched, per_point);
}

// Thousands of cells, so that the cells of one color really are assembled on many threads at once. The
// result must not depend on the thread count at all.
TEST_CASE("dDG assembly does not depend on the thread count", "[discretization][dg]")
{
  DiFfRG::Init();
  Setup<1, true, true, false, false> s(2, "0:0.00025:1");
  const auto r_all = s.residual(s.u);
  typename decltype(s)::SparseMatrixType J_all(s.assembler.get_sparsity_pattern_jacobian());
  J_all.copy_from(s.jacobian());
  std::optional<tbb::global_control> serial;
  serial.emplace(tbb::global_control::max_allowed_parallelism, 1);
  const auto r_one = s.residual(s.u);
  const auto &J_one = s.jacobian();
  serial.reset();

  auto deviation = r_all;
  deviation -= r_one;
  REQUIRE(max_abs(r_all) > 0.);
  REQUIRE(max_abs(deviation) == 0.);
  const auto [worst, scale] = matrix_deviation(J_all, J_one);
  REQUIRE(scale > 0.);
  REQUIRE(worst == 0.);
}
