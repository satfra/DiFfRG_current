#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>

using namespace DiFfRG;

void method1(Catch::Benchmark::Chronometer &);
void method2(Catch::Benchmark::Chronometer &);
void method3(Catch::Benchmark::Chronometer &);
void method4(Catch::Benchmark::Chronometer &);

TEST_CASE("Benchmark different ND quadrature integrals", "[integration][quadrature]")
{
  DiFfRG::Init();
  BENCHMARK_ADVANCED("method1 GPU")(Catch::Benchmark::Chronometer meter) { method1(meter); };
  BENCHMARK_ADVANCED("method2 GPU")(Catch::Benchmark::Chronometer meter) { method2(meter); };
  BENCHMARK_ADVANCED("method3 GPU")(Catch::Benchmark::Chronometer meter) { method3(meter); };
  BENCHMARK_ADVANCED("method4 GPU")(Catch::Benchmark::Chronometer meter) { method4(meter); };
}
