#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/model/component_descriptor.hh>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test", "[model]")
{
  using namespace DiFfRG;

  using FEFunctions = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;

  constexpr FEFunctions idxf{};

  std::vector<int> data{GENERATE(take(10, random(0, 100))), GENERATE(take(10, random(0, 100))),
                        GENERATE(take(10, random(0, 100))), GENERATE(take(10, random(0, 100)))};

  [[maybe_unused]] constexpr auto idxu = idxf["u"];
  [[maybe_unused]] constexpr auto idxv = idxf["v"];

  const auto data_u = data[idxf["u"]];
  const auto data_v = data[idxf["v"]];

  REQUIRE(data_u == data[0]);
  REQUIRE(data_v == data[1]);
}
