#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/model/component_descriptor.hh>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test", "[model]") {
  constexpr auto FEFunctions = DiFfRG::FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;

  constexpr FEFunctions idxf{};

}
