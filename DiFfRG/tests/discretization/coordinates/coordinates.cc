#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/discretization/coordinates/coordinates.hh>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test Coordinate template constraints", "[coordinates]")
{
  using namespace DiFfRG;

  // Test static assertions for coordinate types
  STATIC_REQUIRE(is_coordinates<LinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLogCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LinLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLogLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<LinLinLinCoordinates>);
  STATIC_REQUIRE(!is_coordinates<int>);
  STATIC_REQUIRE(!is_coordinates<std::array<double, 3>>);
}