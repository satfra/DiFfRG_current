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
  STATIC_REQUIRE(is_coordinates<LinPeriodicCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLinPeriodicCoordinates>);
  STATIC_REQUIRE(is_coordinates<LogLinLinPeriodicCoordinates>);
  STATIC_REQUIRE(!is_coordinates<int>);
  STATIC_REQUIRE(!is_coordinates<std::array<double, 3>>);
}

TEST_CASE("Test coordinate periodicity traits", "[coordinates]")
{
  using namespace DiFfRG;

  STATIC_REQUIRE(is_periodic_coordinate_v<LinPeriodicCoordinates>);
  STATIC_REQUIRE(is_periodic_coordinate_v<LinearPeriodicCoordinates1D<float>>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<LinCoordinates>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<LogCoordinates>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<int>);

  // per-axis queries on packs
  STATIC_REQUIRE(is_periodic_axis_v<LogLinLinPeriodicCoordinates, 2>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinPeriodicCoordinates, 1>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinPeriodicCoordinates, 0>);
  STATIC_REQUIRE(is_periodic_axis_v<LogLinPeriodicCoordinates, 1>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinCoordinates, 2>);

  // coordinate systems which do not expose their axes as types are never reported periodic
  STATIC_REQUIRE(!is_periodic_axis_v<LinCoordinates, 0>);
}