#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/discretization/coordinates/combined_coordinates.hh>
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

  // focused logarithmic coordinates and the packs built from them
  STATIC_REQUIRE(is_coordinates<FocusedLogCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedLogCoordinates1D<float>>);
  STATIC_REQUIRE(is_coordinates<FocusedLogLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedLogLinLinCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedLogLinPeriodicCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedLogLinLinPeriodicCoordinates>);
  STATIC_REQUIRE(is_coordinates<FocusedBosonicCoordinates1DFiniteT>);
  STATIC_REQUIRE(is_coordinates<FocusedFermionicCoordinates1DFiniteT>);

  // the radial axis of the finite-T coordinates is a template parameter which defaults to the
  // logarithmic one, so the historical two-argument spellings must keep working
  STATIC_REQUIRE(is_coordinates<BosonicCoordinates1DFiniteT<>>);
  STATIC_REQUIRE(is_coordinates<BosonicCoordinates1DFiniteT<int, float>>);
  STATIC_REQUIRE(is_coordinates<FermionicCoordinates1DFiniteT<int, double>>);
  STATIC_REQUIRE(std::is_same_v<BosonicCoordinates1DFiniteT<int, float>::radial_type, LogarithmicCoordinates1D<float>>);
  STATIC_REQUIRE(
      std::is_same_v<FocusedBosonicCoordinates1DFiniteT::radial_type, FocusedLogCoordinates1D<double>>);
}

TEST_CASE("Test coordinate periodicity traits", "[coordinates]")
{
  using namespace DiFfRG;

  STATIC_REQUIRE(is_periodic_coordinate_v<LinPeriodicCoordinates>);
  STATIC_REQUIRE(is_periodic_coordinate_v<LinearPeriodicCoordinates1D<float>>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<LinCoordinates>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<LogCoordinates>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<FocusedLogCoordinates>);
  STATIC_REQUIRE(!is_periodic_coordinate_v<int>);

  // per-axis queries on packs
  STATIC_REQUIRE(is_periodic_axis_v<FocusedLogLinLinPeriodicCoordinates, 2>);
  STATIC_REQUIRE(!is_periodic_axis_v<FocusedLogLinLinPeriodicCoordinates, 1>);
  STATIC_REQUIRE(!is_periodic_axis_v<FocusedLogLinLinPeriodicCoordinates, 0>);
  STATIC_REQUIRE(is_periodic_axis_v<LogLinLinPeriodicCoordinates, 2>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinPeriodicCoordinates, 1>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinPeriodicCoordinates, 0>);
  STATIC_REQUIRE(is_periodic_axis_v<LogLinPeriodicCoordinates, 1>);
  STATIC_REQUIRE(!is_periodic_axis_v<LogLinLinCoordinates, 2>);

  // coordinate systems which do not expose their axes as types are never reported periodic
  STATIC_REQUIRE(!is_periodic_axis_v<LinCoordinates, 0>);
}