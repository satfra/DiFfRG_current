// Tests for Mesh Configuration

#include "DiFfRG/common/json.hh"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/generators/catch_generators_adapters.hpp"
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>
#include <catch2/catch_all.hpp>
#include <deal.II/base/point.h>

using namespace DiFfRG::Config;

// // Test case for default constructor
TEST_CASE("Mesh Configuration exists", "[MeshConfiguration]")
{
  SECTION("initializes correctly the mesh config")
  {

    // triangulation.refine_global(json.get_uint("/discretization/grid/refine"));
    DiFfRG::JSONValue json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1,1:0.2:2"}, {"z_grid", "0:0.1:1"}}},
         }},
    });
    ConfigurationMesh<3> mesh_config(json);
    REQUIRE(size(mesh_config.grids[0]) == 1);
    REQUIRE(size(mesh_config.grids[1]) == 2);
    REQUIRE(size(mesh_config.grids[2]) == 1);
  }

  SECTION("refine parameter works correctly")
  {

    DiFfRG::JSONValue json_with_refine = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1"}, {"refine", 1}}},
         }},
    });
    ConfigurationMesh<1> mesh_config_with_refine(json_with_refine);
    CHECK(mesh_config_with_refine.refine == 1);
    DiFfRG::JSONValue json_without_refine = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1"}}},
         }},
    });
    ConfigurationMesh<1> mesh_config_without_refine(json_without_refine);
    CHECK(mesh_config_without_refine.refine == 0);
  }

  SECTION("works only with x_axis")
  {
    DiFfRG::JSONValue json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1"}}},
         }},
    });
    ConfigurationMesh<1> mesh_config(json);
    REQUIRE(size(mesh_config.grids[0]) == 1);
  }
  SECTION("check range consistency")
  {
    DiFfRG::JSONValue json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1,1.1:0.2:2"}}},
         }},
    });
    CHECK_THROWS(ConfigurationMesh<1>(json));
  }
  SECTION("check a correct step size array")
  {
    DiFfRG::JSONValue json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.5:1,1:0.6:2"}}},
         }},
    });
    ConfigurationMesh<1> mesh_config(json);
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[0]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[0] == std::vector<double>{0.5, 0.5, 0.5, 0.5});
  }
  SECTION("check a correct step size array in 2D")
  {
    DiFfRG::JSONValue json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.5:1,1:0.6:2"}, {"y_grid", "0:1.0:2,2:1.0:4"}}},
         }},
    });
    ConfigurationMesh<2> mesh_config(json);
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[0]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[0] == std::vector<double>{0.5, 0.5, 0.5, 0.5});
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[1]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[1] == std::vector<double>(4, 1.0));
  }
  SECTION("check a correct step size array in 3D")
  {
    DiFfRG::JSONValue json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.5:1,1:0.6:2"}, {"y_grid", "0:1.0:2,2:1.0:4"}, {"z_grid", "0:2.0:2,2:2.0:4"}}},
         }},
    });
    ConfigurationMesh<3> mesh_config(json);
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[0]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[0] == std::vector<double>{0.5, 0.5, 0.5, 0.5});
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[1]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[1] == std::vector<double>(4, 1.0));
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[2]) == 2);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[2] == std::vector<double>(2, 2.0));
  }

  SECTION("Get lower_left and upper_right of grid in 3D", "[MeshConfiguration]")
  {
    using dealii::Point;
    DiFfRG::JSONValue json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid",
              {{"x_grid", "0:0.5:1,1:0.6:2"}, {"y_grid", "0.1:1.0:2,2:1.0:4"}, {"z_grid", "0.2:0.2:2,2:2.0:4"}}},
         }},
    });
    ConfigurationMesh<3> mesh_config(json);
    CHECK(mesh_config.get_lower_left() == Point<3, double>(0.0, 0.1, 0.2));
    CHECK(mesh_config.get_upper_right() == Point<3, double>(2.0, 4.0, 4.0));

    DiFfRG::JSONValue json2 = DiFfRG::json::value({
        {"discretization",
         {
             {"grid",
              {{"x_grid", "0.2:0.5:1,1:0.6:3"}, {"y_grid", "0.3:1.0:2,2:1.0:6"}, {"z_grid", "0.4:0.2:2,2:2.0:8"}}},
         }},
    });
    ConfigurationMesh<3> mesh_config2(json2);
    CHECK(mesh_config2.get_lower_left() == Point<3, double>(0.2, 0.3, 0.4));
    CHECK(mesh_config2.get_upper_right() == Point<3, double>(3.0, 6.0, 8.0));
  }
}

TEST_CASE("GridAxis tests", "[MeshConfiguration]")
{
  SECTION("parses range_string \"0:0.1:1\" correctly")
  {
    GridAxis grid_axis("0:0.1:1");
    REQUIRE(grid_axis.min == 0.0);
    REQUIRE(grid_axis.step == 0.1);
    REQUIRE(grid_axis.max == 1.0);
  }

  SECTION("parses range_string \"0.1:0.1:2\" correctly")
  {
    GridAxis grid_axis("0.1:0.1:2");
    REQUIRE(grid_axis.min == 0.1);
    REQUIRE(grid_axis.step == 0.1);
    REQUIRE(grid_axis.max == 2.0);
  }

  SECTION("returns array of steps")
  {
    GridAxis grid_axis("0.0:0.5:2");
    REQUIRE(size(grid_axis.get_stepwiths()) == 4);
    REQUIRE(grid_axis.get_stepwiths() == std::vector<double>{0.5, 0.5, 0.5, 0.5});
  }

  SECTION("returns array of steps if step does not add up exactly to max")
  {
    GridAxis grid_axis("0.0:0.6:2");
    REQUIRE(size(grid_axis.get_stepwiths()) == 4);
    REQUIRE(grid_axis.get_stepwiths() == std::vector<double>{0.5, 0.5, 0.5, 0.5});
  }

  SECTION("returns array of steps if step does not add up exactly to max, with one step")
  {
    GridAxis grid_axis("0:2.0:2");
    REQUIRE(size(grid_axis.get_stepwiths()) == 1);
    REQUIRE(grid_axis.get_stepwiths() == std::vector<double>{2.0});
  }
}

TEST_CASE("throws when range is invalid", "[MeshConfiguration]")
{
  CHECK_THROWS(GridAxis("0.1:0.01:0.0"));
  CHECK_THROWS(GridAxis("0.1:2.0:1.0"));
}