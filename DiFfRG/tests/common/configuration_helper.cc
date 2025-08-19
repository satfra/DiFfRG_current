#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <fstream>

#include <DiFfRG/common/configuration_helper.hh>

using namespace DiFfRG;

TEST_CASE("Test configuration helper", "[config][common]")
{
  SECTION("File io")
  {
    // Generate some random parameter names and values
    json::value jv = {{"a", 1},
                      {"b", 2},
                      {"c", 3},
                      {"d", 4},
                      {"e", 5},
                      {"f", 6},
                      {"g", 7},
                      {"h", 8},
                      {"i", 9},
                      {"j", true},
                      {"output", {{"folder", "./"}, {"name", "output"}, {"verbosity", 0}}}};

    // Write jv to a file
    const std::string filename = "test_config_helper.json";
    std::ofstream file(filename);
    file << jv;
    file.close();

    // Use a ConfigurationHelper to read the file
    int argc = 3;
    char *argv[] = {(char *)"test", (char *)"-p", (char *)filename.c_str()};
    ConfigurationHelper config(argc, argv);

    // Check that the values are read correctly
    REQUIRE(config.get_json().get_int("/a") == 1);
    REQUIRE(config.get_json().get_int("/b") == 2);
    REQUIRE(config.get_json().get_int("/c") == 3);
    REQUIRE(config.get_json().get_int("/d") == 4);
    REQUIRE(config.get_json().get_int("/e") == 5);
    REQUIRE(config.get_json().get_int("/f") == 6);
    REQUIRE(config.get_json().get_int("/g") == 7);
    REQUIRE(config.get_json().get_int("/h") == 8);
    REQUIRE(config.get_json().get_int("/i") == 9);
    REQUIRE(config.get_json().get_bool("/j") == true);
  }
}