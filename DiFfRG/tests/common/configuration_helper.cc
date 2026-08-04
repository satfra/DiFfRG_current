#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <fstream>

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/output_path.hh>

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

TEST_CASE("Legacy configuration output getters forward to OutputPath", "[config][common][migration]")
{
  const auto root = std::filesystem::absolute("configuration-helper-output").lexically_normal();
  const JSONValue json(json::value{
      {"output", {{"folder", root.string()}, {"name", "run"}, {"field_directory", "fields"}}}});
  const ConfigurationHelper config(json);
  const OutputPath output_path(json);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  CHECK(config.get_log_file() == output_path.run_file(".log").filename().string());
  CHECK(config.get_output_name() == output_path.run_name());
  CHECK(config.get_output_folder() == make_folder(output_path.field_directory().generic_string()));
  CHECK(config.get_top_folder() == make_folder(output_path.root().generic_string()));
#pragma GCC diagnostic pop
}
