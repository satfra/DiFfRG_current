#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <deal.II/base/multithread_info.h>
#include <fstream>

#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/init.hh>
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

TEST_CASE("Test configuration helper with TOML", "[config][common][toml]")
{
  const std::string document = R"TOML(
[physical]
T = 0.1
Lambda = 0.65

[output]
folder = "./"
name = "output"
verbosity = 0
)TOML";

  SECTION("A TOML file given with -p is read, and CLI overrides still apply")
  {
    const std::string filename = "test_config_helper.toml";
    std::ofstream(filename, std::ofstream::trunc) << document;

    int argc = 7;
    char *argv[] = {(char *)"test",           (char *)"-p", (char *)filename.c_str(), (char *)"-sd",
                    (char *)"/physical/T=0.5", (char *)"-ss", (char *)"/output/name=renamed"};
    ConfigurationHelper config(argc, argv);

    CHECK(config.get_config().get_double("/physical/Lambda") == Catch::Approx(0.65));
    CHECK(config.get_config().get_double("/physical/T") == Catch::Approx(0.5));
    CHECK(config.get_config().get_string("/output/name") == "renamed");
    // get_json() is the historical spelling of the same accessor.
    CHECK(config.get_json().get_double("/physical/Lambda") == Catch::Approx(0.65));

    std::filesystem::remove(filename);
  }

  SECTION("A missing default parameter file falls back to the TOML file of the same name")
  {
    const std::string stem = "test_config_helper_fallback";
    std::ofstream(stem + ".toml", std::ofstream::trunc) << document;
    std::filesystem::remove(stem + ".json");

    int argc = 1;
    char *argv[] = {(char *)"test"};
    ConfigurationHelper config(argc, argv, stem + ".json");

    CHECK(config.get_parameter_file() == stem + ".toml");
    CHECK(config.get_config().get_double("/physical/Lambda") == Catch::Approx(0.65));

    std::filesystem::remove(stem + ".toml");
  }
}

TEST_CASE("probe() reads the configuration without diagnostics or exiting", "[config][common][threads]")
{
  const std::string document = R"JSON({"discretization": {"threads": 3, "mesh_workers": 7}})JSON";

  SECTION("A -p file and CLI overrides are honoured just like the normal parse")
  {
    const std::string filename = "test_probe.json";
    std::ofstream(filename, std::ofstream::trunc) << document;

    int argc = 5;
    char *argv[] = {(char *)"test", (char *)"-p", (char *)filename.c_str(), (char *)"-si",
                    (char *)"/discretization/threads=5"};
    const ConfigTree config = ConfigurationHelper::probe(argc, argv);

    CHECK(config.get_uint("/discretization/threads", 0) == 5);
    CHECK(config.get_uint("/discretization/mesh_workers", 0) == 7);

    std::filesystem::remove(filename);
  }

  SECTION("An absent parameter file yields an empty tree rather than terminating")
  {
    int argc = 1;
    char *argv[] = {(char *)"test"};
    const ConfigTree config = ConfigurationHelper::probe(argc, argv, "test_probe_does_not_exist.json");

    CHECK_FALSE(config.contains("/discretization/threads"));
    CHECK(config.get_uint("/discretization/threads", 0) == 0);
  }

  SECTION("--help and --generate-parameter-file do not terminate a probe")
  {
    int argc = 3;
    char *argv[] = {(char *)"test", (char *)"--help", (char *)"--generate-parameter-file"};
    const ConfigTree config = ConfigurationHelper::probe(argc, argv, "test_probe_does_not_exist.json");

    CHECK_FALSE(config.contains("/discretization/threads"));
    CHECK_FALSE(std::filesystem::exists("test_probe_does_not_exist.json"));
  }
}

TEST_CASE("contains() distinguishes a missing key from a present one", "[config][common]")
{
  const ConfigTree json(json::value{{"discretization", {{"threads", 4}}}});

  CHECK(json.contains("/discretization/threads"));
  CHECK_FALSE(json.contains("/discretization/mesh_workers"));
  CHECK_FALSE(json.contains("/nonexistent/section/key"));
  // A defaulted getter must agree with contains().
  CHECK(json.get_uint("/discretization/threads", 99) == 4);
  CHECK(json.get_uint("/discretization/mesh_workers", 99) == 99);
}

TEST_CASE("set_thread_limit caps the process thread count", "[config][common][threads]")
{
  const unsigned int original = dealii::MultithreadInfo::n_threads();

  set_thread_limit(2);
  CHECK(dealii::MultithreadInfo::n_threads() == 2);

  // A tree without the key leaves the limit at "all cores".
  set_thread_limit(ConfigTree(json::value{{"discretization", {{"mesh_workers", 8}}}}));
  CHECK(dealii::MultithreadInfo::n_threads() == dealii::MultithreadInfo::n_cores());

  set_thread_limit(ConfigTree(json::value{{"discretization", {{"threads", 3}}}}));
  CHECK(dealii::MultithreadInfo::n_threads() == 3);

  set_thread_limit(original);
}

TEST_CASE("Legacy configuration output getters forward to OutputPath", "[config][common][migration]")
{
  const auto root = std::filesystem::absolute("configuration-helper-output").lexically_normal();
  const ConfigTree json(json::value{
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
