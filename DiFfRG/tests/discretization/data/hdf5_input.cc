#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/json.hh>
#include <DiFfRG/discretization/data/hdf5_input.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test HDF5 input", "[input][hdf5]")
{
  DiFfRG::Init();

#ifndef H5CPP
  // Test that HDF5Input throws appropriate errors when H5CPP is not enabled
  fs::path tmp{std::filesystem::temp_directory_path()};
  std::string hdf5FileName = "DiFfRG_test_hdf5_input_no_h5cpp.h5";

  // Create the file path
  std::filesystem::path hdf5_file(tmp / hdf5FileName);

  // For this test, just check that the constructor works without H5CPP
  // The actual functionality will be tested in the H5CPP section
  DiFfRG::HDF5Input hdf5_input(hdf5_file.string());

  return; // Skip the rest of the test if H5CPP is not available
#endif

  using namespace DiFfRG;
  namespace fs = std::filesystem;

  // Setup JSON configuration for HDF5Output
  JSONValue json = json::value({{"physical", {}},
                                {"integration", {{"x_quadrature_order", 32}, {"angle_quadrature_order", 8}}},
                                {"discretization",
                                 {{"fe_order", 3},
                                  {"threads", 8},
                                  {"batch_size", 64},
                                  {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}}}}},
                                {"timestepping", {{"final_time", 1.}, {"output_dt", 1e-1}}},
                                {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG HDF5Input test started");
  } catch (const spdlog::spdlog_ex &e) {
    // Logger already exists, continue
  }

  fs::path tmp{std::filesystem::temp_directory_path()};
  std::string hdf5FileName = "DiFfRG_test_hdf5_input.h5";
  std::filesystem::path hdf5_file(tmp / hdf5FileName);

  // Clean up any existing test file
  if (std::filesystem::exists(hdf5_file)) std::filesystem::remove(hdf5_file);

  SECTION("Test scalar loading")
  {
    // First, write test data using HDF5Output
    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, json);

      // Write various scalar types (only basic types that work reliably)
      hdf5_output.scalar("test_double", 42.5);
      hdf5_output.scalar("test_int", 123);
      hdf5_output.scalar("test_string", "hello_world");

      hdf5_output.flush(1.0);
    } // HDF5Output destructor will close the file

    // Now read the data back using HDF5Input
    {
      HDF5Input hdf5_input(hdf5_file.string());

      // Test loading different scalar types
      auto double_vec = hdf5_input.load_scalar<double>("test_double");
      REQUIRE(double_vec.size() == 1);
      CHECK(double_vec[0] == Catch::Approx(42.5));

      auto int_vec = hdf5_input.load_scalar<int>("test_int");
      REQUIRE(int_vec.size() == 1);
      CHECK(int_vec[0] == 123);

      auto string_vec = hdf5_input.load_scalar<std::string>("test_string");
      REQUIRE(string_vec.size() == 1);
      CHECK(string_vec[0] == "hello_world");

      // Test error when scalar doesn't exist
      REQUIRE_THROWS_AS(hdf5_input.load_scalar<double>("nonexistent_scalar"), std::runtime_error);
    }
  }

  SECTION("Test map loading with coordinates")
  {
    // Create test coordinates
    LinearCoordinates1D<double> coords1(10, 0.0, 1.0);
    LinearCoordinates1D<double> coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> coords2d(coords1, coords2);

    // Create test data
    std::vector<double> test_data(coords2d.size());
    for (size_t i = 0; i < coords2d.size(); ++i) {
      auto coord = coords2d.forward(coords2d.from_linear_index(i));
      test_data[i] = coord[0] * coord[0] + coord[1] * coord[1]; // x² + y²
    }

    // Write test data using HDF5Output
    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, json);
      hdf5_output.map("test_map", coords2d, test_data.data());
      hdf5_output.flush(1.0);
    }

    // Read the data back using HDF5Input
    {
      HDF5Input hdf5_input(hdf5_file.string());
      std::vector<double> loaded_data(coords2d.size());

      // Test loading map with matching coordinates
      REQUIRE_NOTHROW(hdf5_input.load_map("test_map", loaded_data.data(), coords2d));

      // Verify the data matches
      for (size_t i = 0; i < coords2d.size(); ++i)
        CHECK(loaded_data[i] == Catch::Approx(test_data[i]));

      // Test loading map without coordinate checking
      std::vector<double> loaded_data2(coords2d.size());
      REQUIRE_NOTHROW(hdf5_input.load_map("test_map", loaded_data2.data()));

      // Verify the data matches
      for (size_t i = 0; i < coords2d.size(); ++i)
        CHECK(loaded_data2[i] == Catch::Approx(test_data[i]));

      // Test error when map doesn't exist
      std::vector<double> dummy_data(coords2d.size());
      REQUIRE_THROWS_AS(hdf5_input.load_map("nonexistent_map", dummy_data.data()), std::runtime_error);
    }
  }

  SECTION("Test coordinate validation errors")
  {
    // Create original coordinates and different coordinates for testing validation
    LinearCoordinates1D<double> coords1(10, 0.0, 1.0);
    LinearCoordinates1D<double> coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> coords2d(coords1, coords2);

    // Create different coordinates that should fail validation
    LinearCoordinates1D<double> wrong_coords1(8, 0.0, 1.0); // Different size
    LinearCoordinates1D<double> wrong_coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> wrong_coords2d(wrong_coords1,
                                                                                              wrong_coords2);

    LinearCoordinates1D<double> wrong_range_coords1(10, 0.1, 1.0); // Different range
    LinearCoordinates1D<double> wrong_range_coords2(5, -1.0, 1.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> wrong_range_coords2d(
        wrong_range_coords1, wrong_range_coords2);

    // Create test data and write it
    std::vector<double> test_data(coords2d.size());
    for (size_t i = 0; i < coords2d.size(); ++i) {
      test_data[i] = static_cast<double>(i);
    }

    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, json);
      hdf5_output.map("test_map_validation", coords2d, test_data.data());
      hdf5_output.flush(1.0);
    }

    // Test coordinate validation
    {
      HDF5Input hdf5_input(hdf5_file.string());
      std::vector<double> loaded_data(coords2d.size());

      // This should work with matching coordinates
      REQUIRE_NOTHROW(hdf5_input.load_map("test_map_validation", loaded_data.data(), coords2d));

      // This should fail with wrong size coordinates
      std::vector<double> wrong_size_data(wrong_coords2d.size());
      REQUIRE_THROWS_AS(hdf5_input.load_map("test_map_validation", wrong_size_data.data(), wrong_coords2d),
                        std::runtime_error);

      // This should fail with wrong range coordinates
      std::vector<double> wrong_range_data(wrong_range_coords2d.size());
      REQUIRE_THROWS_AS(hdf5_input.load_map("test_map_validation", wrong_range_data.data(), wrong_range_coords2d),
                        std::runtime_error);
    }
  }

  SECTION("Test missing coordinates error")
  {
    // Create coordinates that haven't been written to the file
    LinearCoordinates1D<double> coords1(6, 0.0, 2.0);
    LinearCoordinates1D<double> coords2(4, -2.0, 2.0);
    CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>> missing_coords(coords1, coords2);

    // Write some data with different coordinates
    LinearCoordinates1D<double> existing_coords1(5, 0.0, 1.0);
    std::vector<double> test_data(existing_coords1.size(), 1.0);

    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, json);
      hdf5_output.map("test_map_missing_coords", existing_coords1, test_data.data());
      hdf5_output.flush(1.0);
    }

    // Try to read with coordinates that don't exist in the file
    {
      HDF5Input hdf5_input(hdf5_file.string());
      std::vector<double> loaded_data(missing_coords.size());

      // This should throw because the coordinates for missing_coords haven't been written
      REQUIRE_THROWS_AS(hdf5_input.load_map("test_map_missing_coords", loaded_data.data(), missing_coords),
                        std::runtime_error);
    }
  }

  // Clean up test file
  if (false && std::filesystem::exists(hdf5_file)) std::filesystem::remove(hdf5_file);
}
