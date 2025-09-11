#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/json.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test HDF5 output", "[output][hdf5]")
{
  DiFfRG::Init();
#ifndef H5CPP
  HDF5Output hdf5_output(tmp.string(), hdf5FileName, json);

  // Output should throw an error if H5CPP is not enabled
  REQUIRE_THROWS_AS(hdf5_output.scalar("d", 42.0), std::runtime_error);
  REQUIRE_THROWS_AS(hdf5_output.map("coords", DiFfRG::Coordinates1D<double>({0.0, 1.0})), std::runtime_error);
  REQUIRE_NOTHROW(hdf5_output.flush());

  exit(0);
#endif

  using namespace dealii;
  using namespace DiFfRG;

  JSONValue json = json::value(
      {{"physical", {}},
       {"integration",
        {{"x_quadrature_order", 32},
         {"angle_quadrature_order", 8},
         {"x0_quadrature_order", 16},
         {"x0_summands", 8},
         {"q0_quadrature_order", 16},
         {"q0_summands", 8},
         {"x_extent_tolerance", 1e-3},
         {"x0_extent_tolerance", 1e-3},
         {"q0_extent_tolerance", 1e-3},
         {"jacobian_quadrature_factor", 0.5}}},
       {"discretization",
        {{"fe_order", 3},
         {"threads", 8},
         {"batch_size", 64},
         {"overintegration", 0},
         {"output_subdivisions", 2},
         {"output_buffer_size", 10},

         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},

         {"grid", {{"x_grid", "0:0.0001:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
         {"adaptivity",
          {{"start_adapt_at", 0.},
           {"adapt_dt", 1e-1},
           {"level", 0},
           {"refine_percent", 1e-1},
           {"coarsen_percent", 5e-2}}}}},
       {"timestepping",
        {{"final_time", 1.},
         {"output_dt", 1e-1},
         {"explicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}},
         {"implicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}}}},
       {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG Application started");
  } catch (const spdlog::spdlog_ex &e) {
    // nothing, the logger is already set up
  }

  fs::path tmp{std::filesystem::temp_directory_path()};
  std::string hdf5FileName = "DiFfRG_test_hdf5_output.h5";

  SECTION("Scalars")
  {
    {
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, json);

      hdf5_output.scalar("d", 42.0);
      hdf5_output.scalar("i", 1);
      hdf5_output.scalar("s", "test_string");
      hdf5_output.scalar("c", complex<double>(1.0, 2.0));
      hdf5_output.scalar("ad", autodiff::real({1.0, 2.0}));
      hdf5_output.scalar("arr", std::array<double, 3>{{1.0, 2.0, 3.0}});
      hdf5_output.flush(0.0); // Flush to ensure the data is written
      hdf5_output.scalar("d", 12.0);
      // this should throw, as the datatype does not match
      REQUIRE_THROWS_AS(hdf5_output.scalar("i", 3.14), std::runtime_error);
      hdf5_output.scalar("i", 6);
      hdf5_output.scalar("s", "another_string");
      hdf5_output.scalar("c", complex<double>(3.0, 4.0));
      hdf5_output.scalar("ad", autodiff::real({2.0, 1.0}));
      hdf5_output.scalar("arr", std::array<double, 3>{{9.0, 8.0, 7.0}});
      // this should throw, as "d" has already been written
      REQUIRE_THROWS_AS(hdf5_output.scalar("d", 24.0), std::runtime_error);
      // this should throw, as "fff" has not been written in the previous flush
      REQUIRE_THROWS_AS(hdf5_output.scalar("fff", 24.0), std::runtime_error);
      hdf5_output.flush(1.0); // Flush again to ensure the data is written
    }
    spdlog::get("log")->info("HDF5 file written.");

    // Check if the file was created and contains the expected data
    std::filesystem::path hdf5_file(tmp / hdf5FileName);
    REQUIRE(std::filesystem::exists(hdf5_file));
    REQUIRE(std::filesystem::is_regular_file(hdf5_file));
    // Check if the file contains the expected datasets

    auto file = hdf5::file::open(hdf5_file, hdf5::file::AccessFlags::ReadOnly);

    auto root = file.root();
    REQUIRE(root.has_group("scalars"));
    auto scalars_group = root.get_group("scalars");

    spdlog::get("log")->info("checking d...");
    REQUIRE(scalars_group.has_dataset("d"));
    auto d_dataset = scalars_group.get_dataset("d");
    REQUIRE(d_dataset.datatype() == hdf5::datatype::create<double>());
    std::vector<double> d_data(2);
    d_dataset.read(d_data);
    REQUIRE(d_data.size() == 2);
    REQUIRE(d_data[0] == 42.0);
    REQUIRE(d_data[1] == 12.0);

    spdlog::get("log")->info("checking i...");
    REQUIRE(scalars_group.has_dataset("i"));
    auto i_dataset = scalars_group.get_dataset("i");
    REQUIRE(i_dataset.datatype() == hdf5::datatype::create<int>());
    std::vector<int> i_data(2);
    i_dataset.read(i_data);
    REQUIRE(i_data.size() == 2);
    REQUIRE(i_data[0] == 1);
    REQUIRE(i_data[1] == 6);

    spdlog::get("log")->info("checking s...");
    REQUIRE(scalars_group.has_dataset("s"));
    auto s_dataset = scalars_group.get_dataset("s");
    REQUIRE(s_dataset.datatype() == hdf5::datatype::create<std::string>());
    std::vector<std::string> s_data(2);
    s_dataset.read(s_data);
    REQUIRE(s_data.size() == 2);
    REQUIRE(s_data[0] == "test_string");
    REQUIRE(s_data[1] == "another_string");

    spdlog::get("log")->info("checking c...");
    REQUIRE(scalars_group.has_dataset("c"));
    auto c_dataset = scalars_group.get_dataset("c");
    REQUIRE(c_dataset.datatype() == hdf5::datatype::create<complex<double>>());
    std::vector<complex<double>> c_data(2);
    c_dataset.read(c_data);
    REQUIRE(c_data.size() == 2);
    REQUIRE(c_data[0] == complex<double>(1.0, 2.0));
    REQUIRE(c_data[1] == complex<double>(3.0, 4.0));

    spdlog::get("log")->info("checking ad...");
    REQUIRE(scalars_group.has_dataset("ad"));
    auto ad_dataset = scalars_group.get_dataset("ad");
    REQUIRE(ad_dataset.datatype() == hdf5::datatype::create<autodiff::real>());
    std::vector<autodiff::real> ad_data(2);
    ad_dataset.read(ad_data);
    REQUIRE(ad_data.size() == 2);
    REQUIRE(ad_data[0][0] == 1.0);
    REQUIRE(ad_data[0][1] == 2.0);
    REQUIRE(ad_data[1][0] == 2.0);
    REQUIRE(ad_data[1][1] == 1.0);

    spdlog::get("log")->info("checking arr...");
    REQUIRE(scalars_group.has_dataset("arr"));
    auto arr_dataset = scalars_group.get_dataset("arr");
    REQUIRE(arr_dataset.datatype() == hdf5::datatype::create<std::array<double, 3>>());
    std::vector<std::array<double, 3>> arr_data(2);
    arr_dataset.read(arr_data);
    REQUIRE(arr_data.size() == 2);
    REQUIRE(arr_data[0][0] == 1.0);
    REQUIRE(arr_data[0][1] == 2.0);
    REQUIRE(arr_data[0][2] == 3.0);
    REQUIRE(arr_data[1][0] == 9.0);
    REQUIRE(arr_data[1][1] == 8.0);
    REQUIRE(arr_data[1][2] == 7.0);

    // Remove the file after the test
    spdlog::get("log")->info("Cleanup.");
    std::filesystem::remove(hdf5_file);
  }

  SECTION("maps")
  {
    // Let's set up some coordinates and interpolators to test on
    const size_t coord_size = GENERATE(take(3, random(8, 64)));
    LogCoordinates log_coords(coord_size, 0.1, 10.0, 5.);
    LogCoordinates log_coords_wrong(coord_size, 0.1, 12.0, 5.);
    LogLogCoordinates loglog_coords(log_coords, log_coords);
    LogLogCoordinates loglog_coords_wrong(log_coords_wrong, log_coords);

    SplineInterpolator1D<double, LogCoordinates> spline(log_coords);
    std::vector<double> spline_data(coord_size);
    for (size_t i = 0; i < coord_size; ++i)
      spline_data[i] = powr<2>(i);
    spline.update(spline_data.data());

    LinearInterpolatorND<double, LogLogCoordinates> lin2d(loglog_coords);
    std::vector<double> lin2d_data(coord_size * coord_size);
    for (size_t i = 0; i < coord_size; ++i)
      for (size_t j = 0; j < coord_size; ++j)
        lin2d_data[i * coord_size + j] = powr<2>(i) + powr<2>(j);
    lin2d.update(lin2d_data.data());

    {
      // Write the coordinates and interpolators very manually to files
      {
        HDF5Output hdf5_output(tmp.string(), hdf5FileName, json);

        hdf5_output.map("log", log_coords);
        hdf5_output.map("log_w", log_coords_wrong);
        hdf5_output.map("loglog", loglog_coords);
        hdf5_output.map("loglog_w", log_coords_wrong);

        // this should throw, as "log" has already been written
        REQUIRE_THROWS_AS(hdf5_output.map("log", log_coords), std::runtime_error);
        // These should throw as the coordinates are inconsistent
        REQUIRE_THROWS_AS(hdf5_output.map("spline", "log_w", spline), std::runtime_error);
        REQUIRE_THROWS_AS(hdf5_output.map("lin2d", "loglog_w", lin2d), std::runtime_error);

        hdf5_output.map("spline", "log", spline);
        hdf5_output.map("lin2d", "loglog", lin2d);
        hdf5_output.map("lin2d_Direct", loglog_coords, lin2d_data.data()); // try direct data input

        hdf5_output.flush(0.0);                   // Flush to ensure the data is written
        hdf5_output.map("spline", "log", spline); // This should not throw, as the map is updated
        hdf5_output.map("lin2d", "loglog", lin2d);
        hdf5_output.flush(1.0); // Flush again to ensure the data is written
      }

      // Do checks!
      std::filesystem::path hdf5_file(tmp / hdf5FileName);
      REQUIRE(std::filesystem::exists(hdf5_file));
      REQUIRE(std::filesystem::is_regular_file(hdf5_file));

      // Check if the file contains the expected dataset
      auto file = hdf5::file::open(hdf5_file, hdf5::file::AccessFlags::ReadOnly);
      auto root = file.root();
      REQUIRE(root.has_group("coordinates"));
      auto coords_group = root.get_group("coordinates");

      spdlog::get("log")->info("checking log coordinates...");
      REQUIRE(coords_group.has_dataset("log"));
      auto log_dataset = coords_group.get_dataset("log");
      std::vector<device::array<double, 1>> log_data(coord_size);
      log_dataset.read(log_data);
      for (size_t i = 0; i < coord_size; ++i) {
        REQUIRE(log_data[i][0] == log_coords.forward(i));
      }

      spdlog::get("log")->info("checking loglog coordinates...");
      REQUIRE(coords_group.has_dataset("loglog"));
      auto loglog_dataset = coords_group.get_dataset("loglog");
      std::vector<device::array<double, 2>> loglog_data(coord_size * coord_size);
      loglog_dataset.read(loglog_data);
      for (size_t i = 0; i < loglog_coords.size(); ++i) {
        REQUIRE(loglog_data[i] == loglog_coords.forward(loglog_coords.from_linear_index(i)));
      }

      spdlog::get("log")->info("checking spline map...");
      REQUIRE(root.has_group("maps"));
      auto maps_group = root.get_group("maps");
      REQUIRE(maps_group.has_group("spline"));
      auto spline_group = maps_group.get_group("spline");
      REQUIRE(spline_group.has_group(int_to_string(0, 6)));
      auto sub_group = spline_group.get_group(int_to_string(0, 6));
      REQUIRE(sub_group.has_dataset("data"));
      auto spline_dataset = sub_group.get_dataset("data");
      std::vector<double> spline_map_data(coord_size);
      spline_dataset.read(spline_map_data);
      for (size_t i = 0; i < coord_size; ++i) {
        const auto lcoord = log_coords.forward(log_coords.from_linear_index(i));
        // There will be a small numerical error in the interpolation (~1e-15 - 1e-14), so we use a tolerance
        REQUIRE(
            is_close(spline_map_data[i], device::apply([&](const auto &...x) { return spline(x...); }, lcoord), 1e-14));
      }

      spdlog::get("log")->info("checking lin2d map...");
      REQUIRE(maps_group.has_group("lin2d"));
      auto lin2d_group = maps_group.get_group("lin2d");
      REQUIRE(lin2d_group.has_group(int_to_string(0, 6)));
      auto lin2d_sub_group = lin2d_group.get_group(int_to_string(0, 6));
      REQUIRE(lin2d_sub_group.has_dataset("data"));
      auto lin2d_dataset = lin2d_sub_group.get_dataset("data");
      std::vector<double> lin2d_map_data(coord_size * coord_size);
      lin2d_dataset.read(lin2d_map_data);
      for (size_t i = 0; i < coord_size * coord_size; ++i) {
        const auto llcoord = loglog_coords.forward(loglog_coords.from_linear_index(i));
        // There will be a small numerical error in the interpolation (~1e-15 - 1e-14), so we use a tolerance
        REQUIRE(
            is_close(lin2d_map_data[i], device::apply([&](const auto &...x) { return lin2d(x...); }, llcoord), 1e-14));
      }

      spdlog::get("log")->info("checking lin2d direct map...");
      REQUIRE(maps_group.has_group("lin2d_Direct"));
      auto lin2d_direct_group = maps_group.get_group("lin2d_Direct");
      REQUIRE(lin2d_direct_group.has_group(int_to_string(0, 6)));
      auto lin2d_direct_sub_group = lin2d_direct_group.get_group(int_to_string(0, 6));
      REQUIRE(lin2d_direct_sub_group.has_dataset("data"));
      auto lin2d_direct_dataset = lin2d_direct_sub_group.get_dataset("data");
      std::vector<double> lin2d_direct_map_data(coord_size * coord_size);
      lin2d_direct_dataset.read(lin2d_direct_map_data);
      for (size_t i = 0; i < coord_size * coord_size; ++i) {
        const auto llcoord = loglog_coords.forward(loglog_coords.from_linear_index(i));
        // There will be a small numerical error in the interpolation (~1e-15 - 1e-14), so we use a tolerance
        REQUIRE(is_close(lin2d_direct_map_data[i],
                         device::apply([&](const auto &...x) { return lin2d(x...); }, llcoord), 1e-14));
      }

      // Remove the file after the test
      spdlog::get("log")->info("Cleanup.");
      std::filesystem::remove(hdf5_file);
    }
    {
      spdlog::get("log")->info("checking convenience writing.");
      HDF5Output hdf5_output(tmp.string(), hdf5FileName, json);

      REQUIRE_NOTHROW(hdf5_output.map("spline", spline));
      REQUIRE_NOTHROW(hdf5_output.map("lin2d", lin2d));

      std::filesystem::path hdf5_file(tmp / hdf5FileName);
      REQUIRE(std::filesystem::exists(hdf5_file));
      REQUIRE(std::filesystem::is_regular_file(hdf5_file));

      // Check if the file contains the expected dataset
      auto file = hdf5::file::open(hdf5_file, hdf5::file::AccessFlags::ReadOnly);
      auto root = file.root();

      spdlog::get("log")->info("checking spline map...");
      REQUIRE(root.has_group("maps"));
      auto maps_group = root.get_group("maps");
      REQUIRE(maps_group.has_group("spline"));
      auto spline_group = maps_group.get_group("spline");
      REQUIRE(spline_group.has_group(int_to_string(0, 6)));
      auto sub_group = spline_group.get_group(int_to_string(0, 6));
      REQUIRE(sub_group.has_dataset("data"));

      spdlog::get("log")->info("checking lin2d map...");
      REQUIRE(maps_group.has_group("lin2d"));
      auto lin2d_group = maps_group.get_group("lin2d");
      REQUIRE(lin2d_group.has_group(int_to_string(0, 6)));
      auto lin2d_sub_group = lin2d_group.get_group(int_to_string(0, 6));
      REQUIRE(lin2d_sub_group.has_dataset("data"));

      // Skip the explicit data verification here, we just wanted to know that the convenience method worked.

      // Remove the file after the test
      spdlog::get("log")->info("Cleanup.");
      std::filesystem::remove(hdf5_file);
    }
  }
}