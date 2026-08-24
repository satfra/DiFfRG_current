#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/data/output_session.hh>

#include <deal.II/lac/vector.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <spdlog/spdlog.h>

namespace
{
  DiFfRG::Config::OutputSettings output_settings()
  {
    DiFfRG::Config::OutputSettings settings;
    settings.Lambda = 1.;
    settings.write_vtk = false;
    settings.write_hdf5 = false;
    return settings;
  }

  DiFfRG::ConfigTree output_json(const std::filesystem::path &path)
  {
    return DiFfRG::json::value(
        {{"physical", {{"Lambda", 1.0}}},
         {"discretization", {{"output_subdivisions", 1}}},
         {"output",
          {{"folder", path.string()}, {"name", "run"}, {"vtk", false}, {"hdf5", false}, {"max_pending_frames", 2}}}});
  }
} // namespace

TEST_CASE("OutputSession rejects an incomplete table frame", "[output][session]")
{
  auto path = DiFfRG::OutputPath::temporary();

  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    output.write_frame(0.0, [](auto &frame) {
      auto table = frame.table("data.csv");
      table.value("a", 1.0);
      table.value("b", 2.0);
    });

    CHECK_THROWS_WITH(output.write_frame(1.0, [](auto &frame) { frame.table("data.csv").value("a", 3.0); }),
                      Catch::Matchers::ContainsSubstring("incomplete"));
    CHECK_THROWS_WITH(output.write_frame(2.0, [](auto &) {}), Catch::Matchers::ContainsSubstring("incomplete"));
  }
}

TEST_CASE("DiagnosticPort serializes concurrent records", "[output][diagnostics][thread-safe]")
{
  auto path = DiFfRG::OutputPath::temporary();

  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    const auto diagnostics = output.diagnostic_port();

    std::vector<std::jthread> writers;
    for (int i = 0; i < 16; ++i)
      writers.emplace_back([diagnostics, i]() { diagnostics.scalar("eigenvalues.csv", "lambda_max", i, i * 0.5); });
  }

  std::ifstream stream(path.root() / "output_eigenvalues.csv");
  REQUIRE(stream.good());
  std::size_t line_count = 0;
  for (std::string line; std::getline(stream, line);)
    ++line_count;
  CHECK(line_count == 17);
}

TEST_CASE("DiagnosticPort is fail-stop after a malformed record", "[output][diagnostics][schema]")
{
  auto path = DiFfRG::OutputPath::temporary();

  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    const auto diagnostics = output.diagnostic_port();
    diagnostics.record("coefficients.csv", 0.0, {{"diffusion", 1.0}, {"viscosity", 2.0}});
    CHECK_THROWS_WITH(diagnostics.scalar("coefficients.csv", "diffusion", 1.0, 3.0),
                      Catch::Matchers::ContainsSubstring("incomplete"));
    CHECK_THROWS_WITH(diagnostics.scalar("coefficients.csv", "diffusion", 2.0, 4.0),
                      Catch::Matchers::ContainsSubstring("incomplete"));
  }
}

TEST_CASE("OutputSession rejects path traversal and use after finish", "[output][session][layout]")
{
  auto path = DiFfRG::OutputPath::temporary();

  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    CHECK_FALSE(spdlog::get("run"));
    CHECK_THROWS_WITH(output.write_frame(0.0, [](auto &frame) { frame.table("../escape.csv"); }),
                      Catch::Matchers::ContainsSubstring("unsafe"));
    CHECK_THROWS_WITH(output.finish(), Catch::Matchers::ContainsSubstring("unsafe"));
  }
  {
    DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());
    output.finish();
    CHECK_THROWS_WITH(output.write_frame(1.0, [](auto &) {}),
                      Catch::Matchers::ContainsSubstring("already been finished"));
  }
}

TEST_CASE("OutputSession drain keeps the session writable until finish", "[output][session][lifecycle]")
{
  auto path = DiFfRG::OutputPath::temporary();
  DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, output_settings());

  output.write_frame(0.0, [](auto &frame) { frame.table("data.csv").value("value", 1.0); });
  REQUIRE_NOTHROW(output.drain());

  output.write_frame(1.0, [](auto &frame) { frame.table("data.csv").value("value", 2.0); });
  REQUIRE_NOTHROW(output.drain());

  output.finish();
  CHECK_THROWS_WITH(output.write_frame(2.0, [](auto &) {}),
                    Catch::Matchers::ContainsSubstring("already been finished"));
}

TEST_CASE("Run log reaches disk while the session is alive", "[output][session][log]")
{
  auto path = DiFfRG::OutputPath::temporary();
  const auto log_file = path.run_file(".log");

  auto settings = output_settings();
  settings.log_flush_interval = 0.1;

  DiFfRG::OutputSession_impl<0, dealii::Vector<double>> output(path, settings);
  output.log_port().info("periodic flush marker");

  // A single short line stays inside the file sink's stdio buffer, so this only passes if the flusher runs.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  bool found = false;
  while (!found && std::chrono::steady_clock::now() < deadline) {
    std::ifstream stream(log_file);
    std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    found = content.find("periodic flush marker") != std::string::npos;
    if (!found) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  CHECK(found);
}

TEST_CASE("OutputPath has explicit move-only ownership", "[output][path]")
{
  STATIC_CHECK_FALSE(std::is_default_constructible_v<DiFfRG::OutputPath>);
  STATIC_CHECK_FALSE(std::is_copy_constructible_v<DiFfRG::OutputPath>);
  STATIC_CHECK_FALSE(std::is_copy_assignable_v<DiFfRG::OutputPath>);
  STATIC_CHECK(std::is_move_constructible_v<DiFfRG::OutputPath>);
  STATIC_CHECK(std::is_move_assignable_v<DiFfRG::OutputPath>);
}

TEST_CASE("OutputPath temporary cleanup and retention are explicit", "[output][path][temporary]")
{
  std::filesystem::path removed_path;
  {
    auto path = DiFfRG::OutputPath::temporary();
    removed_path = path.root();
    REQUIRE(std::filesystem::is_directory(removed_path));
  }
  CHECK_FALSE(std::filesystem::exists(removed_path));

  std::filesystem::path kept_path;
  {
    auto path = DiFfRG::OutputPath::temporary(DiFfRG::TemporaryRetention::keep);
    kept_path = path.root();
  }
  CHECK(std::filesystem::is_directory(kept_path));
  std::filesystem::remove_all(kept_path);
}

TEST_CASE("OutputPath moves transfer temporary cleanup ownership", "[output][path][temporary]")
{
  std::filesystem::path root;
  {
    auto original = DiFfRG::OutputPath::temporary();
    root = original.root();
    auto moved = std::move(original);
    CHECK(std::filesystem::is_directory(root));

    auto replacement = DiFfRG::OutputPath::temporary();
    replacement = std::move(moved);
    CHECK(std::filesystem::is_directory(root));
  }
  CHECK_FALSE(std::filesystem::exists(root));
}

TEST_CASE("OutputPath child shares its temporary root owner", "[output][path][temporary]")
{
  std::filesystem::path root;
  {
    auto child = [&]() {
      auto parent = DiFfRG::OutputPath::temporary();
      root = parent.root();
      return parent.child("tuning/run-1", "trial");
    }();
    CHECK(std::filesystem::is_directory(root));
    CHECK(child.root() == root / "tuning/run-1");
  }
  CHECK_FALSE(std::filesystem::exists(root));
}

TEST_CASE("The configuration copy is only written when nothing else records it", "[output][session][json]")
{
  using Session = DiFfRG::OutputSession_impl<0, dealii::Vector<double>>;

  const auto log_json = [](const DiFfRG::OutputPath &path) { return path.root() / (path.run_name() + ".log.json"); };

  SECTION("HDF5 off: the configuration has nowhere else to go, so it is written")
  {
    auto path = DiFfRG::OutputPath::temporary();
    auto settings = output_settings();
    REQUIRE_FALSE(settings.write_hdf5);
    {
      Session output(path, settings);
    }
    CHECK(std::filesystem::exists(log_json(path)));
  }

  SECTION("HDF5 on: the configuration travels in /config, so no separate copy")
  {
    auto path = DiFfRG::OutputPath::temporary();
    auto settings = output_settings();
    settings.write_hdf5 = true;
    {
      Session output(path, settings);
    }
    CHECK_FALSE(std::filesystem::exists(log_json(path)));
  }

  SECTION("/output/json forces the copy even with HDF5 on")
  {
    auto path = DiFfRG::OutputPath::temporary();
    auto settings = output_settings();
    settings.write_hdf5 = true;
    settings.write_json = true;
    {
      Session output(path, settings);
    }
    CHECK(std::filesystem::exists(log_json(path)));
  }

  SECTION("/output/json is read from the configuration and defaults to off")
  {
    auto parent = DiFfRG::OutputPath::temporary();
    auto json = output_json(parent.root());
    CHECK_FALSE(DiFfRG::Config::OutputSettings(json).write_json);

    json().as_object().at("output").as_object()["json"] = true;
    CHECK(DiFfRG::Config::OutputSettings(json).write_json);
  }
}

TEST_CASE("The session records how the run ended", "[output][session][hdf5][status]")
{
  using Session = DiFfRG::OutputSession_impl<0, dealii::Vector<double>>;

  const auto status = [](const DiFfRG::OutputPath &path) {
    auto file =
        DiFfRG::hdf5::File::open((path.root() / (path.run_name() + ".h5")).string(), DiFfRG::hdf5::Access::ReadOnly);
    auto root = file.root();
    return std::pair{root.read_attribute<int>("finished"), root.read_attribute<int>("crashed")};
  };

  const auto hdf5_settings = [] {
    auto settings = output_settings();
    settings.write_hdf5 = true;
    return settings;
  };

  SECTION("an explicit finish() is a clean end")
  {
    auto path = DiFfRG::OutputPath::temporary();
    {
      Session output(path, hdf5_settings());
      output.finish();
    }
    CHECK(status(path) == std::pair{1, 0});
  }

  SECTION("falling off the end of the scope is also a clean end")
  {
    auto path = DiFfRG::OutputPath::temporary();
    {
      Session output(path, hdf5_settings());
    }
    CHECK(status(path) == std::pair{1, 0});
  }

  SECTION("unwinding through the destructor is finished, but crashed")
  {
    auto path = DiFfRG::OutputPath::temporary();
    try {
      Session output(path, hdf5_settings());
      throw std::runtime_error("the timestepper gave up");
    } catch (const std::runtime_error &) {
    }
    // The session still flushed and closed its files -- that is the point of the distinction.
    CHECK(status(path) == std::pair{1, 1});
  }

  SECTION("an explicit finish() before the throw still counts as clean")
  {
    auto path = DiFfRG::OutputPath::temporary();
    try {
      Session output(path, hdf5_settings());
      output.finish();
      throw std::runtime_error("thrown after the outputs were closed");
    } catch (const std::runtime_error &) {
    }
    CHECK(status(path) == std::pair{1, 0});
  }
}

TEST_CASE("OutputSettings keeps debugging and memory policy in C++", "[output][settings][json]")
{
  auto parent = DiFfRG::OutputPath::temporary();
  auto json = output_json(parent.root());
  auto &output = json().as_object().at("output").as_object();
  output["async"] = false;
  output["max_pending_bytes"] = 1;

  const DiFfRG::Config::OutputSettings settings(json);
  CHECK(settings.asynchronous);
  CHECK(settings.max_pending_bytes == DiFfRG::Config::OutputSettings::default_max_pending_bytes);
}

TEST_CASE("OutputPath creates unique temporary roots concurrently", "[output][path][thread-safe]")
{
  std::mutex mutex;
  std::vector<std::filesystem::path> roots;
  std::vector<std::jthread> threads;
  for (unsigned int i = 0; i < 16; ++i)
    threads.emplace_back([&]() {
      auto path = DiFfRG::OutputPath::temporary();
      std::scoped_lock lock(mutex);
      roots.push_back(path.root());
    });
  threads.clear();
  CHECK(std::set(roots.begin(), roots.end()).size() == roots.size());
}

TEST_CASE("OutputPath JSON constructor forwards to the persistent layout", "[output][path][json]")
{
  auto temporary_parent = DiFfRG::OutputPath::temporary();
  const auto configured_root = temporary_parent.root() / "configured";
  auto json = output_json(configured_root);
  {
    DiFfRG::OutputPath typed(configured_root, "run", "run");
    DiFfRG::OutputPath path(json);
    CHECK(path.root() == typed.root());
    CHECK(path.run_name() == typed.run_name());
    CHECK(path.field_directory() == typed.field_directory());
    CHECK(path.root() == std::filesystem::absolute(configured_root).lexically_normal());
    CHECK(path.run_name() == "run");
    CHECK(path.field_directory() == "run");
    std::filesystem::create_directories(path.root());
  }
  CHECK(std::filesystem::is_directory(configured_root));
}

TEST_CASE("OutputPath rejects unsafe paths and failed artifact copies", "[output][path][safety]")
{
  CHECK_THROWS_AS(DiFfRG::OutputPath({}, "run"), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::OutputPath("output", ""), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::OutputPath("output", "nested/run"), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::OutputPath("output", "../run"), std::invalid_argument);
  CHECK_THROWS_AS(DiFfRG::OutputPath("output", "run", "../fields"), std::invalid_argument);

  auto path = DiFfRG::OutputPath::temporary();
  CHECK_THROWS_AS(path.resolve("../escape"), std::invalid_argument);
  CHECK_THROWS_AS(path.resolve(std::filesystem::current_path()), std::invalid_argument);
  CHECK_THROWS_AS(path.run_file("csv"), std::invalid_argument);
  CHECK_THROWS_AS(path.run_file(".dir/file"), std::invalid_argument);
  CHECK_THROWS_AS(path.child("../escape", "child"), std::invalid_argument);
  CHECK_THROWS_WITH(path.copy_tree_from(path.root() / "missing"), Catch::Matchers::ContainsSubstring("does not exist"));
}
