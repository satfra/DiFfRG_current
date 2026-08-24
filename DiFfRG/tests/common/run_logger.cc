#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/common/run_logger.hh>
#include <DiFfRG/discretization/data/output_path.hh>

#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

using namespace DiFfRG;

namespace
{
  std::string read_file(const std::filesystem::path &path)
  {
    std::ifstream stream(path);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  }

  struct LoggerRun {
    std::string console;
    std::string file;
  };

  /** Emits one record per level through a logger built at the given verbosity, and returns what each sink received.
   * The console sink writes to the `stdout` stream, so the capture has to happen on the file descriptor. */
  LoggerRun run_at_verbosity(const std::filesystem::path &folder, const int verbosity)
  {
    std::filesystem::create_directories(folder);
    const auto console_capture = folder / "console.txt";

    Config::OutputSettings settings;
    settings.verbosity = verbosity;
    settings.log_flush_interval = 0.;

    const int saved_stdout = ::dup(STDOUT_FILENO);
    REQUIRE(saved_stdout >= 0);
    std::fflush(stdout);
    {
      const int capture = ::open(console_capture.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      REQUIRE(capture >= 0);
      ::dup2(capture, STDOUT_FILENO);
      ::close(capture);
    }

    {
      RunLogger logger(OutputPath(folder.string(), "run"), settings, true);
      const auto port = logger.port();
      port.debug("marker-debug");
      port.info("marker-info");
      port.warn("marker-warn");
      port.error("marker-error");
    }

    std::fflush(stdout);
    ::dup2(saved_stdout, STDOUT_FILENO);
    ::close(saved_stdout);

    return {read_file(console_capture), read_file(OutputPath(folder.string(), "run").run_file(".log"))};
  }
} // namespace

TEST_CASE("Run logger console output follows /output/verbosity", "[logger]")
{
  const auto root = std::filesystem::temp_directory_path() / "DiFfRG_run_logger_test";

  SECTION("verbosity 0 reports only problems")
  {
    const auto folder = root / "v0";
    std::filesystem::remove_all(folder);
    const auto result = run_at_verbosity(folder, 0);

    CHECK_THAT(result.console, !Catch::Matchers::ContainsSubstring("marker-info"));
    CHECK_THAT(result.console, Catch::Matchers::ContainsSubstring("marker-warn"));
    CHECK_THAT(result.console, Catch::Matchers::ContainsSubstring("marker-error"));
    // The file is never gated by the verbosity, only by /output/log_level.
    CHECK_THAT(result.file, Catch::Matchers::ContainsSubstring("marker-info"));
  }

  SECTION("verbosity 1 reports the ordinary messages")
  {
    const auto folder = root / "v1";
    std::filesystem::remove_all(folder);
    const auto result = run_at_verbosity(folder, 1);

    CHECK_THAT(result.console, Catch::Matchers::ContainsSubstring("marker-info"));
    CHECK_THAT(result.console, !Catch::Matchers::ContainsSubstring("marker-debug"));
  }

  SECTION("verbosity 2 reports the debug records")
  {
    const auto folder = root / "v2";
    std::filesystem::remove_all(folder);
    const auto result = run_at_verbosity(folder, 2);

    CHECK_THAT(result.console, Catch::Matchers::ContainsSubstring("marker-debug"));
    CHECK_THAT(result.file, Catch::Matchers::ContainsSubstring("marker-debug"));
  }
}
