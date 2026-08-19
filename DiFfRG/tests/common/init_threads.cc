#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>

#include <Kokkos_Core.hpp>
#include <deal.II/base/multithread_info.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>

namespace
{
  std::string write_config(const std::string &name, const std::string &discretization)
  {
    const std::string filename = name + ".json";
    std::ofstream(filename, std::ofstream::trunc) << "{\"discretization\":" << discretization << "}";
    return filename;
  }

  void check_threads_host_concurrency(const unsigned int expected)
  {
#ifdef KOKKOS_ENABLE_THREADS
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace, Kokkos::Threads>)
      CHECK(Kokkos::DefaultHostExecutionSpace().concurrency() == expected);
    else
      SUCCEED("Kokkos Threads is enabled but is not the default host execution space");
#else
    (void)expected;
    SUCCEED("Kokkos Threads is not enabled");
#endif
  }
} // namespace

TEST_CASE("Init separates the application and Kokkos host thread counts", "[common][init][threads][kokkos]")
{
  const std::string filename = write_config("init_threads_separate", R"({"threads":4,"kokkos_threads":1})");

  DiFfRG::Init init(filename);

  CHECK(dealii::MultithreadInfo::n_threads() == 4);
  REQUIRE(Kokkos::is_initialized());
  check_threads_host_concurrency(1);
  std::filesystem::remove(filename);
}

TEST_CASE("Init preserves the historical Kokkos host thread count by default", "[common][init][threads][kokkos]")
{
  const std::string filename = write_config("init_threads_default", R"({"threads":4})");

  DiFfRG::Init init(filename);

  CHECK(dealii::MultithreadInfo::n_threads() == 4);
  REQUIRE(Kokkos::is_initialized());
  check_threads_host_concurrency(4);
  std::filesystem::remove(filename);
}

TEST_CASE("A zero Kokkos host thread count uses the resolved application count", "[common][init][threads][kokkos]")
{
  const std::string filename = write_config("init_threads_zero", R"({"threads":4,"kokkos_threads":0})");

  DiFfRG::Init init(filename);

  CHECK(dealii::MultithreadInfo::n_threads() == 4);
  REQUIRE(Kokkos::is_initialized());
  check_threads_host_concurrency(4);
  std::filesystem::remove(filename);
}

TEST_CASE("Init applies command-line thread overrides before Kokkos initialization",
          "[common][init][threads][kokkos][cli]")
{
  const std::string filename = write_config("init_threads_cli", R"({"threads":2,"kokkos_threads":2})");
  char program[] = "init_threads";
  char parameter_flag[] = "-p";
  char threads_flag[] = "-si";
  char threads_override[] = "/discretization/threads=4";
  char kokkos_flag[] = "-si";
  char kokkos_override[] = "/discretization/kokkos_threads=1";
  char *argv[] = {program,     parameter_flag, const_cast<char *>(filename.c_str()), threads_flag, threads_override,
                  kokkos_flag, kokkos_override};

  DiFfRG::Init init(7, argv);

  CHECK(dealii::MultithreadInfo::n_threads() == 4);
  REQUIRE(Kokkos::is_initialized());
  check_threads_host_concurrency(1);
  std::filesystem::remove(filename);
}

TEST_CASE("DEAL_II_NUM_THREADS caps both application and Kokkos host threads",
          "[common][init][threads][kokkos][environment]")
{
  const std::string filename = write_config("init_threads_environment", R"({"threads":4,"kokkos_threads":3})");
  REQUIRE(::setenv("DEAL_II_NUM_THREADS", "2", 1) == 0);

  DiFfRG::Init init(filename);

  CHECK(dealii::MultithreadInfo::n_threads() == 2);
  REQUIRE(Kokkos::is_initialized());
  check_threads_host_concurrency(2);
  CHECK(::unsetenv("DEAL_II_NUM_THREADS") == 0);
  std::filesystem::remove(filename);
}
