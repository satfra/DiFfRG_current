#pragma once

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>

// external libraries
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/mpi.h>

namespace DiFfRG
{
  /**
   * @brief Limit the number of CPU threads this process may use.
   *
   * Sets deal.II's thread limit, which installs the process-wide tbb::global_control capping
   * every TBB pipeline in deal.II and DiFfRG. A value of 0 means "all available cores".
   *
   * @note If the DEAL_II_NUM_THREADS environment variable is set, the smaller of the two wins.
   *
   * @note DiFfRG::Init uses /discretization/threads for this application/TBB limit and may
   * temporarily use /discretization/kokkos_threads while initializing Kokkos. Calling this
   * afterwards still re-caps TBB, but the Kokkos host backend keeps the thread count it was
   * created with.
   *
   * Applications go through DiFfRG::Init and need not call this. It exists for tests and for
   * embedders that drive the library without Init.
   */
  void set_thread_limit(const unsigned int threads);

  /**
   * @brief Limit the number of CPU threads from a configuration tree.
   *
   * Reads /discretization/threads, defaulting to 0 (all available cores).
   */
  void set_thread_limit(const ConfigTree &config);

  class Init
  {
  public:
    /**
     * Initialize deal.II's supported libraries and the process-global Kokkos runtime.
     *
     * /discretization/threads controls the persistent TBB/deal.II thread limit. A positive
     * /discretization/kokkos_threads independently sizes Kokkos's host backend; if the entry is
     * absent or zero, Kokkos uses the resolved application thread count for backward
     * compatibility. DEAL_II_NUM_THREADS remains an upper bound for both counts.
     */
    Init(int argc, char *argv[], const std::string parameter_file = "parameter.json");
    Init(const std::string parameter_file = "parameter.json");

    const ConfigurationHelper get_configuration_helper() const;

    static bool is_initialized();

  private:
    static bool initialized;

    int argc;
    char **argv;
    std::string parameter_file;
    static dealii::InitFinalize *mpi_initialization;
  };
} // namespace DiFfRG
