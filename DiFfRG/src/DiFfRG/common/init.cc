#include <DiFfRG/common/init.hh>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

// standard libraries
#include <cstdlib>
#include <iostream>

// external libraries
#include <deal.II/base/kokkos.h>
#include <deal.II/base/multithread_info.h>
#include <tbb/tbb.h>

namespace DiFfRG
{
  void set_thread_limit(const unsigned int threads)
  {
    // invalid_unsigned_int is deal.II's "decide for me", which resolves to the core count.
    dealii::MultithreadInfo::set_thread_limit(threads == 0 ? dealii::numbers::invalid_unsigned_int : threads);
  }

  void set_thread_limit(const ConfigTree &config) { set_thread_limit(config.get_uint("/discretization/threads", 0)); }

  bool Init::initialized = false;

  dealii::InitFinalize *Init::mpi_initialization = nullptr;

  Init::Init(int argc, char *argv[], const std::string parameter_file)
      : argc(argc), argv(argv), parameter_file(parameter_file)
  {
    if (!initialized) {
      // The thread count has to be known *before* the libraries come up: dealii::InitFinalize
      // forwards it to MultithreadInfo::set_thread_limit(), which installs the process-wide
      // tbb::global_control that caps every TBB pipeline in deal.II and DiFfRG. Kokkos also reads
      // MultithreadInfo::n_threads() exactly once below, so both thread counts must be available
      // here. probe() therefore takes an early, silent look at the parameter file; the
      // application's own ConfigurationHelper does the authoritative parse.
      const ConfigTree config = ConfigurationHelper::probe(argc, argv, parameter_file);
      const unsigned int configured_threads = config.get_uint("/discretization/threads", 0);
      const unsigned int configured_kokkos_threads = config.get_uint("/discretization/kokkos_threads", 0);

      if (config.contains("/discretization/threads") && !config.contains("/discretization/mesh_workers"))
        std::cerr << "WARNING: '/discretization/threads' has changed meaning. It used to set the length of the\n"
                     "         assembly pipeline; that is now '/discretization/mesh_workers' (default 8).\n"
                     "         It now sets the number of CPU threads the whole program may use, so this run is\n"
                     "         limited to "
                  << configured_threads << " threads. Set '/discretization/threads' to 0 to use all available cores.\n"
                  << std::endl;

      // A configured count is handed to deal.II, which takes the minimum with DEAL_II_NUM_THREADS.
      // 0 means "keep whatever is already in force", i.e. the machine's core count unless the
      // application installed a tbb::global_control of its own before constructing us.
      const unsigned int max_num_threads =
          configured_threads > 0 ? configured_threads
                                 : tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);

      // We initialize every library that deal.II's MPI_InitFinalize would, *except* Kokkos, and then
      // hand the Kokkos lifecycle to deal.II via ensure_kokkos_initialized() below. The reason is a
      // teardown-ordering bug:
      //
      // MPI_InitFinalize initializes Kokkos by calling Kokkos::initialize() directly, which leaves
      // dealii::internal::dealii_initialized_kokkos == false. With that flag false, deal.II's
      // GrowingVectorMemory installs a Kokkos finalize-hook (release_unused_memory) that dereferences
      // a function-local-static memory pool. Because we finalize Kokkos from a std::atexit handler,
      // that hook runs *after* the pool static has already been destroyed during normal static
      // teardown -> dereference of freed memory -> intermittent segfault on program exit.
      //
      // ensure_kokkos_initialized() instead sets dealii_initialized_kokkos = true (so the dangerous
      // finalize-hook is never installed) and registers Kokkos::finalize() itself via std::atexit.
      // The vector-memory pool is then emptied by its own (~Pool) destructor while Kokkos is still
      // alive, and Kokkos is finalized afterwards.
      mpi_initialization = new dealii::InitFinalize(
          argc, argv,
          dealii::InitializeLibrary::MPI | dealii::InitializeLibrary::SLEPc | dealii::InitializeLibrary::PETSc |
              dealii::InitializeLibrary::Zoltan | dealii::InitializeLibrary::P4EST,
          max_num_threads);

      // InitFinalize has now resolved the application limit, including DEAL_II_NUM_THREADS. Keep
      // that resolved value so Kokkos can be initialized with an independent host-backend count
      // without changing the TBB/deal.II limit observed by the rest of the application. An absent
      // kokkos_threads entry and an explicit zero both retain the historical application count.
      const unsigned int application_threads = dealii::MultithreadInfo::n_threads();
      const unsigned int kokkos_threads =
          configured_kokkos_threads > 0 ? configured_kokkos_threads : application_threads;

      std::atexit([]() {
        if (initialized) {
          // Finalizes MPI/PETSc/... (but not Kokkos, which deal.II finalizes via its own atexit hook
          // registered by ensure_kokkos_initialized()).
          delete Init::mpi_initialization;
          initialized = false;
        }
      });

      // Initialize Kokkos *after* registering the teardown above so that Kokkos::finalize() (also
      // registered via std::atexit, here) runs before the MPI finalization, matching the order
      // deal.II's own ~MPI_InitFinalize uses. deal.II sizes the Kokkos host backend from the
      // current MultithreadInfo limit, so apply the requested Kokkos count only for that call and
      // then restore the resolved application/TBB count. set_thread_limit() continues to apply
      // DEAL_II_NUM_THREADS as an upper bound to both values.
      dealii::MultithreadInfo::set_thread_limit(kokkos_threads);
      try {
        dealii::internal::ensure_kokkos_initialized();
      } catch (...) {
        dealii::MultithreadInfo::set_thread_limit(application_threads);
        throw;
      }
      dealii::MultithreadInfo::set_thread_limit(application_threads);

      initialized = true;
    }
  }

  Init::Init(const std::string parameter_file) : Init(0, nullptr, parameter_file) {}

  const ConfigurationHelper Init::get_configuration_helper() const
  {
    ConfigurationHelper configuration_helper(argc, argv, parameter_file);
    return configuration_helper;
  }

  bool Init::is_initialized() { return initialized; }
} // namespace DiFfRG
