#include <DiFfRG/common/init.hh>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

// standard libraries
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// external libraries
#include <deal.II/base/kokkos.h>
#include <deal.II/base/multithread_info.h>
#include <tbb/tbb.h>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef __linux__
#include <sched.h>
#endif

namespace DiFfRG
{
  namespace
  {
    /// Length of the PCI bus id string CUDA returns, e.g. "0000:65:00.0".
    constexpr int bus_id_len = 16;

    /**
     * @brief This rank's CPU budget: how many threads it may start, and why.
     */
    struct CpuBudget {
      unsigned int threads;
      /// Node-local ranks competing for the same CPUs, this rank included. 1 == exclusive.
      unsigned int sharing;
      /// Whether the launcher handed this rank its own CPUs.
      bool pinned;
    };

    /**
     * @brief Work out how many CPU threads this rank may start.
     *
     * The naive version of this -- take the machine's core count and divide by the number of
     * node-local ranks -- is wrong on exactly the systems that matter, and wrong in both directions:
     *
     *  - `std::thread::hardware_concurrency()`, which is what deal.II's `MultithreadInfo::n_cores()`
     *    calls, **ignores the CPU affinity mask**. Measured on this machine: pinned to 8 of 32 CPUs
     *    with taskset, it still reports 32. So a rank pinned to its own cores does not know it.
     *  - Conversely, when a launcher *has* pinned each rank (Slurm `--cpus-per-task`), TBB's
     *    `default_concurrency()` does see the mask (8 in the same test), so the thread limit is
     *    already per-rank -- and dividing it again by the node-local rank count halves it a second
     *    time. On a 128-core node split into 2 pinned ranks that yields 32 threads each, not 64.
     *
     * So the divisor is not "how many ranks are on this node" but "how many node-local ranks are
     * competing for *my* CPUs". That is answered by intersecting the affinity masks, the same way
     * GPU sharing is answered by comparing bus ids rather than counting devices.
     */
    CpuBudget cpu_budget([[maybe_unused]] const unsigned int local_size,
                         const unsigned int configured_threads)
    {
      // An explicit /discretization/threads is a statement about one process. Honour it verbatim:
      // silently dividing it would make the setting mean "threads per node", which is neither what
      // it says nor what it means in a serial run.
      if (configured_threads > 0) return CpuBudget{configured_threads, 1, false};

#ifdef __linux__
      cpu_set_t mask;
      CPU_ZERO(&mask);
      if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
        return CpuBudget{std::max(1u, dealii::MultithreadInfo::n_threads() / std::max(1u, local_size)), local_size,
                         false};

      const unsigned int my_cpus = static_cast<unsigned int>(CPU_COUNT(&mask));

      unsigned int sharing = 1;
      if (local_size > 1) {
        // Gather every node-local rank's mask and count how many overlap mine. Disjoint masks mean
        // the launcher already partitioned the node; identical masks mean the ranks share it.
        MPI_Comm node_comm = MPI::split_shared(MPI_COMM_WORLD);
        const size_t bytes = sizeof(cpu_set_t);
        std::vector<char> all(bytes * local_size, '\0');
        std::vector<int> counts(local_size, static_cast<int>(bytes));
        std::vector<int> displs(local_size);
        for (unsigned int i = 0; i < local_size; ++i)
          displs[i] = static_cast<int>(i * bytes);
        MPI::allgatherv_bytes(node_comm, &mask, bytes, all.data(), counts, displs);
        MPI::free_comm(node_comm);

        const char *mine = reinterpret_cast<const char *>(&mask);
        sharing = 0;
        for (unsigned int p = 0; p < local_size; ++p) {
          const char *other = all.data() + p * bytes;
          bool overlaps = false;
          for (size_t b = 0; b < bytes && !overlaps; ++b)
            overlaps = (mine[b] & other[b]) != 0;
          if (overlaps) ++sharing;
        }
        if (sharing == 0) sharing = 1; // cannot happen: a mask always overlaps itself
      }

      // Cap by whatever limit is already in force (DEAL_II_NUM_THREADS, an application's own
      // global_control), so this only ever lowers the budget.
      const unsigned int budget = std::max(1u, my_cpus / sharing);
      return CpuBudget{std::min(budget, dealii::MultithreadInfo::n_threads()), sharing, sharing == 1 && local_size > 1};
#else
      // No portable way to read an affinity mask; fall back to an even split of the node.
      return CpuBudget{std::max(1u, dealii::MultithreadInfo::n_threads() / std::max(1u, local_size)), local_size,
                       false};
#endif
    }

    /**
     * @brief Give this rank its own GPU, and warn loudly if it has to share one.
     *
     * Without this every rank on a node lands on device 0, which is the single biggest thing
     * standing between a multi-rank launch and actual multi-GPU execution.
     *
     * Sharing is detected by comparing the devices' **PCI bus ids**, not by counting. Under Slurm's
     * `--gpus-per-task=1` every rank sees exactly one device and `local_rank % n_devices == 0` is
     * the correct answer; a count-based check would report sharing on every well-configured
     * cluster. It is a warning rather than an error because a shared-GPU run is still correct, just
     * slower -- and a scaling study has to be able to measure that case.
     *
     * @return the device id this rank should use, or -1 for "no choice to make".
     */
    int select_device([[maybe_unused]] const unsigned int local_rank, [[maybe_unused]] const unsigned int local_size)
    {
#ifdef KOKKOS_ENABLE_CUDA
      int n_devices = 0;
      if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices <= 0) return -1;

      const int device = static_cast<int>(local_rank % static_cast<unsigned int>(n_devices));
      if (local_size <= 1) return device;

      char bus_id[bus_id_len] = {};
      if (cudaDeviceGetPCIBusId(bus_id, bus_id_len, device) != cudaSuccess) return device;

      // Gather every node-local rank's bus id and look for duplicates. This is the only way to see
      // through CUDA_VISIBLE_DEVICES, which renumbers devices per process.
      MPI_Comm node_comm = MPI::split_shared(MPI_COMM_WORLD);
      std::vector<char> all_ids(static_cast<size_t>(local_size) * bus_id_len, '\0');
      std::vector<int> counts(local_size, bus_id_len);
      std::vector<int> displs(local_size);
      for (unsigned int i = 0; i < local_size; ++i)
        displs[i] = static_cast<int>(i * bus_id_len);
      MPI::allgatherv_bytes(node_comm, bus_id, bus_id_len, all_ids.data(), counts, displs);

      std::vector<unsigned int> sharing;
      for (unsigned int p = 0; p < local_size; ++p)
        if (std::strncmp(all_ids.data() + p * bus_id_len, bus_id, bus_id_len) == 0) sharing.push_back(p);

      if (sharing.size() > 1) {
        // stderr on every affected rank: timestepper progress is stdout-only and a cluster launcher
        // may well swallow one of the two streams.
        std::string ranks;
        for (size_t i = 0; i < sharing.size(); ++i)
          ranks += (i ? ", " : "") + std::to_string(sharing[i]);
        std::cerr << "\n"
                  << "================================ WARNING ================================\n"
                  << " GPU " << bus_id << " is shared by " << sharing.size() << " node-local ranks (" << ranks << ").\n"
                  << " The run is still correct, but those ranks time-slice the device and each\n"
                  << " costs several hundred MB of extra context, so it will NOT scale.\n"
                  << " Launch with one GPU per rank (e.g. Slurm --gpus-per-task=1), or set\n"
                  << " CUDA_VISIBLE_DEVICES per rank.\n"
                  << "=========================================================================\n"
                  << std::endl;
      }
      MPI::free_comm(node_comm);
      return device;
#else
      return -1;
#endif
    }
  } // namespace

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
      // tbb::global_control that caps every TBB pipeline in deal.II and DiFfRG, and which
      // ensure_kokkos_initialized() below reads to size the Kokkos host backend. Neither can be
      // lowered meaningfully afterwards. probe() therefore takes an early, silent look at the
      // parameter file; the application's own ConfigurationHelper does the authoritative parse.
      const ConfigTree config = ConfigurationHelper::probe(argc, argv, parameter_file);
      const unsigned int configured_threads = config.get_uint("/discretization/threads", 0);

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

      std::atexit([]() {
        if (initialized) {
          // Finalizes MPI/PETSc/... (but not Kokkos, which deal.II finalizes via its own atexit hook
          // registered by ensure_kokkos_initialized()).
          delete Init::mpi_initialization;
          initialized = false;
        }
      });

      // From here on MPI_Init has run, so the node-local rank is finally knowable -- which is why
      // none of the following could be folded into max_num_threads above.
      MPI_Comm node_comm = MPI::split_shared(MPI_COMM_WORLD);
      const unsigned int local_rank = MPI::rank(node_comm);
      const unsigned int local_size = MPI::size(node_comm);
      MPI::free_comm(node_comm);

      // Without a per-rank cap, N ranks each start a full-width TBB arena on the same cores and
      // spend their time fighting each other. cpu_budget() works out the right cap from the affinity
      // masks rather than by dividing a machine-wide count -- see its comment for why the obvious
      // version is wrong in both directions.
      const CpuBudget budget = cpu_budget(local_size, configured_threads);
      if (budget.threads != dealii::MultithreadInfo::n_threads()) {
        dealii::MultithreadInfo::set_thread_limit(budget.threads);
        // deal.II's limit governs its own pipelines; DiFfRG's TBB integrators need the arena capped
        // too, and that is a separate global_control.
        static tbb::global_control rank_parallelism(tbb::global_control::max_allowed_parallelism, budget.threads);
        (void)rank_parallelism;
      }
      if (local_size > 1 && MPI::rank(MPI_COMM_WORLD) == 0)
        std::cerr << "DiFfRG: " << local_size << " ranks per node, " << budget.threads
                  << " CPU threads each ("
                  << (configured_threads > 0
                          ? "from /discretization/threads, applied per rank"
                          : budget.pinned ? "the launcher pinned each rank to its own CPUs"
                                          : std::to_string(budget.sharing) + " ranks share one CPU set")
                  << ")." << std::endl;

      const int device = select_device(local_rank, local_size);

      // Initialize Kokkos *after* registering the teardown above so that Kokkos::finalize() (also
      // registered via std::atexit, here) runs before the MPI finalization, matching the order
      // deal.II's own ~MPI_InitFinalize uses.
      //
      // With more than one rank we cannot simply call ensure_kokkos_initialized(): it offers no way
      // to pick a device, and every rank would land on device 0. We therefore do exactly what it
      // would have done -- including setting dealii_initialized_kokkos and registering the atexit
      // hook, BOTH of which live inside its `if (!Kokkos::is_initialized())` branch and would
      // otherwise be skipped. Getting that wrong reintroduces the teardown segfault described
      // above: GrowingVectorMemory installs a Kokkos finalize-hook into an already-destroyed
      // static, and nothing ever finalizes Kokkos.
      if (device >= 0 && local_size > 1 && !Kokkos::is_initialized() && !Kokkos::is_finalized()) {
        dealii::internal::dealii_initialized_kokkos = true;
        Kokkos::initialize(Kokkos::InitializationSettings()
                               .set_device_id(device)
                               .set_num_threads(dealii::MultithreadInfo::n_threads()));
        std::atexit([]() { Kokkos::finalize(); });
      } else {
        // set_thread_limit() has already run (in InitFinalize), so ensure_kokkos_initialized()
        // picks up the correct thread count.
        dealii::internal::ensure_kokkos_initialized();
      }

      // The scheduler defaults are fine for most runs; expose them anyway, since the split width is
      // the one knob a scaling study may want to turn.
      MapScheduler::instance().set_quantum(config.get_double("/integration/map_quantum", 0.));
      if (config.get_bool("/integration/map_verbose", false)) MapScheduler::instance().set_verbose(true);

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