#pragma once

// DiFfRG
#include <DiFfRG/common/mpi.hh>

// std
#include <cstddef>
#include <cstdint>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief This rank's window into the external grid of one QuadratureIntegrator::map() call.
   */
  struct MapSlice {
    /// First external grid point this rank computes.
    size_t offset = 0;
    /// Number of grid points; 0 means this rank does not participate in this map().
    size_t count = 0;

    bool owns_all(const size_t grid_size) const { return offset == 0 && count == grid_size; }
  };

  /**
   * @brief Decides, without any user input, which rank computes which part of each map().
   *
   * ## What is split
   *
   * Only the **external** coordinate grid of a `map()` -- never a quadrature reduction. Every
   * individual integral is therefore still summed entirely on one rank, in the original order, so
   * the result is **bitwise identical to a serial run at any rank count**. That is not a nicety:
   * the flow kernels feed a stiff DAE (see common/tbb.hh), where a last-bit change in a residual
   * changes the accepted step sequence and hence the whole trajectory. It also makes rebalancing
   * numerically free -- moving a partition boundary changes *which* rank computes a point, never
   * *what* it computes.
   *
   * Splitting is only applied to one-dimensional coordinate systems. `SubCoordinates` derives its
   * per-axis window from `from_linear_index()`, which is exact for `dim == 1` and not for higher
   * dimensions; a multi-dimensional grid is therefore assigned whole to a single rank.
   *
   * ## How the split is chosen
   *
   * Per call, with `G` external grid points and a quadrature volume `Q = prod(grid_size)`:
   *
   *     S = G * Q                                   (kernel evaluations in this map)
   *     r = clamp(S / quantum, 1, min(n_ranks, G))  (how many ranks to spread it over)
   *     owners = the r least-loaded ranks
   *
   * `quantum` is a **GPU fill threshold**, not a launch-overhead threshold: below roughly 5e4
   * evaluations these register-heavy kernels do not occupy a modern device anyway, so splitting
   * further buys nothing and costs a launch plus a gather. Deriving it from launch cost (~1e3
   * evaluations) would oversplit by two orders of magnitude.
   *
   * The practical consequence is the one that matters for load balance: a **cheap flow is assigned
   * whole** to the least-loaded rank rather than chopped into slivers, so a block of many small
   * flows spreads across ranks instead of every rank doing a sliver of every flow.
   *
   * ## Determinism
   *
   * Every rank runs the same program and issues the same sequence of `map()` calls, so a schedule
   * that is a pure function of that sequence is identical on all ranks without any communication.
   * The running per-rank load is part of that state and is reset by `complete()`. A plan checksum
   * is verified inside `complete()`; a mismatch aborts the job rather than hanging it.
   */
  class MapScheduler
  {
  public:
    static MapScheduler &instance();

    /// Whether there is more than one rank to schedule over. False in a serial run, which makes
    /// every method below a no-op and the whole feature zero-overhead.
    bool active() const { return m_n_ranks > 1; }

    uint rank() const { return m_rank; }
    uint n_ranks() const { return m_n_ranks; }
    MPI_Comm comm() const { return m_comm; }

    /**
     * @brief Register one map() call and return this rank's slice of it.
     *
     * Called by every rank for every map(), **including ranks that end up computing nothing** --
     * the entry with `count == 0` is what keeps the collective in `complete()` matched. Deciding
     * from rank-local state instead is the one way to deadlock this design.
     *
     * @param integrator_id stable, construction-order id of the calling integrator
     * @param dest base of the caller's destination array (unoffset)
     * @param elem_size sizeof one result element
     * @param grid_size number of external grid points, G
     * @param quadrature_volume points per integral, Q
     * @param splittable whether this coordinate system may be windowed (see class docs)
     */
    MapSlice schedule(size_t integrator_id, void *dest, size_t elem_size, size_t grid_size, size_t quadrature_volume,
                      bool splittable);

    /// Whether the open plan already contains a map() from this integrator. Used in place of a
    /// rank-local "is a result still staged" test, which ranks would answer differently.
    bool plan_contains(size_t integrator_id) const;

    bool has_open_plan() const { return !m_plan.empty(); }

    /**
     * @brief Exchange every slice registered since the last completion.
     *
     * One `MPI_Allgatherv` for the whole batch. Called by MapCompletion::flush() *after* the device
     * results have landed in host memory.
     */
    void complete();

    /// Discard the open plan without communicating and poison the scheduler, so that the next
    /// collective aborts with a diagnosis instead of hanging. See DeferredMaps' destructor.
    void poison(const char *reason);

    void set_quantum(double quantum);
    double quantum() const { return m_quantum; }
    void set_verbose(bool verbose) { m_verbose = verbose; }

  protected:
    MapScheduler();

    /**
     * @brief Bind to an explicit rank and rank count instead of MPI_COMM_WORLD.
     *
     * For tests that simulate several ranks inside one process: the schedule, the byte accounting
     * and the packing are all pure functions of the call sequence, so they can be exercised without
     * an MPI build. Only the single Allgatherv in between cannot.
     */
    MapScheduler(uint rank, uint n_ranks);

    /// Verify agreement, compute the byte accounting and pack this rank's slices. False if there is
    /// nothing to exchange, in which case the batch is already closed.
    bool prepare_batch();
    /// Land every other rank's slices out of the receive buffer, then close the batch.
    void finish_batch();

    // Grow-only scratch, so a flush does not allocate. Protected rather than private so the
    // multi-rank simulation in the tests can play the part of the network.
    std::vector<char> m_send;
    std::vector<char> m_recv;
    std::vector<int> m_counts;
    std::vector<int> m_displs;

  private:
    struct Entry {
      size_t integrator_id;
      void *dest;
      size_t elem_size;
      size_t grid_size;
      /// The r owning ranks, ascending. Slice j of the grid belongs to owners[j].
      std::vector<uint> owners;
    };

    /// Canonical partition: slice j of r covers [part(G, r, j), part(G, r, j + 1)).
    static size_t part(size_t grid_size, size_t r, size_t j) { return (j * grid_size) / r; }

    std::vector<uint> least_loaded(size_t r) const;
    uint64_t plan_hash() const;
    void log_plan() const;

    MPI_Comm m_comm;
    uint m_rank = 0;
    uint m_n_ranks = 1;

    double m_quantum = 5e4;
    bool m_quantum_pinned = false;
    bool m_verbose = false;
    bool m_logged = false;
    const char *m_poisoned = nullptr;

    std::vector<Entry> m_plan;
    /// Running kernel-evaluation count per rank within the open plan; reset by complete().
    std::vector<double> m_load;
    /// Simulated instances (see the rank/n_ranks constructor) never communicate.
    bool m_simulated = false;
  };
} // namespace DiFfRG
