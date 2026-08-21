#pragma once

// external libraries
#include <deal.II/base/multithread_info.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/types.hh>

// std
#include <algorithm>
#include <cmath>
#include <optional>

namespace DiFfRG
{
  /**
   * @brief The two trailing arguments of dealii::MeshWorker::mesh_loop.
   *
   * Neither value can change a computed number: mesh_loop drives a tbb::parallel_pipeline whose
   * copier stage is serial and runs in iterator order, so the assembled result is bitwise
   * identical for any schedule. They are purely a performance choice, which is what makes it safe
   * to derive them rather than pin them per configuration.
   */
  struct AssemblySchedule {
    /// mesh_loop's queue_length: how many work items are live in the pipeline at once.
    uint queue_length = 1;
    /// mesh_loop's chunk_size: cells per work item, processed back-to-back by one worker.
    uint chunk_size = 1;

    friend bool operator==(const AssemblySchedule &, const AssemblySchedule &) = default;
  };

  /**
   * @brief Representative per-cell costs, in nanoseconds, for the loops DiFfRG assembles.
   *
   * These have been approximately measured on a Ryzen 9 7945HX - getting them more precise is probably not worth the
   * effort, as the approximate sizing effects are already well-captured.
   */
  namespace assembly_cost
  {
    /// Local FE work only: mass matrices, refinement indicators. ~0.2-0.3 us/cell for CG p=3.
    inline constexpr double local_fe = 250.;
    /// Algebraic couplings without momentum quadrature, e.g. LDG's level-to-level transfers.
    inline constexpr double algebraic = 1000.;
    /// A cell worker that evaluates the model's momentum integrals. ~2.7 us/cell for a KT
    /// residual, ~8.9 us/cell for its Jacobian.
    inline constexpr double momentum_integral = 3000.;
  } // namespace assembly_cost

  namespace assembly_schedule_defaults
  {
    /**
     * Work a single chunk should carry for the pipeline hand-off around it to be negligible. ~ 8 us
     */
    inline constexpr double chunk_work_ns = 8.0e3;

    /**
     * Work that has to exist before another pipeline worker pays for itself.
     *
     * Measured on some cheap loops, where waking workers on a small mesh is a net loss: 128 cells
     * are assembled fastest by a single worker, 512 by two, and only past a few thousand does it
     * pay to use the whole budget. ~ 64 us.
     */
    inline constexpr double worker_work_ns = 64.0e3;

    /// Ceiling on chunk_size. queue_length * chunk_size CopyData objects are allocated up front,
    /// so an unbounded chunk turns into allocation cost on a mesh that cannot use it.
    inline constexpr uint chunk_cap = 256;
  } // namespace assembly_schedule_defaults

  /**
   * @brief Overrides, if really wanted by the user: /discretization/{mesh_workers,batch_size}.
   *
   * Setting either pins every loop to these values. DiFfRG::Init warns about this.
   */
  struct AssemblyScheduleOverrides {
    std::optional<uint> queue_length;
    std::optional<uint> chunk_size;

    static AssemblyScheduleOverrides from_config(const ConfigTree &config)
    {
      AssemblyScheduleOverrides overrides;
      if (config.contains("/discretization/mesh_workers"))
        overrides.queue_length = config.get_uint("/discretization/mesh_workers");
      if (config.contains("/discretization/batch_size"))
        overrides.chunk_size = config.get_uint("/discretization/batch_size");
      return overrides;
    }
  };

  /**
   * @brief Derive one mesh_loop schedule from the cost of a cell.
   *
   * Both quantities follow from "how much work is this", in this order of priority:
   *   1. a work item must carry enough work that the pipeline hand-off around it is negligible,
   *      which fixes the chunk size, and
   *   2. subject to that, spread the remaining work over as many workers as it can pay for.
   *
   * Because (1) is a per-chunk property it holds at any mesh size, so a mesh too small to fill the
   * pipeline loses workers rather than chunk size. That is the right trade: the cores freed are
   * taken by the momentum integrators' own nested TBB work for the expensive loops, and for the
   * cheap ones a small mesh is measurably faster assembled serially than split up.
   *
   * @param n_local_cells cells this rank assembles, i.e. the length of locally_owned_cells().
   * @param cost_ns estimated cost of one cell, in nanoseconds; see namespace assembly_cost.
   */
  inline AssemblySchedule make_assembly_schedule(const uint n_local_cells, const uint n_threads,
                                                 const double cost_ns, const AssemblyScheduleOverrides &overrides = {})
  {
    using namespace assembly_schedule_defaults;

    const uint threads = std::max(1u, n_threads);
    const double cost = std::max(1.0, cost_ns);

    uint chunk =
        static_cast<uint>(std::clamp(std::llround(chunk_work_ns / cost), 1ll, static_cast<long long>(chunk_cap)));

    // Allocating past the end of the mesh is pure cost: mesh_loop reserves queue_length *
    // chunk_size CopyData objects whether or not there are cells to put in them.
    if (n_local_cells > 0) chunk = std::min(chunk, n_local_cells);
    chunk = std::max(1u, chunk);

    const uint n_chunks = (n_local_cells + chunk - 1) / chunk;
    const auto workers_paid_for = std::llround(double(n_local_cells) * cost / worker_work_ns);
    // Below the thread count, never at it: the integrators spawn their own TBB work from inside a
    // cell worker, in the same arena, and a pipeline holding every thread starves that nesting.
    const uint w_max = threads > 1 ? threads - 1 : 1u;

    uint queue = static_cast<uint>(std::clamp(workers_paid_for, 1ll, static_cast<long long>(w_max)));
    if (n_chunks > 0) queue = std::min(queue, n_chunks);

    if (overrides.queue_length) queue = *overrides.queue_length;
    if (overrides.chunk_size) chunk = *overrides.chunk_size;

    // deal.II only guards these with Assert, which is compiled out in Release, and a zero here is
    // undefined behaviour inside tbb::parallel_pipeline rather than a diagnostic.
    return {std::max(1u, queue), std::max(1u, chunk)};
  }

  /**
   * @brief The cells this rank assembles.
   *
   * mesh_loop visits exactly this range, so a work item carries only cells that do work and a
   * worker never has to re-check ownership. On a serial triangulation the filter accepts every
   * cell, so this is the full mesh and costs one trivial predicate per cell.
   */
  template <int dim> auto locally_owned_cells(const dealii::DoFHandler<dim> &dof_handler)
  {
    return dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                    dealii::IteratorFilters::LocallyOwnedCell());
  }

  /**
   * @brief How many cells locally_owned_cells() yields.
   *
   * Kept beside it on purpose: the schedule is sized for the range the loop walks, and the two
   * disagreeing would size the pipeline for work that is not there. Plain Triangulation has no
   * n_locally_owned_active_cells(), hence the dispatch.
   */
  template <typename Discretization> uint n_locally_owned_cells(const Discretization &discretization)
  {
    const auto &tria = discretization.get_triangulation();
    if constexpr (Discretization::Mesh::is_parallel)
      return static_cast<uint>(tria.n_locally_owned_active_cells());
    else
      return static_cast<uint>(tria.n_active_cells());
  }
} // namespace DiFfRG
