#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

// std
#include <cstring>
#include <exception>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Deferred landing of QuadratureIntegrator::map() results in host memory.
   *
   * Why this exists. `map()` used to finish with
   *
   *     Kokkos::deep_copy(space, dest_view, dest_device_view);
   *
   * where `dest_view` wraps caller memory — a `dealii::Vector` element range, i.e. ordinary
   * *pageable* host memory. A device-to-pageable `cudaMemcpyAsync` is not actually asynchronous:
   * the driver has to stage it, so the call blocks until the copy (and therefore the kernels
   * feeding it) has completed. Measured on a YangMills solve: `cudaMemcpyAsync` totalled 15.933 s
   * against 15.966 s of kernel time — the host sat inside that copy for essentially the entire
   * GPU execution. The consequence is that there is **zero host run-ahead**: the GPU drains after
   * every one of the ~3400 map() calls, and all host preparation for the next flow is exposed as
   * GPU idle time. On a part where the kernels are slow (an RTX 4070) that costs a few percent;
   * on one where they are ~6x faster (an A100) it is over half the wall clock.
   *
   * The fix is to copy into a *pinned* staging buffer instead, which really is asynchronous, and
   * to defer the final pinned->caller memcpy until someone needs the numbers. Integrators register
   * that pending copy here; `flush()` fences once and lands all of them together.
   *
   * Deferral is **opt-in and scoped**, via DeferredMaps. Outside such a scope `map()` keeps its
   * original contract — the result is in `dest` when it returns — so every existing caller (tests,
   * Examples, downstream models) is unaffected. That matters because the failure mode of a silent
   * contract change here is not a compile error but wrong numbers.
   *
   * Inside a DeferredMaps scope the caller promises not to read any `dest` until the scope ends
   * (or it calls flush_maps() itself). Use it around a run of *independent* flows.
   *
   * Not thread safe: the flow evaluation path is single-threaded (the parallelism lives inside the
   * Kokkos kernels), and this registry is only touched from it.
   */
  class MapCompletion
  {
  public:
    struct PendingCopy {
      void *dst;
      const void *src;
      size_t bytes;
    };

    /**
     * @brief Register a device->host result that still has to be copied from staging into `dst`.
     */
    static void record(void *dst, const void *src, const size_t bytes)
    {
      pending().push_back(PendingCopy{dst, src, bytes});
    }

    /**
     * @brief Fence, land every pending map result, then exchange slices between MPI ranks.
     *
     * Always fences, even with nothing pending, so this is a drop-in replacement for the
     * `Kokkos::fence()` it supersedes at the end of a residual evaluation.
     *
     * The MPI step is driven by MapScheduler's plan, **not** by the local pending list. A rank that
     * owns no slice of a given map records no pending copy, so a pending-list-driven decision would
     * have some ranks enter the collective and others skip it -- a silent hang. See MapScheduler.
     */
    static void flush()
    {
      Kokkos::fence("DiFfRG::MapCompletion::flush");
      auto &p = pending();
      for (const auto &c : p)
        std::memcpy(c.dst, c.src, c.bytes);
      p.clear();
      MapScheduler::instance().complete();
    }

    /**
     * @brief Drop every pending copy without performing it, and abandon the MPI plan.
     *
     * Only for stack unwinding, where the destinations may already be gone.
     */
    static void discard(const char *reason)
    {
      pending().clear();
      MapScheduler::instance().poison(reason);
    }

    /**
     * @brief Whether `src` already has an unlanded result queued.
     *
     * An integrator reuses one staging buffer, so a second map() from the same integrator before a
     * flush would overwrite the first result. The integrator asks this and flushes first.
     */
    static bool has_pending(const void *src)
    {
      for (const auto &c : pending())
        if (c.src == src) return true;
      return false;
    }

    /**
     * @brief Whether the caller has opened a DeferredMaps scope.
     */
    static bool deferral_enabled() { return deferral(); }
    static void set_deferral(const bool enabled) { deferral() = enabled; }

  private:
    static std::vector<PendingCopy> &pending()
    {
      static std::vector<PendingCopy> p;
      return p;
    }
    static bool &deferral()
    {
      static bool enabled = false;
      return enabled;
    }
  };

  /**
   * @brief Land all outstanding map() results. See MapCompletion.
   */
  inline void flush_maps() { MapCompletion::flush(); }

  /**
   * @brief Scope in which map() results are landed lazily instead of one blocking copy per call.
   *
   * Wrap a run of flows that do not read each other's results:
   *
   *     {
   *       DeferredMaps defer;
   *       flows.ZA4.map(&residual[idxv("ZA4")], coords, args);
   *       flows.ZA3.map(&residual[idxv("ZA3")], coords, args);
   *       ...
   *     } // one fence here, then all results land
   *
   * Inside the scope the host keeps issuing work instead of blocking on each result, so the GPU
   * does not drain between flows. Reading any of the destinations before the scope closes is a
   * use-before-write bug; nest a flush_maps() if a value is genuinely needed early.
   *
   * Non-reentrant by design: a nested scope would flush at the inner closing brace and surprise
   * the outer one, so nesting is not supported.
   */
  class DeferredMaps
  {
  public:
    DeferredMaps() { MapCompletion::set_deferral(true); }
    ~DeferredMaps()
    {
      MapCompletion::set_deferral(false);
      // Leaving via an exception means this rank abandons the batch while the others are still
      // filling theirs. Running the collective here would deadlock; poisoning the scheduler turns
      // the next flush into a diagnosable abort instead. Under MPI an exception thrown out of a
      // flow block is not recoverable anyway -- the ranks have already diverged.
      if (std::uncaught_exceptions() > 0) {
        MapCompletion::discard("an exception was thrown inside a DeferredMaps scope");
        return;
      }
      MapCompletion::flush();
    }
    DeferredMaps(const DeferredMaps &) = delete;
    DeferredMaps &operator=(const DeferredMaps &) = delete;
  };
} // namespace DiFfRG
