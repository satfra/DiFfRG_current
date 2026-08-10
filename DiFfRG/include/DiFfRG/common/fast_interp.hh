#pragma once

// std
#include <atomic>

namespace DiFfRG
{
  /**
   * @brief Process-global switch for the tabulated inverse coordinate transform in the
   * interpolators ("fastInterpLookups").
   *
   * When enabled, SplineInterpolator1D builds a per-cell polynomial inverse table at construction
   * and its index() dispatches to the tabulated transform (~50 device ops) instead of the analytic
   * backward() (a fp64 log1p for LogarithmicCoordinates1D, log+asinh for FocusedLogCoordinates1D,
   * ~155-400 fp64 ops). The tabulated transform reproduces the analytic one to <= 1e-12 absolute
   * in index units (verified against a dense sweep at construction), which sits three to six
   * orders below the spline interpolation error itself — but it is NOT bit-identical, which is
   * why it is an explicit runtime opt-in.
   *
   * Set from the parameter file key `/discretization/fastInterpLookups` (default false) by
   * ConfigurationHelper, i.e. before any model — and hence any interpolator — is constructed.
   * Interpolators sample the flag ONCE at construction; flipping it afterwards only affects
   * interpolators constructed later.
   */
  inline std::atomic<bool> &fast_interp_lookups_flag()
  {
    static std::atomic<bool> flag{false};
    return flag;
  }

  inline void set_fast_interp_lookups(const bool enabled) { fast_interp_lookups_flag().store(enabled); }
  inline bool fast_interp_lookups() { return fast_interp_lookups_flag().load(); }
} // namespace DiFfRG
