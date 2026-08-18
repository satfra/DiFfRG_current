#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/interpolation/interpolation_stencil.hh>

// std
#include <cstring>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 1D data, callable from host AND device code.
   *
   * The object owns one device-resident allocation and its host mirror, and update() leaves both
   * current. There is therefore no memory-space template parameter and no lazily built twin in the
   * other space: at() simply reads whichever buffer belongs to the side it is executing on.
   *
   * That dispatch cannot be an `if constexpr`. "Am I on the device" is a property of the
   * compilation pass, not of a template argument, and only __CUDA_ARCH__ carries it. Under nvcc
   * with a GNU host compiler KOKKOS_IF_ON_DEVICE/KOKKOS_IF_ON_HOST are a plain preprocessor
   * selection (Kokkos_Macros.hpp; the NV_IF_TARGET spelling there is gated on
   * KOKKOS_COMPILER_NVHPC, which this build does not use), so each pass compiles exactly one body
   * and each pass ends in a return. This is NOT the nvcc extended-lambda `if constexpr` miscompile
   * documented in quadrature_integrator.hh -- that one is about lambda bodies.
   *
   * If the dispatch is ever wrong the failure is loud rather than silent: View::operator() runs
   * runtime_check_memory_access_violation, which Kokkos::abort()s with "attempt to access
   * inaccessible memory space". See tests/physics/interpolation/host_device_dispatch.cc.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class LinearInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "LinearInterpolator1D requires 1D coordinates");

    static constexpr bool periodic = is_periodic_coordinate_v<Coordinates>;

    using ViewType = Kokkos::View<NT *, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::host_mirror_type;

    // Without a distinct device space create_mirror_view aliases the source: the two members are
    // then one allocation, and the upload must not be compiled at all so a host-only build pays no
    // fence.
    static constexpr bool has_separate_device =
        !std::is_same_v<typename ViewType::memory_space, typename HostViewType::memory_space>;

  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 1;

    /**
     * @brief Construct a LinearInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator1D(const Coordinates &coordinates)
        : coordinates(coordinates), size(coordinates.size()),
          device_data("LinearInterpolator1D_data", coordinates.size()),
          host_data(Kokkos::create_mirror_view(device_data))
    {
    }

    /**
     * @brief Shallow copy of BOTH views, valid in host and in device code.
     *
     * Carrying the HostSpace mirror into a device closure is exactly what Kokkos::DualView does:
     * the view tracker force-disables reference counting on the device side, so the host pointer
     * is copied and never dereferenced there. Defaulted rather than user-defined, because a
     * host-only copy constructor makes capture into a KOKKOS_LAMBDA ill-formed.
     */
    KOKKOS_DEFAULTED_FUNCTION LinearInterpolator1D(const LinearInterpolator1D &) = default;

    /**
     * @brief Replace the data, leaving host AND device current. The only mutator.
     *
     * The mirror is filled by a plain host copy rather than a Kokkos::deep_copy. A deep_copy
     * dispatches onto an execution space and would have to be fenced before the H2D below could be
     * enqueued; a memcpy/loop is complete when it returns. That is what leaves exactly one fence in
     * this function.
     *
     * The trailing fence is NOT optional: it is what lets the caller refill host_data on the next
     * update() without racing an in-flight H2D copy. Dropping it requires double-buffering the
     * host staging array.
     *
     * The H2D names an execution space instance. The space-less deep_copy brackets EVERY copy in a
     * global Kokkos::fence() that synchronizes all execution space instances of all enabled
     * backends -- measured at ~11.4 cudaDeviceSynchronize calls per copy, i.e. ~98k device-wide
     * barriers in a single 425-RHS YangMills solve. This copy only needs to be ordered against the
     * kernels that read it, which share this execution space instance.
     */
    template <typename NT2> void update(const NT2 *in_data)
    {
      if constexpr (std::is_same_v<NT, NT2> && std::is_trivially_copyable_v<NT>)
        std::memcpy(host_data.data(), in_data, size * sizeof(NT));
      else
        for (size_t i = 0; i < size; ++i)
          host_data(i) = in_data[i];

      if constexpr (has_separate_device) {
        typename ViewType::execution_space exec;
        Kokkos::deep_copy(exec, device_data, host_data);
        exec.fence();
      }
    }

    /**
     * @brief Host-side element access. Always valid, including on a copy.
     */
    NT operator[](size_t i) const { return host_data(i); }

    /**
     * @brief Map a physical coordinate onto the grid index.
     *
     * Split out for the same reason as in SplineInterpolator1D: for a logarithmic or focused-log
     * axis Coordinates::backward is a fp64 log/log1p costing ~200 fp64 instructions, and a kernel
     * evaluating several dressings at one momentum otherwise pays it once per dressing (the
     * compiler cannot CSE it, since each interpolator owns its own coordinate members).
     *
     * Depends only on the coordinate system, so the result may be shared across interpolators that
     * share one -- stencil resolution and clamping stay in at().
     */
    typename Coordinates::ctype KOKKOS_FUNCTION index(const typename Coordinates::ctype x) const
    {
      return coordinates.backward(x);
    }

    /**
     * @brief Interpolate at a grid index previously obtained from index().
     */
    NT KOKKOS_FUNCTION at(const typename Coordinates::ctype idx) const
    {
      // Resolve the stencil: clamped [i, i+1] for a bounded axis, wrapping [i, (i+1) % size] for a periodic one
      const auto stencil = make_interpolation_stencil<periodic>(idx, size);
      const auto t = stencil.t;
      const auto lower = value(stencil.lower);
      const auto upper = value(stencil.upper);
      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower));
      else
        return t * upper + (1 - t) * lower;
    }

    /**
     * @brief Interpolate the data at a given point.
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype x) const { return at(index(x)); }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

    /**
     * @brief Read-only handle to the host values.
     *
     * Deliberately const: there is no public way to push a host-side edit to the device, so a
     * write handle could only ever desynchronize the two. Route mutations through update().
     */
    const NT *data() const { return host_data.data(); }

  private:
    /**
     * @brief Read one element from whichever buffer belongs to the executing side.
     */
    KOKKOS_FORCEINLINE_FUNCTION NT value(const size_t i) const
    {
      KOKKOS_IF_ON_DEVICE((return device_data(i);))
      KOKKOS_IF_ON_HOST((return host_data(i);))
    }

    const Coordinates coordinates;
    const size_t size;

    ViewType device_data;
    HostViewType host_data;
  };
} // namespace DiFfRG
