#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <limits>

namespace DiFfRG
{
  /**
   * @brief A stack of 1D splines, callable from host AND device code.
   *
   * Nearest-neighbour in the stack axis, spline in the data axis. See LinearInterpolator1D for the
   * host/device dispatch rationale.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class SplineInterpolator1DStack
  {
    static_assert(Coordinates::dim == 2, "SplineInterpolator1DStack requires 2D coordinates");
    // The spline coefficients come from a non-cyclic tridiagonal solve with boundary conditions at the grid edges,
    // which cannot close a periodic axis. Use LinearInterpolator2D there instead.
    static_assert(!is_periodic_axis_v<Coordinates, 0> && !is_periodic_axis_v<Coordinates, 1>,
                  "SplineInterpolator1DStack does not support periodic coordinates; use LinearInterpolator2D.");

    // SoA layout: separate views for values and spline coefficients, for coalesced GPU access
    using ValueViewType = Kokkos::View<NT **, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using CoeffViewType = ValueViewType;
    using HostValueViewType = typename ValueViewType::host_mirror_type;
    using HostCoeffViewType = typename CoeffViewType::host_mirror_type;

    static constexpr bool has_separate_device =
        !std::is_same_v<typename ValueViewType::memory_space, typename HostValueViewType::memory_space>;

  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 2;

    /**
     * @brief Construct a SplineInterpolator1DStack with zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1DStack(const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()),
          device_values("SplineInterpolator1DStack_values", coordinates.sizes()[0], coordinates.sizes()[1]),
          device_coeffs("SplineInterpolator1DStack_coeffs", coordinates.sizes()[0], coordinates.sizes()[1]),
          host_values(Kokkos::create_mirror_view(device_values)),
          host_coeffs(Kokkos::create_mirror_view(device_coeffs))
    {
    }

    /// Shallow copy of ALL views, valid in host and in device code. See LinearInterpolator1D.
    KOKKOS_DEFAULTED_FUNCTION SplineInterpolator1DStack(const SplineInterpolator1DStack &) = default;

    /**
     * @brief Replace the data, leaving host AND device current. The only mutator.
     *
     * `in_data` is row-major; the fill indexes the mirror through operator(), which keeps that
     * contract independent of the mirror's own layout. See LinearInterpolator1D::update() for why
     * the host fill is a plain loop and why the single trailing fence is not optional.
     */
    template <typename NT2>
    void update(const NT2 *in_data, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Copy values from input data (row-major)
      for (size_t i = 0; i < sizes[0]; ++i)
        for (size_t j = 0; j < sizes[1]; ++j)
          host_values(i, j) = in_data[i * sizes[1] + j];

      // Build the spline coefficients
      for (size_t i = 0; i < sizes[0]; ++i)
        build_y2(i, lower_y1, upper_y1);

      if constexpr (has_separate_device) {
        typename ValueViewType::execution_space exec;
        Kokkos::deep_copy(exec, device_values, host_values);
        Kokkos::deep_copy(exec, device_coeffs, host_coeffs);
        exec.fence();
      }
    }

    /**
     * @brief Host-side element access, in the row-major order update() takes its input in.
     */
    NT operator[](size_t i) const { return host_values(i / sizes[1], i % sizes[1]); }

    /**
     * @brief Map physical coordinates onto grid indices.
     *
     * Split out of operator() because it is the expensive half: Coordinates::backward is a fp64
     * log/log1p per logarithmic axis, ~200 fp64 instructions each on current NVIDIA parts against
     * ~10 for the interpolation. A generated kernel evaluating several dressings at the SAME point
     * otherwise pays it once per dressing -- the compiler cannot CSE it, because each interpolator
     * owns its own `coordinates` members and cannot be proven to agree with another's.
     *
     * Depends only on the coordinate system, so the result may be shared across interpolators that
     * share one. Clamping and stencil resolution stay in at(), since they are size dependent and
     * would otherwise make an index untransferable between interpolators of different extent.
     *
     * The return type is spelled out rather than deduced: nvcc loses `decltype(var)` when the
     * initializer has a deduced return type inside a class-template member, and the consumer stores
     * this in a `const auto`.
     */
    device::array<typename Coordinates::ctype, 2> KOKKOS_FUNCTION
    index(const typename Coordinates::ctype s, const typename Coordinates::ctype x) const
    {
      // backward() returns device::array for CoordinatePackND but std::tuple<Idx, NT> for the
      // combined finite-T systems; the structured binding normalises both onto what at() expects.
      auto [sidx, xidx] = coordinates.backward(s, x);
      return {{static_cast<typename Coordinates::ctype>(sidx), static_cast<typename Coordinates::ctype>(xidx)}};
    }

    /**
     * @brief Interpolate at grid indices previously obtained from index().
     */
    NT KOKKOS_FUNCTION at(const device::array<typename Coordinates::ctype, 2> &raw) const
    {
      auto _sidx = raw[0];
      auto xidx = raw[1];
      // Clamp indices to the range [0, sizes[i] - 1]
      xidx = Kokkos::max(static_cast<decltype(xidx)>(0), Kokkos::min(xidx, static_cast<decltype(xidx)>(sizes[1] - 1)));
      _sidx =
          Kokkos::max(static_cast<decltype(_sidx)>(0), Kokkos::min(_sidx, static_cast<decltype(_sidx)>(sizes[0] - 1)));
      // for the x part
      const size_t lidx = Kokkos::min(size_t(Kokkos::floor(xidx)), sizes[1] - 2);
      const size_t uidx = lidx + 1;
      // t is the fractional part of the index
      const ctype t = xidx - lidx;

      // the s part is rounded to the nearest integer
      const size_t sidx = size_t(Kokkos::round(_sidx));

      const NT lower = value(sidx, lidx);
      const NT upper = value(sidx, uidx);
      const NT cl = coeff(sidx, lidx);
      const NT cu = coeff(sidx, uidx);

      const ctype tm1 = t - 1;
      const NT cubic = t * tm1 * ((t + 1) * cl - (t - 2) * cu);

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower)) + cubic; // linear + cubic
      else
        return t * upper + (1 - t) * lower + cubic; // linear + cubic
    }

    /**
     * @brief Interpolate the data at a given point.
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype s,
                                  const typename Coordinates::ctype x) const
    {
      return at(index(s, x));
    }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

    /**
     * @brief Read-only handle to the host values, in the mirror's storage order.
     *
     * NOTE the storage order is NOT the row-major order update() takes: the device view is pinned
     * to GPU_memory, hence LayoutLeft. Use operator[] for row-major access.
     */
    const NT *data() const { return host_values.data(); }

  private:
    /// Read one value from whichever buffer belongs to the executing side.
    KOKKOS_FORCEINLINE_FUNCTION NT value(const size_t i, const size_t j) const
    {
      KOKKOS_IF_ON_DEVICE((return device_values(i, j);))
      KOKKOS_IF_ON_HOST((return host_values(i, j);))
    }

    /// Read one spline coefficient from whichever buffer belongs to the executing side.
    KOKKOS_FORCEINLINE_FUNCTION NT coeff(const size_t i, const size_t j) const
    {
      KOKKOS_IF_ON_DEVICE((return device_coeffs(i, j);))
      KOKKOS_IF_ON_HOST((return host_coeffs(i, j);))
    }

    const Coordinates coordinates;
    const device::array<size_t, 2> sizes;

    ValueViewType device_values;
    CoeffViewType device_coeffs;
    HostValueViewType host_values;
    HostCoeffViewType host_coeffs;

    void build_y2(const size_t sidx, const ctype lower_y1, const ctype upper_y1)
    {
      const auto &size = sizes[1];

      NT p, qn, sig, un;
      std::vector<NT> u(size - 1);

      if (!std::isfinite(lower_y1) || lower_y1 >= std::numeric_limits<ctype>::max() / 2)
        host_coeffs(sidx, 0) = u[0] = 0.0;
      else {
        host_coeffs(sidx, 0) = -0.5;
        u[0] = 3.0 * ((host_values(sidx, 1) - host_values(sidx, 0)) - lower_y1);
      }
      for (size_t i = 1; i < size - 1; i++) {
        sig = 0.5;
        p = sig * host_coeffs(sidx, i - 1) + 2.0;
        host_coeffs(sidx, i) = (sig - 1.0) / p;
        u[i] = (host_values(sidx, i + 1) - host_values(sidx, i)) - (host_values(sidx, i) - host_values(sidx, i - 1));
        u[i] = (6.0 * u[i] / 2. - sig * u[i - 1]) / p;
      }
      if (!std::isfinite(upper_y1) || upper_y1 >= std::numeric_limits<ctype>::max() / 2)
        qn = un = 0.0;
      else {
        qn = 0.5;
        un = 3.0 * (upper_y1 - (host_values(sidx, size - 1) - host_values(sidx, size - 2)));
      }
      host_coeffs(sidx, size - 1) = (un - qn * u[size - 2]) / (qn * host_coeffs(sidx, size - 2) + 1);
      for (int k = size - 2; k >= 0; k--)
        host_coeffs(sidx, k) = host_coeffs(sidx, k) * host_coeffs(sidx, k + 1) + u[k];

      // Precompute division by 6 so operator() avoids per-call divides
      for (size_t k = 0; k < size; ++k)
        host_coeffs(sidx, k) /= (ctype)6;
    }
  };
} // namespace DiFfRG
