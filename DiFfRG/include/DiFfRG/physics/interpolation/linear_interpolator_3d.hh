#pragma once

// DiFfRG
#include "DiFfRG/common/utils.hh"
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/interpolation/interpolation_stencil.hh>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 3D data, callable from host AND device code.
   *
   * See LinearInterpolator1D for the host/device dispatch rationale.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class LinearInterpolator3D
  {
    static_assert(Coordinates::dim == 3, "LinearInterpolator3D requires 3D coordinates");

    static constexpr bool periodic_x = is_periodic_axis_v<Coordinates, 0>;
    static constexpr bool periodic_y = is_periodic_axis_v<Coordinates, 1>;
    static constexpr bool periodic_z = is_periodic_axis_v<Coordinates, 2>;

    using ViewType = Kokkos::View<NT ***, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::host_mirror_type;

    static constexpr bool has_separate_device =
        !std::is_same_v<typename ViewType::memory_space, typename HostViewType::memory_space>;

  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 3;

    /**
     * @brief Construct a LinearInterpolator3D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()),
          device_data("LinearInterpolator3D_data", coordinates.sizes()[0], coordinates.sizes()[1],
                      coordinates.sizes()[2]),
          host_data(Kokkos::create_mirror_view(device_data))
    {
    }

    /// Shallow copy of BOTH views, valid in host and in device code. See LinearInterpolator1D.
    KOKKOS_DEFAULTED_FUNCTION LinearInterpolator3D(const LinearInterpolator3D &) = default;

    /**
     * @brief Replace the data, leaving host AND device current. The only mutator.
     *
     * `in_data` is row-major; see LinearInterpolator2D::update() for why the fill indexes the
     * mirror through operator() instead of copying flat, and LinearInterpolator1D::update() for
     * why the single trailing fence is not optional.
     */
    template <typename NT2> void update(const NT2 *in_data)
    {
      for (size_t i = 0; i < sizes[0]; ++i)
        for (size_t j = 0; j < sizes[1]; ++j)
          for (size_t k = 0; k < sizes[2]; ++k)
            host_data(i, j, k) = in_data[(i * sizes[1] + j) * sizes[2] + k];

      if constexpr (has_separate_device) {
        typename ViewType::execution_space exec;
        Kokkos::deep_copy(exec, device_data, host_data);
        exec.fence();
      }
    }

    /**
     * @brief Host-side element access, in the row-major order update() takes its input in.
     */
    NT operator[](size_t i) const
    {
      return host_data(i / (sizes[1] * sizes[2]), (i / sizes[2]) % sizes[1], i % sizes[2]);
    }

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
    device::array<ctype, 3> KOKKOS_FUNCTION index(const ctype &x, const ctype &y, const ctype &z) const
    {
      return coordinates.backward(x, y, z);
    }

    /**
     * @brief Interpolate at grid indices previously obtained from index().
     */
    NT KOKKOS_FUNCTION at(const device::array<ctype, 3> &idx) const
    {
      const auto idx_x = idx[0];
      const auto idx_y = idx[1];
      const auto idx_z = idx[2];
      // Clamped [i, i+1] stencil on bounded axes, wrapping [i, (i+1) % n] on periodic ones
      const auto sx = make_interpolation_stencil<periodic_x>(idx_x, sizes[0]);
      const auto sy = make_interpolation_stencil<periodic_y>(idx_y, sizes[1]);
      const auto sz = make_interpolation_stencil<periodic_z>(idx_z, sizes[2]);

      const size_t x0 = sx.lower, x1 = sx.upper;
      const size_t y0 = sy.lower, y1 = sy.upper;
      const size_t z0 = sz.lower, z1 = sz.upper;

      const auto corner000 = value(x0, y0, z0);
      const auto corner010 = value(x0, y1, z0);
      const auto corner100 = value(x1, y0, z0);
      const auto corner110 = value(x1, y1, z0);
      const auto corner001 = value(x0, y0, z1);
      const auto corner011 = value(x0, y1, z1);
      const auto corner101 = value(x1, y0, z1);
      const auto corner111 = value(x1, y1, z1);

      const auto tx = sx.t;
      const auto ty = sy.t;
      const auto tz = sz.t;

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(
            tx,
            Kokkos::fma(ty, Kokkos::fma(tz, corner111, Kokkos::fma(-tz, corner110, corner110)),
                        (1 - ty) * Kokkos::fma(tz, corner101, Kokkos::fma(-tz, corner100, corner100))),
            (1 - tx) * Kokkos::fma(ty, Kokkos::fma(tz, corner011, Kokkos::fma(-tz, corner010, corner010)),
                                   (1 - ty) * Kokkos::fma(tz, corner001, Kokkos::fma(-tz, corner000, corner000))));
      else
        return corner000 * (1 - tx) * (1 - ty) * (1 - tz) + corner001 * (1 - tx) * (1 - ty) * tz +
               corner010 * (1 - tx) * ty * (1 - tz) + corner011 * (1 - tx) * ty * tz +
               corner100 * tx * (1 - ty) * (1 - tz) + corner101 * tx * (1 - ty) * tz +
               corner110 * tx * ty * (1 - tz) + corner111 * tx * ty * tz;
    }

    /**
     * @brief Interpolate the data at a given point.
     */
    NT KOKKOS_FUNCTION operator()(const ctype &x, const ctype &y, const ctype &z) const
    {
      return at(index(x, y, z));
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
     * NOTE the storage order is NOT the row-major order update() takes; see LinearInterpolator2D.
     */
    const NT *data() const { return host_data.data(); }

  private:
    /// Read one element from whichever buffer belongs to the executing side.
    KOKKOS_FORCEINLINE_FUNCTION NT value(const size_t i, const size_t j, const size_t k) const
    {
      KOKKOS_IF_ON_DEVICE((return device_data(i, j, k);))
      KOKKOS_IF_ON_HOST((return host_data(i, j, k);))
    }

    const Coordinates coordinates;
    const device::array<size_t, 3> sizes;

    ViewType device_data;
    HostViewType host_data;
  };
} // namespace DiFfRG
