#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 2D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename MemorySpace, typename NT, typename Coordinates> class LinearInterpolator2D
  {
    static_assert(Coordinates::dim == 2, "LinearInterpolator2D requires 2D coordinates");

  public:
    using memory_space = MemorySpace;

    /**
     * @brief Construct a LinearInterpolator2D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator2D(const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()), total_size(sizes[0] * sizes[1])
    {
      // Allocate Kokkos View
      device_data = ViewType("LinearInterpolator2D_data", sizes[0], sizes[1]);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
    }
    /**
     * @brief Construct a LinearInterpolator2D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param size size of the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator2D(const NT *in_data, const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()), total_size(sizes[0] * sizes[1])
    {
      // Allocate Kokkos View
      device_data = ViewType("LinearInterpolator2D_data", sizes[0], sizes[1]);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
      // Update device data
      update(in_data);
    }

    template <typename NT2> void update(const NT2 *in_data)
    {
      // Populate host mirror
      for (uint i = 0; i < sizes[0]; ++i)
        for (uint j = 0; j < sizes[1]; ++j)
          host_data(i, j) = static_cast<NT>(in_data[i * sizes[1] + j]);
      // Copy data to device
      Kokkos::deep_copy(device_data, host_data);
    }

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    NT KOKKOS_INLINE_FUNCTION operator()(const typename Coordinates::ctype x, const typename Coordinates::ctype y) const
    {
      using Kokkos::min, Kokkos::max;
      auto [idx_x, idx_y] = coordinates.backward(x, y);
      idx_x = max(static_cast<decltype(idx_x)>(0), min(idx_x, static_cast<decltype(idx_x)>(sizes[0] - 1)));
      idx_y = max(static_cast<decltype(idx_y)>(0), min(idx_y, static_cast<decltype(idx_y)>(sizes[1] - 1)));

      // Clamp the (upper) index to the range [1, sizes - 1]
      uint x1 = static_cast<uint>(min(ceil(idx_x + static_cast<decltype(idx_x)>(1e-16)), static_cast<decltype(idx_x)>(sizes[0] - 1)));
      uint y1 = static_cast<uint>(min(ceil(idx_y + static_cast<decltype(idx_y)>(1e-16)), static_cast<decltype(idx_y)>(sizes[1] - 1)));

      const auto corner00 = device_data(x1 - 1, y1 - 1);
      const auto corner01 = device_data(x1 - 1, y1);
      const auto corner10 = device_data(x1, y1 - 1);
      const auto corner11 = device_data(x1, y1);

      const auto tx = x1 - idx_x;
      const auto ty = y1 - idx_y;

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(ty, Kokkos::fma(tx, corner00, Kokkos::fma(-tx, corner10, corner10)),
                           (1 - ty) * Kokkos::fma(tx, corner01, Kokkos::fma(-tx, corner11, corner11)));
      else
        return corner00 * tx * ty + corner01 * tx * (1 - ty) + corner10 * (1 - tx) * ty +
               corner11 * (1 - tx) * (1 - ty);
    }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const Coordinates coordinates;
    const std::array<uint, 2> sizes;
    const uint total_size;

    using ViewType = Kokkos::View<NT **, MemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType device_data;
    HostViewType host_data;
  };
} // namespace DiFfRG