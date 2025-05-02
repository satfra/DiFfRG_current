#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 1D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename MemorySpace, typename NT, typename Coordinates> class LinearInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "LinearInterpolator1D requires 1D coordinates");

  public:
    using memory_space = MemorySpace;

    /**
     * @brief Construct a LinearInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator1D(const Coordinates &coordinates) : coordinates(coordinates), size(coordinates.size())
    {
      // Allocate Kokkos View
      device_data = Kokkos::View<NT *, MemorySpace>("LinearInterpolator1D_data", size);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
    }

    /**
     * @brief Construct a LinearInterpolator1D object from a pointer to data and a coordinate system.
     *
     * @param in_data pointer to the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator1D(const NT *in_data, const Coordinates &coordinates)
        : coordinates(coordinates), size(coordinates.size())
    {
      // Allocate Kokkos View
      device_data = Kokkos::View<NT *, MemorySpace>("LinearInterpolator1D_data", size);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
      // Update device data
      update(in_data);
    }

    template <typename NT2> void update(const NT2 *in_data)
    {
      // Populate host mirror
      for (uint i = 0; i < size; ++i)
        host_data[i] = static_cast<NT>(in_data[i]);
      // Copy data to device
      Kokkos::deep_copy(device_data, host_data);
    }

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    NT KOKKOS_INLINE_FUNCTION operator()(const typename Coordinates::ctype x) const
    {
      auto idx = coordinates.backward(x);
      // Clamp the index to the range [0, size - 1]
      idx = Kokkos::max(static_cast<decltype(idx)>(0), Kokkos::min(idx, static_cast<decltype(idx)>(size - 1)));
      // t is the fractional part of the index
      const auto t = idx - Kokkos::floor(idx);
      // Do the linear interpolation
      const auto lower = device_data[uint(Kokkos::floor(idx))];
      const auto upper = device_data[uint(Kokkos::ceil(idx))];
      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower));
      else
        return t * upper + (1 - t) * lower;
    }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const Coordinates coordinates;
    const uint size;

    Kokkos::View<NT *, MemorySpace> device_data;
    Kokkos::View<NT *, MemorySpace>::HostMirror host_data;
  };
} // namespace DiFfRG