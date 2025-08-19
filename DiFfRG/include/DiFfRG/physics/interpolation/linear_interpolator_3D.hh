#pragma once

// DiFfRG
#include "DiFfRG/common/utils.hh"
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 3D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates, typename DefaultMemorySpace = CPU_memory> class LinearInterpolator3D
  {
    static_assert(Coordinates::dim == 3, "LinearInterpolator3D requires 3D coordinates");

  public:
    using memory_space = DefaultMemorySpace;
    using other_memory_space = other_memory_space_t<DefaultMemorySpace>;
    using ctype = typename Coordinates::ctype;

    /**
     * @brief Construct a LinearInterpolator3D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()), total_size(sizes[0] * sizes[1] * sizes[2]),
          other_instance(nullptr)
    {
      // Allocate Kokkos View
      device_data = ViewType("LinearInterpolator3D_data", sizes[0], sizes[1], sizes[2]);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
    }
    /**
     * @brief Construct a LinearInterpolator3D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param size size of the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const NT *in_data, const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()), total_size(sizes[0] * sizes[1] * sizes[2]),
          other_instance(nullptr)
    {
      // Allocate Kokkos View
      device_data = ViewType("LinearInterpolator3D_data", sizes[0], sizes[1], sizes[2]);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
      // Update device data
      update(in_data);
    }

    KOKKOS_FUNCTION
    LinearInterpolator3D(const LinearInterpolator3D &other)
        : coordinates(other.coordinates), sizes(other.sizes), total_size(other.total_size), other_instance(nullptr)
    {
      // Use the same data
      device_data = other.device_data;
    }

    template <typename NT2> void update(const NT2 *in_data)
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator3D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");
      // Populate host mirror
      for (uint i = 0; i < sizes[0]; ++i)
        for (uint j = 0; j < sizes[1]; ++j)
          for (uint k = 0; k < sizes[2]; ++k)
            host_data(i, j, k) = static_cast<NT>(in_data[i * sizes[1] * sizes[2] + j * sizes[2] + k]);
      // Copy data to device
      Kokkos::deep_copy(device_data, host_data);
    }

    NT operator[](size_t i) const
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator3D: You probably called operator[]() on a copied instance. This is not allowed. "
            "You need to call operator[]() on the original instance.");
      return host_data.data()[i]; // Access the host data directly
    }

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    NT KOKKOS_INLINE_FUNCTION operator()(const ctype &x, const ctype &y, const ctype &z) const
    {
      using Kokkos::min, Kokkos::max;
      auto [idx_x, idx_y, idx_z] = coordinates.backward(x, y, z);
      idx_x = max(static_cast<decltype(idx_x)>(0), min(idx_x, static_cast<decltype(idx_x)>(sizes[0] - 1)));
      idx_y = max(static_cast<decltype(idx_y)>(0), min(idx_y, static_cast<decltype(idx_y)>(sizes[1] - 1)));
      idx_z = max(static_cast<decltype(idx_z)>(0), min(idx_z, static_cast<decltype(idx_z)>(sizes[2] - 1)));

      // Clamp the (upper) index to the range [1, sizes - 1]
      uint x1 = static_cast<uint>(
          min(ceil(idx_x + static_cast<decltype(idx_x)>(1e-16)), static_cast<decltype(idx_x)>(sizes[0] - 1)));
      uint y1 = static_cast<uint>(
          min(ceil(idx_y + static_cast<decltype(idx_y)>(1e-16)), static_cast<decltype(idx_y)>(sizes[1] - 1)));
      uint z1 = static_cast<uint>(
          min(ceil(idx_z + static_cast<decltype(idx_z)>(1e-16)), static_cast<decltype(idx_z)>(sizes[2] - 1)));

      const auto corner000 = device_data(x1 - 1, y1 - 1, z1 - 1);
      const auto corner010 = device_data(x1 - 1, y1, z1 - 1);
      const auto corner100 = device_data(x1, y1 - 1, z1 - 1);
      const auto corner110 = device_data(x1, y1, z1 - 1);
      const auto corner001 = device_data(x1 - 1, y1 - 1, z1);
      const auto corner011 = device_data(x1 - 1, y1, z1);
      const auto corner101 = device_data(x1, y1 - 1, z1);
      const auto corner111 = device_data(x1, y1, z1);

      const auto tx = x1 - idx_x;
      const auto ty = y1 - idx_y;
      const auto tz = z1 - idx_z;

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(
            tx,
            Kokkos::fma(ty, Kokkos::fma(tz, corner000, Kokkos::fma(-tz, corner001, corner001)),
                        (1 - ty) * Kokkos::fma(tz, corner010, Kokkos::fma(-tz, corner011, corner011))),
            (1 - tx) * Kokkos::fma(ty, Kokkos::fma(tz, corner100, Kokkos::fma(-tz, corner101, corner101)),
                                   (1 - ty) * Kokkos::fma(tz, corner110, Kokkos::fma(-tz, corner111, corner111))));
      else
        return corner000 * tx * ty * tz + corner001 * tx * ty * (1 - tz) + corner010 * tx * (1 - ty) * tz +
               corner011 * tx * (1 - ty) * (1 - tz) + corner100 * (1 - tx) * ty * tz +
               corner101 * (1 - tx) * ty * (1 - tz) + corner110 * (1 - tx) * (1 - ty) * tz +
               corner111 * (1 - tx) * (1 - ty) * (1 - tz);
    }

    template <typename MemorySpace> auto &get_on()
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator3D: You probably called get_on[]() on a copied instance. This is not allowed. "
            "You need to call get_on[]() on the original instance.");

      if constexpr (std::is_same_v<MemorySpace, DefaultMemorySpace>) {
        return *this; // Return the current instance if the memory space matches
      } else {
        // Create a new instance with the same data but in the requested memory space
        if (other_instance == nullptr)
          other_instance = std::make_shared<LinearInterpolator3D<NT, Coordinates, MemorySpace>>(coordinates);
        // Copy the data from the current instance to the new one
        other_instance->update(host_data.data());
        // Return the new instance
        return *other_instance;
      }
    }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const Coordinates coordinates;
    const device::array<uint, 3> sizes;
    const uint total_size;

    using ViewType = Kokkos::View<NT ***, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType device_data;
    HostViewType host_data;

    std::shared_ptr<LinearInterpolator3D<NT, Coordinates, other_memory_space>> other_instance;
  };
} // namespace DiFfRG