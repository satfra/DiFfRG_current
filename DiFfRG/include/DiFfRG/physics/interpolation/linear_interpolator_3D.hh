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
    using value_type = NT;
    static constexpr size_t dim = 3;

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
      Kokkos::View<const NT2 ***, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          in_view(in_data, sizes[0], sizes[1], sizes[2]);
      update(in_view);
    }

    template <typename View>
      requires Kokkos::is_view<View>::value
    void update(const View &view)
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator2D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");
      // Populate host mirror
      Kokkos::deep_copy(host_data, view);
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
      size_t x1 = static_cast<size_t>(
          min(ceil(idx_x + static_cast<decltype(idx_x)>(1e-16)), static_cast<decltype(idx_x)>(sizes[0] - 1)));
      size_t y1 = static_cast<size_t>(
          min(ceil(idx_y + static_cast<decltype(idx_y)>(1e-16)), static_cast<decltype(idx_y)>(sizes[1] - 1)));
      size_t z1 = static_cast<size_t>(
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

    auto &CPU() const { return get_on<CPU_memory>(); }
    auto &GPU() const { return get_on<GPU_memory>(); }

    template <typename MemorySpace> auto &get_on() const
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
        if (other_instance == nullptr) {
          other_instance = std::make_shared<LinearInterpolator3D<NT, Coordinates, MemorySpace>>(coordinates);
          other_instance->other_instance = std::shared_ptr<std::decay_t<decltype(*this)>>(
              const_cast<std::decay_t<decltype(*this)> *>(this), [](std::decay_t<decltype(*this)> *) {});
        }
        // Copy the data from the current instance to the new one
        other_instance->update(host_data);
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

    NT *data()
    {
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator1D: You probably called data() on a copied instance. This is not allowed. "
            "You need to call data() on the original instance.");
      return host_data.data();
    }

    friend class LinearInterpolator3D<NT, Coordinates, other_memory_space>;

  private:
    const Coordinates coordinates;
    const device::array<size_t, 3> sizes;
    const size_t total_size;

    using ViewType = Kokkos::View<NT ***, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType device_data;
    HostViewType host_data;

    mutable std::shared_ptr<LinearInterpolator3D<NT, Coordinates, other_memory_space>> other_instance;
  };
} // namespace DiFfRG