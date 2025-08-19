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
  template <typename NT, typename Coordinates, typename DefaultMemorySpace = CPU_memory> class LinearInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "LinearInterpolator1D requires 1D coordinates");

  public:
    using memory_space = DefaultMemorySpace;
    using other_memory_space = other_memory_space_t<DefaultMemorySpace>;

    /**
     * @brief Construct a LinearInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator1D(const Coordinates &coordinates)
        : coordinates(coordinates), size(coordinates.size()), other_instance(nullptr)
    {
      // Allocate Kokkos View
      device_data = ViewType("LinearInterpolator1D_data", size);
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
        : coordinates(coordinates), size(coordinates.size()), other_instance(nullptr)
    {
      // Allocate Kokkos View
      device_data = ViewType("LinearInterpolator1D_data", size);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
      // Update device data
      update(in_data);
    }

    /**
     * @brief Copy constructor for LinearInterpolator1D. This is ONLY for usage inside Kokkos parallel loops.
     *
     */
    KOKKOS_FUNCTION
    LinearInterpolator1D(const LinearInterpolator1D &other)
        : coordinates(other.coordinates), size(other.size), other_instance(nullptr)
    {
      // Use the same data
      device_data = other.device_data;
    }

    template <typename NT2> void update(const NT2 *in_data)
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator1D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");
      // Populate host mirror
      for (uint i = 0; i < size; ++i)
        host_data[i] = static_cast<NT>(in_data[i]);
      // Copy data to device
      Kokkos::deep_copy(device_data, host_data);
    }

    NT operator[](size_t i) const
    {
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator1D: You probably called operator[]() on a copied instance. This is not allowed. "
            "You need to call operator[]() on the original instance.");
      return host_data.data()[i]; // Access the host data directly
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

    template <typename MemorySpace> auto &get_on()
    {
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator1D: You probably called get_on() on a copied instance. This is not allowed. "
            "You need to call get_on() on the original instance.");

      if constexpr (std::is_same_v<MemorySpace, DefaultMemorySpace>) {
        return *this; // Return the current instance if the memory space matches
      } else {
        // Create a new instance with the same data but in the requested memory space
        if (other_instance == nullptr)
          other_instance = std::make_shared<LinearInterpolator1D<NT, Coordinates, MemorySpace>>(coordinates);
        // Copy the data from the current instance to the new one
        other_instance->update(host_data.data());
        // Return the new instance
        return *other_instance;
      }
    }

  private:
    const Coordinates coordinates;
    const uint size;

    using ViewType = Kokkos::View<NT *, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType device_data;
    HostViewType host_data;

    std::shared_ptr<LinearInterpolator1D<NT, Coordinates, other_memory_space>> other_instance;
  };
} // namespace DiFfRG