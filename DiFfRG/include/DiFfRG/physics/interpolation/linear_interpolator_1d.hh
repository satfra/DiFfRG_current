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
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 1;

    /**
     * @brief Construct a LinearInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator1D(const Coordinates &coordinates)
        : coordinates(coordinates), size(coordinates.size())
    {
      // Allocate Kokkos View
      device_data = ViewType("LinearInterpolator1D_data", size);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
    }

    /**
     * @brief Copy constructor for LinearInterpolator1D. This is ONLY for usage inside Kokkos parallel loops.
     *
     */
    KOKKOS_FUNCTION
    LinearInterpolator1D(const LinearInterpolator1D &other)
        : coordinates(other.coordinates), size(other.size)
    {
      // Use the same data
      device_data = other.device_data;
    }

    KOKKOS_FUNCTION ~LinearInterpolator1D()
    {
      KOKKOS_IF_ON_HOST((if (owns_other_instance) delete other_instance;))
    }

    template <typename NT2> void update(const NT2 *in_data)
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator1D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");

      Kokkos::View<const NT2 *, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          in_view(in_data, size);
      update(in_view);
    }

    template <typename View>
      requires Kokkos::is_view<View>::value
    void update(const View &view)
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator1D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");
      // Populate host mirror
      Kokkos::deep_copy(host_data, view);
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
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype x) const
    {
      auto idx = coordinates.backward(x);
      // Clamp the index to the range [0, size - 1]
      idx = Kokkos::max(static_cast<decltype(idx)>(0), Kokkos::min(idx, static_cast<decltype(idx)>(size - 1)));
      // t is the fractional part of the index
      const auto t = idx - Kokkos::floor(idx);
      // Do the linear interpolation, clamping upper index to valid range
      const size_t lower_idx = Kokkos::min(size_t(Kokkos::floor(idx)), size - 2);
      const size_t upper_idx = lower_idx + 1;
      const auto lower = device_data[lower_idx];
      const auto upper = device_data[upper_idx];
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

    auto &CPU() const { return get_on<CPU_memory>(); }
    auto &GPU() const { return get_on<GPU_memory>(); }

    template <typename MemorySpace> auto &get_on() const
    {
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator1D: You probably called get_on() on a copied instance. This is not allowed. "
            "You need to call get_on() on the original instance.");

      if constexpr (std::is_same_v<MemorySpace, DefaultMemorySpace>) {
        // remove constness
        return const_cast<LinearInterpolator1D<NT, Coordinates, MemorySpace> &>(*this);
      } else {
        // Create a new instance with the same data but in the requested memory space
        if (other_instance == nullptr) {
          other_instance = new LinearInterpolator1D<NT, Coordinates, MemorySpace>(coordinates);
          owns_other_instance = true;
          other_instance->other_instance = const_cast<std::decay_t<decltype(*this)> *>(this);
        }
        // Copy the data from the current instance to the new one
        other_instance->update(host_data);
        // Return the new instance
        return *other_instance;
      }
    }

    NT *data()
    {
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator1D: You probably called data() on a copied instance. This is not allowed. "
            "You need to call data() on the original instance.");
      return host_data.data();
    }

    friend class LinearInterpolator1D<NT, Coordinates, other_memory_space>;

  private:
    const Coordinates coordinates;
    const size_t size;

    using ViewType = Kokkos::View<NT *, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::host_mirror_type;

    ViewType device_data;
    HostViewType host_data;

    mutable LinearInterpolator1D<NT, Coordinates, other_memory_space> *other_instance = nullptr;
    mutable bool owns_other_instance = false;
  };
} // namespace DiFfRG