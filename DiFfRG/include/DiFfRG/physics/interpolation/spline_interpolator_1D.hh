#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <limits>

namespace DiFfRG
{
  /**
   * @brief A spline interpolator for 1D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates, typename DefaultMemorySpace = CPU_memory> class SplineInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "SplineInterpolator1D requires 1D coordinates");

  public:
    using memory_space = DefaultMemorySpace;
    using other_memory_space = other_memory_space_t<DefaultMemorySpace>;
    using ctype = typename Coordinates::ctype;
    using value_type = NT;

    /**
     * @brief Construct a SplineInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1D(const Coordinates &coordinates)
        : coordinates(coordinates), size(coordinates.size()), other_instance(nullptr)
    {
      // Allocate Kokkos View
      device_data = ViewType("SplineInterpolator1D_data", size);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
    }

    /**
     * @brief Construct a SplineInterpolator1D object from a pointer to data and a coordinate system.
     *
     * @param in_data pointer to the data
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1D(const NT *in_data, const Coordinates &coordinates,
                         const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                         const ctype upper_y1 = std::numeric_limits<ctype>::max())
        : coordinates(coordinates), size(coordinates.size()), other_instance(nullptr)
    {
      // Allocate Kokkos View
      device_data = ViewType("SplineInterpolator1D_data", size);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
      // Update device data
      update(in_data, lower_y1, upper_y1);
    }

    /**
     * @brief Copy constructor for SplineInterpolator1D. This is ONLY for usage inside Kokkos parallel loops.
     *
     */
    KOKKOS_FUNCTION
    SplineInterpolator1D(const SplineInterpolator1D &other)
        : coordinates(other.coordinates), size(other.size), other_instance(nullptr)
    {
      // Use the same data
      device_data = other.device_data;
    }

    template <typename NT2>
    void update(const NT2 *in_data, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");

      // Populate host mirror
      for (uint i = 0; i < size; ++i)
        host_data(i, 0) = static_cast<NT>(in_data[i]);
      // Build the spline coefficients
      build_y2(lower_y1, upper_y1);
      // Copy data to device
      Kokkos::deep_copy(device_data, host_data);
    }

    NT operator[](size_t i) const
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called operator[]() on a copied instance. This is not allowed. "
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
      ctype idx = coordinates.backward(x);
      // Clamp the index to the range [0, size - 1]
      idx = Kokkos::max(static_cast<decltype(idx)>(0), Kokkos::min(idx, static_cast<decltype(idx)>(size - 1)));
      const uint lidx = uint(Kokkos::floor(idx));
      const uint uidx = lidx + 1;
      // t is the fractional part of the index
      const ctype t = idx - lidx;

      // Do the spline interpolation
      const NT lower = device_data(lidx, 0);
      const NT upper = device_data(uidx, 0);

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower)) // linear part
               + t * (powr<2>(t) - 1) * device_data(lidx, 1) / (ctype)(6) -
               (t - 2) * (t - 1) * t * device_data(uidx, 1) / (ctype)(6); // cubic part
      else
        return t * upper + (1 - t) * lower // linear part
               + t * (powr<2>(t) - 1) * device_data(lidx, 1) / (ctype)(6) -
               (t - 2) * (t - 1) * t * device_data(uidx, 1) / (ctype)(6); // cubic part
    }

    template <typename MemorySpace> auto &get_on()
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called get_on() on a copied instance. This is not allowed. "
            "You need to call get_on() on the original instance.");

      if constexpr (std::is_same_v<MemorySpace, DefaultMemorySpace>) {
        return *this; // Return the current instance if the memory space matches
      } else {
        // Create a new instance with the same data but in the requested memory space
        if (other_instance == nullptr)
          other_instance = std::make_shared<SplineInterpolator1D<NT, Coordinates, MemorySpace>>(coordinates);
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
    const uint size;

    using ViewType = Kokkos::View<NT *[2], DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType device_data;
    HostViewType host_data;

    std::shared_ptr<SplineInterpolator1D<NT, Coordinates, other_memory_space>> other_instance;

    void build_y2(const ctype lower_y1, const ctype upper_y1)
    {
      const auto &yv = host_data;
      auto &y2 = host_data;

      NT p, qn, sig, un;
      std::vector<NT> u(size - 1);

      if (lower_y1 > 0.99e99)
        y2(0, 1) = u[0] = 0.0;
      else {
        y2(0, 1) = -0.5;
        u[0] = 3.0 * ((yv(1, 0) - yv(0, 0)) - lower_y1);
      }
      for (uint i = 1; i < size - 1; i++) {
        sig = 0.5;
        p = sig * y2(i - 1, 1) + 2.0;
        y2(i, 1) = (sig - 1.0) / p;
        u[i] = (yv(i + 1, 0) - yv(i, 0)) - (yv(i, 0) - yv(i - 1, 0));
        u[i] = (6.0 * u[i] / 2. - sig * u[i - 1]) / p;
      }
      if (upper_y1 > 0.99e99)
        qn = un = 0.0;
      else {
        qn = 0.5;
        un = 3.0 * (upper_y1 - (yv(size - 1, 0) - yv(size - 2, 0)));
      }
      y2(size - 1, 1) = (un - qn * u[size - 2]) / (qn * y2(size - 2, 1) + 1);
      for (int k = size - 2; k >= 0; k--)
        y2(k, 1) = y2(k, 1) * y2(k + 1, 1) + u[k];
    }
  };
} // namespace DiFfRG
