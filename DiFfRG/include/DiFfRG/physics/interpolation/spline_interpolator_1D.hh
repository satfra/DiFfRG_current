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
  template <typename MemorySpace, typename NT, typename Coordinates> class SplineInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "SplineInterpolator1D requires 1D coordinates");

  public:
    using memory_space = MemorySpace;
    using ctype = typename Coordinates::ctype;

    /**
     * @brief Construct a SplineInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1D(const Coordinates &coordinates) : coordinates(coordinates), size(coordinates.size())
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
        : coordinates(coordinates), size(coordinates.size())
    {
      // Allocate Kokkos View
      device_data = ViewType("SplineInterpolator1D_data", size);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
      // Update device data
      update(in_data, lower_y1, upper_y1);
    }

    template <typename NT2>
    void update(const NT2 *in_data, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Populate host mirror
      for (uint i = 0; i < size; ++i)
        host_data(i, 0) = static_cast<NT>(in_data[i]);
      // Build the spline coefficients
      build_y2(lower_y1, upper_y1);
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

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const Coordinates coordinates;
    const uint size;

    using ViewType = Kokkos::View<NT *[2], MemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType device_data;
    HostViewType host_data;

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
