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
    static constexpr size_t dim = 1;

    /**
     * @brief Construct a SplineInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1D(const Coordinates &coordinates)
        : coordinates(coordinates), size(coordinates.size())
    {
      // Allocate separate views for values and spline coefficients (SoA layout)
      device_values = ValueViewType("SplineInterpolator1D_values", size);
      device_coeffs = CoeffViewType("SplineInterpolator1D_coeffs", size);
      // Create host mirrors
      host_values = Kokkos::create_mirror_view(device_values);
      host_coeffs = Kokkos::create_mirror_view(device_coeffs);
    }

    /**
     * @brief Copy constructor for SplineInterpolator1D. This is ONLY for usage inside Kokkos parallel loops.
     *
     */
    KOKKOS_FUNCTION
    SplineInterpolator1D(const SplineInterpolator1D &other)
        : coordinates(other.coordinates), size(other.size)
    {
      // Use the same data (reference-counted)
      device_values = other.device_values;
      device_coeffs = other.device_coeffs;
    }

    KOKKOS_FUNCTION ~SplineInterpolator1D()
    {
      KOKKOS_IF_ON_HOST((if (owns_other_instance) delete other_instance;))
    }

    template <typename NT2>
    void update(const NT2 *in_data, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Check if the host data is already allocated
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");

      // Copy values from input data
      for (size_t i = 0; i < size; ++i)
        host_values(i) = in_data[i];

      // Build the spline coefficients
      build_y2(lower_y1, upper_y1);

      // Copy data to device
      Kokkos::deep_copy(device_values, host_values);
      Kokkos::deep_copy(device_coeffs, host_coeffs);
    }

    template <typename View>
      requires(Kokkos::is_view<View>::value && View::rank == 1)
    void update(const View &view, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Check if the host data is already allocated
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");

      // Copy values from input view
      Kokkos::deep_copy(host_values, view);

      // Build the spline coefficients
      build_y2(lower_y1, upper_y1);

      // Copy data to device
      Kokkos::deep_copy(device_values, host_values);
      Kokkos::deep_copy(device_coeffs, host_coeffs);
    }

    NT operator[](size_t i) const
    {
      // Check if the host data is already allocated
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called operator[]() on a copied instance. This is not allowed. "
            "You need to call operator[]() on the original instance.");

      return host_values(i);
    }

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype x) const
    {
      ctype idx = coordinates.backward(x);
      // Clamp the index to the range [0, size - 1]
      idx = Kokkos::max(static_cast<decltype(idx)>(0), Kokkos::min(idx, static_cast<decltype(idx)>(size - 1)));
      const size_t lidx = Kokkos::min(size_t(Kokkos::floor(idx)), size - 2);
      const size_t uidx = lidx + 1;
      // t is the fractional part of the index
      const ctype t = idx - lidx;

      // SoA layout: separate reads from values and coefficients views for coalesced GPU access
      const NT lower = device_values(lidx);
      const NT upper = device_values(uidx);
      const NT cl = device_coeffs(lidx);
      const NT cu = device_coeffs(uidx);

      const ctype tm1 = t - 1;
      const NT cubic = t * tm1 * ((t + 1) * cl - (t - 2) * cu);

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower)) + cubic; // linear + cubic
      else
        return t * upper + (1 - t) * lower + cubic; // linear + cubic
    }

    auto &CPU() const { return get_on<CPU_memory>(); }
    auto &GPU() const { return get_on<GPU_memory>(); }

    template <typename MemorySpace> auto &get_on() const
    {
      // Check if the host data is already allocated
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called get_on() on a copied instance. This is not allowed. "
            "You need to call get_on() on the original instance.");

      if constexpr (std::is_same_v<MemorySpace, DefaultMemorySpace>) {
        // remove constness
        return const_cast<SplineInterpolator1D<NT, Coordinates, MemorySpace> &>(*this);
      } else {
        // Create a new instance with the same data but in the requested memory space
        if (other_instance == nullptr) {
          other_instance = new SplineInterpolator1D<NT, Coordinates, MemorySpace>(coordinates);
          owns_other_instance = true;
          other_instance->other_instance = const_cast<std::decay_t<decltype(*this)> *>(this);
        }
        // Copy the data from the current instance to the new one
        other_instance->update(host_values);
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
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called data() on a copied instance. This is not allowed. "
            "You need to call data() on the original instance.");
      return host_values.data();
    }

    friend class SplineInterpolator1D<NT, Coordinates, other_memory_space>;

  private:
    const Coordinates coordinates;
    const size_t size;

    using ValueViewType = Kokkos::View<NT *, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using CoeffViewType = Kokkos::View<NT *, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostValueViewType = typename ValueViewType::host_mirror_type;
    using HostCoeffViewType = typename CoeffViewType::host_mirror_type;

    ValueViewType device_values;
    CoeffViewType device_coeffs;
    HostValueViewType host_values;
    HostCoeffViewType host_coeffs;

    mutable SplineInterpolator1D<NT, Coordinates, other_memory_space> *other_instance = nullptr;
    mutable bool owns_other_instance = false;

    void build_y2(const ctype lower_y1, const ctype upper_y1)
    {
      NT p, qn, sig, un;
      std::vector<NT> u(size - 1);

      if (lower_y1 > 0.99e99)
        host_coeffs(0) = u[0] = 0.0;
      else {
        host_coeffs(0) = -0.5;
        u[0] = 3.0 * ((host_values(1) - host_values(0)) - lower_y1);
      }
      for (size_t i = 1; i < size - 1; i++) {
        sig = 0.5;
        p = sig * host_coeffs(i - 1) + 2.0;
        host_coeffs(i) = (sig - 1.0) / p;
        u[i] = (host_values(i + 1) - host_values(i)) - (host_values(i) - host_values(i - 1));
        u[i] = (6.0 * u[i] / 2. - sig * u[i - 1]) / p;
      }
      if (upper_y1 > 0.99e99)
        qn = un = 0.0;
      else {
        qn = 0.5;
        un = 3.0 * (upper_y1 - (host_values(size - 1) - host_values(size - 2)));
      }
      host_coeffs(size - 1) = (un - qn * u[size - 2]) / (qn * host_coeffs(size - 2) + 1);
      for (int k = size - 2; k >= 0; k--)
        host_coeffs(k) = host_coeffs(k) * host_coeffs(k + 1) + u[k];

      // Precompute division by 6 so operator() avoids per-call divides
      for (size_t k = 0; k < size; ++k)
        host_coeffs(k) /= (ctype)6;
    }
  };
} // namespace DiFfRG
