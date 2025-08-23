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
        : coordinates(coordinates), size(coordinates.size()), other_instance(nullptr)
    {
      // Allocate Kokkos View
      device_data = ViewType("SplineInterpolator1D_data", size);
      // Create host mirror
      host_data = Kokkos::create_mirror_view(device_data);
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

      Kokkos::View<const NT2 *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> in_view(in_data, size);
      update(in_view, lower_y1, upper_y1);
    }

    template <typename View>
      requires(Kokkos::is_view<View>::value && (View::rank == 1 || View::rank == 2))
    void update(const View &view, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator2D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");
      // Populate host mirror
      if constexpr (View::rank == ViewType::rank) {
        Kokkos::deep_copy(host_data, view);
      } else {
        auto host_data_subview = Kokkos::subview(host_data, Kokkos::ALL(), 0);
        Kokkos::deep_copy(host_data_subview, view);

        // Build the spline coefficients
        build_y2(lower_y1, upper_y1);
      }

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
      const size_t lidx = size_t(Kokkos::floor(idx));
      const size_t uidx = lidx + 1;
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

    auto &CPU() const { return get_on<CPU_memory>(); }
    auto &GPU() const { return get_on<GPU_memory>(); }

    template <typename MemorySpace> auto &get_on() const
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
        if (other_instance == nullptr) {
          other_instance = std::make_shared<SplineInterpolator1D<NT, Coordinates, MemorySpace>>(coordinates);
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

    friend class SplineInterpolator1D<NT, Coordinates, other_memory_space>;

  private:
    const Coordinates coordinates;
    const size_t size;

    using ViewType = Kokkos::View<NT *[2], DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::HostMirror;

    ViewType device_data;
    HostViewType host_data;

    mutable std::shared_ptr<SplineInterpolator1D<NT, Coordinates, other_memory_space>> other_instance;

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
      for (size_t i = 1; i < size - 1; i++) {
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
