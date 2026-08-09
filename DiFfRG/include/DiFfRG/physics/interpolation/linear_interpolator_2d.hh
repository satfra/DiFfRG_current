#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/interpolation/interpolation_stencil.hh>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 2D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates, typename DefaultMemorySpace = CPU_memory> class LinearInterpolator2D
  {
    static_assert(Coordinates::dim == 2, "LinearInterpolator2D requires 2D coordinates");

    static constexpr bool periodic_x = is_periodic_axis_v<Coordinates, 0>;
    static constexpr bool periodic_y = is_periodic_axis_v<Coordinates, 1>;

  public:
    using memory_space = DefaultMemorySpace;
    using other_memory_space = other_memory_space_t<DefaultMemorySpace>;
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 2;

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
     * @brief Copy constructor for LinearInterpolator2D. This is ONLY for usage inside Kokkos parallel loops.
     *
     */
    KOKKOS_FUNCTION
    LinearInterpolator2D(const LinearInterpolator2D &other)
        : coordinates(other.coordinates), sizes(other.sizes), total_size(other.total_size)
    {
      // Use the same data
      device_data = other.device_data;
    }

    KOKKOS_FUNCTION ~LinearInterpolator2D()
    {
      KOKKOS_IF_ON_HOST((if (owns_other_instance) delete other_instance;))
    }

    template <typename NT2> void update(const NT2 *in_data)
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator2D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");
      // Make a View from the input data
      Kokkos::View<const NT2 **, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          in_view(in_data, sizes[0], sizes[1]);
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
            "LinearInterpolator2D: You probably called operator[]() on a copied instance. This is not allowed. "
            "You need to call operator[]() on the original instance.");
      return host_data.data()[i]; // Access the host data directly
    }

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    /**
     * @brief Map physical coordinates onto grid indices.
     *
     * Split out of operator() because it is the expensive half: Coordinates::backward is a fp64
     * log/log1p per logarithmic axis, ~200 fp64 instructions each on current NVIDIA parts against
     * ~10 for the interpolation. A generated kernel evaluating several dressings at the SAME point
     * otherwise pays it once per dressing -- the compiler cannot CSE it, because each interpolator
     * owns its own `coordinates` members and cannot be proven to agree with another's.
     *
     * Depends only on the coordinate system, so the result may be shared across interpolators that
     * share one. Clamping and stencil resolution stay in at(), since they are size dependent and
     * would otherwise make an index untransferable between interpolators of different extent.
     *
     * The return type is spelled out rather than deduced: nvcc loses `decltype(var)` when the
     * initializer has a deduced return type inside a class-template member, and the consumer stores
     * this in a `const auto`.
     */
    device::array<typename Coordinates::ctype, 2> KOKKOS_FUNCTION
    index(const typename Coordinates::ctype x, const typename Coordinates::ctype y) const
    {
      return coordinates.backward(x, y);
    }

    /**
     * @brief Interpolate at grid indices previously obtained from index().
     */
    NT KOKKOS_FUNCTION at(const device::array<typename Coordinates::ctype, 2> &idx) const
    {
      const auto idx_x = idx[0];
      const auto idx_y = idx[1];
      // Clamped [i, i+1] stencil on bounded axes, wrapping [i, (i+1) % n] on periodic ones
      const auto sx = make_interpolation_stencil<periodic_x>(idx_x, sizes[0]);
      const auto sy = make_interpolation_stencil<periodic_y>(idx_y, sizes[1]);

      const size_t x0 = sx.lower, x1 = sx.upper;
      const size_t y0 = sy.lower, y1 = sy.upper;

      const auto corner00 = device_data(x0, y0);
      const auto corner01 = device_data(x0, y1);
      const auto corner10 = device_data(x1, y0);
      const auto corner11 = device_data(x1, y1);

      const auto tx = sx.t;
      const auto ty = sy.t;

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(ty, Kokkos::fma(tx, corner11, Kokkos::fma(-tx, corner01, corner01)),
                           (1 - ty) * Kokkos::fma(tx, corner10, Kokkos::fma(-tx, corner00, corner00)));
      else
        return corner00 * (1 - tx) * (1 - ty) + corner01 * (1 - tx) * ty + corner10 * tx * (1 - ty) +
               corner11 * tx * ty;
    }

    /**
     * @brief Interpolate the data at a given point.
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype x,
                                  const typename Coordinates::ctype y) const
    {
      return at(index(x, y));
    }

    auto &CPU() const { return get_on<CPU_memory>(); }
    auto &GPU() const { return get_on<GPU_memory>(); }

    template <typename MemorySpace> auto &get_on() const
    {
      // Check if the host data is already allocated
      if (!host_data.is_allocated())
        throw std::runtime_error(
            "LinearInterpolator2D: You probably called get_on[]() on a copied instance. This is not allowed. "
            "You need to call get_on[]() on the original instance.");

      if constexpr (std::is_same_v<MemorySpace, DefaultMemorySpace>) {
        // remove constness
        return const_cast<LinearInterpolator2D<NT, Coordinates, MemorySpace> &>(*this);
      } else {
        // Create a new instance with the same data but in the requested memory space
        if (other_instance == nullptr) {
          other_instance = new LinearInterpolator2D<NT, Coordinates, MemorySpace>(coordinates);
          owns_other_instance = true;
          other_instance->other_instance = const_cast<std::decay_t<decltype(*this)> *>(this);
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

    friend class LinearInterpolator2D<NT, Coordinates, other_memory_space>;

  private:
    const Coordinates coordinates;
    const device::array<size_t, 2> sizes;
    const size_t total_size;

    using ViewType = Kokkos::View<NT **, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::host_mirror_type;

    ViewType device_data;
    HostViewType host_data;

    mutable LinearInterpolator2D<NT, Coordinates, other_memory_space> *other_instance = nullptr;
    mutable bool owns_other_instance = false;
  };
} // namespace DiFfRG