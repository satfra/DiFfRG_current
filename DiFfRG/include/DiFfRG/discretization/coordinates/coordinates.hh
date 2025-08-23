#pragma once

// standard library
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  // A concept for what is a coordinate class
  template <typename T>
  concept is_coordinates = requires(T t) {
    typename T::ctype;
    T::dim;
    t.size();
    t.forward(device::array<size_t, T::dim>{});
    { t.from_linear_index(size_t{}) } -> std::same_as<device::array<size_t, T::dim>>;
  };

  /**
   * @brief Utility class for combining multiple coordinate systems into one
   *
   * @tparam Coordinates... the coordinate systems to combine
   */
  template <typename... Coordinates> class CoordinatePackND
  {
    // Assert that all coordinates have the same ctype
    static_assert((std::is_same<typename Coordinates::ctype,
                                typename device::tuple_element<0, device::tuple<Coordinates...>>::type::ctype>::value &&
                   ...));
    static_assert(sizeof...(Coordinates) > 0, "CoordinatePackND requires at least one coordinate system");
    // Assert that all coordinates have the dim 1
    static_assert(((Coordinates::dim == 1) && ...), "CoordinatePackND requires all coordinates to have dim 1");

  public:
    using ctype = typename device::tuple_element<0, device::tuple<Coordinates...>>::type::ctype;
    static constexpr size_t dim = sizeof...(Coordinates);

    /**
     * @brief Construct a new CoordinatePackND object
     *
     * @param coordinates the coordinate systems to combine
     */
    CoordinatePackND(Coordinates... coordinates) : coordinates(coordinates...) {}

    template <typename... I>
      requires(std::is_convertible_v<std::decay_t<I>, size_t> && ...)
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return forward_impl(device::make_integer_sequence<int, sizeof...(I)>(), device::forward<I>(i)...);
    }
    template <typename IT>
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(const device::array<IT, dim> &coords) const
    {
      return device::apply([&](const auto &...iargs) { return forward(iargs...); }, coords);
    }

    template <typename... I, int... Is>
    device::array<ctype, dim> KOKKOS_FORCEINLINE_FUNCTION forward_impl(device::integer_sequence<int, Is...>,
                                                                       I &&...i) const
    {
      return {{device::get<Is>(coordinates).forward(device::get<Is>(device::tie(i...)))...}};
    }

    template <typename... I> KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, sizeof...(I)> backward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return backward_impl(device::make_integer_sequence<int, sizeof...(I)>(), device::forward<I>(i)...);
    }

    template <typename... I, int... Is>
    device::array<ctype, sizeof...(I)> KOKKOS_FORCEINLINE_FUNCTION backward_impl(device::integer_sequence<int, Is...>,
                                                                                 I &&...i) const
    {
      return {{device::get<Is>(coordinates).backward(device::get<Is>(device::tie(i...)))...}};
    }

    template <size_t i> const auto &get_coordinates() const { return device::get<i>(coordinates); }

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const
    {
      // multiply all sizes
      size_t size = 1;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { size *= device::get<i>(coordinates).size(); });
      return size;
    }

    device::array<size_t, sizeof...(Coordinates)> KOKKOS_FORCEINLINE_FUNCTION sizes() const
    {
      device::array<size_t, sizeof...(Coordinates)> sizes;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { sizes[i] = device::get<i>(coordinates).size(); });
      return sizes;
    }

    device::array<size_t, sizeof...(Coordinates)> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t s) const
    {
      device::array<size_t, sizeof...(Coordinates)> idx;
      // calculate the index for each coordinate system
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) {
        idx[sizeof...(Coordinates) - 1 - i] = s % device::get<sizeof...(Coordinates) - 1 - i>(coordinates).size();
        s = s / device::get<sizeof...(Coordinates) - 1 - i>(coordinates).size();
      });

      return idx;
    }

    template <typename... Coordinates2>
    friend bool operator==(const CoordinatePackND<Coordinates...> &r, const CoordinatePackND<Coordinates2...> &l)
    {
      if constexpr (sizeof...(Coordinates) != sizeof...(Coordinates2)) return false; // Different number of coordinates
      if constexpr ((!std::is_same_v<Coordinates, Coordinates2> && ...))
        return false; // Different types of coordinates
      else if constexpr (sizeof...(Coordinates) == 0)
        return true; // Empty coordinate pack, always equal
      else {
        // Check if all coordinates are equal
        bool equal = true;
        constexpr_for<0, sizeof...(Coordinates), 1>(
            [&](auto i) { equal &= (device::get<i>(r.coordinates) == device::get<i>(l.coordinates)); });
        return equal;
      }
    }

    std::string to_string() const
    {
      std::string ret = "CoordinatePackND(";
      constexpr_for<0, dim, 1>([&](auto i) {
        ret += device::get<i>(coordinates).to_string();
        if (i + 1 < dim) ret += ", ";
      });
      ret += ")";
      return ret;
    }

  protected:
    const device::tuple<Coordinates...> coordinates;
  };

  template <typename Base> class SubCoordinates : public Base
  {
  public:
    using ctype = typename Base::ctype;
    static constexpr size_t dim = Base::dim;

    SubCoordinates(const Base &base, size_t offset, size_t size) : Base(base), m_size(size)
    {
      if (size == 0) throw std::runtime_error("SubCoordinates: size must be > 0");
      if (offset + size > base.size()) throw std::runtime_error("SubCoordinates: offset + size must be <= base.size()");
      // calculate the offsets and sizes from the continuous index
      offsets = base.from_linear_index(offset);
      m_sizes = base.from_linear_index(offset + size);
      for (size_t i = 0; i < dim; ++i)
        m_sizes[i] -= offsets[i];
    }

    device::array<size_t, dim> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return m_sizes; }

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return m_size; }

    device::array<size_t, dim> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t s) const
    {
      device::array<size_t, dim> result{};
      // we do it for m_sizes
      for (size_t i = 0; i < dim; ++i) {
        result[dim - 1 - i] = s % m_sizes[dim - 1 - i];
        s = s / m_sizes[dim - 1 - i];
      }
      return result;
    }

    template <typename... I>
      requires(std::is_convertible_v<std::decay_t<I>, size_t> && ...)
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == dim);
      return forward({{i...}});
    }

    template <typename IT> KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(device::array<IT, dim> i) const
    {
      for (size_t j = 0; j < dim; ++j) {
        i[j] += offsets[j];
      }
      return Base::forward(i);
    }

    template <typename... I> KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> backward(I &&...i) const
    {
      static_assert(sizeof...(I) == dim);
      return backward({{i...}});
    }
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> backward(device::array<size_t, dim> i) const
    {
      auto result = Base::backward(i);
      for (size_t j = 0; j < dim; ++j) {
        result[j] -= offsets[j];
      }
      return result;
    }

    std::string to_string() const
    {
      std::string ret = "SubCoordinates(" + Base::to_string() + ", {";
      for (uint i = 0; i < dim; ++i) {
        ret += std::to_string(offsets[i]);
        if (i + 1 < dim) ret += ", ";
      }
      ret += "}, {";
      for (uint i = 0; i < dim; ++i) {
        ret += std::to_string(m_sizes[i]);
        if (i + 1 < dim) ret += ", ";
      }
      return ret + "})";
    }

  private:
    device::array<size_t, dim> offsets, m_sizes;
    const size_t m_size;
  };

  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class LinearCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 1;

    LinearCoordinates1D(size_t grid_extent, double start, double stop)
        : start(start), stop(stop), grid_extent(grid_extent)
    {
      if (grid_extent == 0) throw std::runtime_error("LinearCoordinates1D: grid_extent must be > 0");
      a = (stop - start) / (grid_extent - 1.);
    }

    template <typename NT2>
    LinearCoordinates1D(const LinearCoordinates1D<NT2> &other)
        : LinearCoordinates1D(other.size(), other.start, other.stop)
    {
    }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(auto i) const
    {
      return device::array<size_t, 1>{i};
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const { return start + a * x; }

    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const { return (y - start) / a; }

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    const NT start, stop;

    template <typename NT2>
    friend bool operator==(const LinearCoordinates1D<NT> &lhs, const LinearCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.grid_extent == rhs.grid_extent;
    }

    std::string to_string() const
    {
      return "LinearCoordinates1D(" + std::to_string(grid_extent) + ", " + std::to_string(start) + ", " +
             std::to_string(stop) + ")";
    }

  private:
    const size_t grid_extent;
    NT a;
  };

  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class LogarithmicCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 1;

    LogarithmicCoordinates1D(size_t grid_extent, NT start, NT stop, NT bias)
        : start(start), stop(stop), bias(bias), grid_extent(grid_extent), gem1(grid_extent - 1.),
          gem1inv(1. / (grid_extent - 1.))
    {
      if (grid_extent == 0) throw std::runtime_error("LogarithmicCoordinates1D: grid_extent must be > 0");
      using Kokkos::expm1;
      a = bias;
      b = (stop - start) / expm1(a);
      c = start;
    }

    template <typename NT2>
    LogarithmicCoordinates1D(const LogarithmicCoordinates1D<NT2> &other)
        : LogarithmicCoordinates1D(other.size(), other.start, other.stop, other.bias)
    {
    }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(auto i) const
    {
      return device::array<size_t, 1>{i};
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const
    {
      using Kokkos::expm1;
      return b * expm1(a * x * gem1inv) + c;
    }

    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const
    {
      using Kokkos::log1p;
      return log1p((y - c) / b) * gem1 / a;
    }

    NT KOKKOS_FORCEINLINE_FUNCTION backward_derivative(const NT &y) const { return 1. / (y - c) * gem1 / a; }

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    const NT start, stop, bias;

    template <typename NT2>
    friend bool operator==(const LogarithmicCoordinates1D<NT> &lhs, const LogarithmicCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.bias == rhs.bias &&
             lhs.grid_extent == rhs.grid_extent;
    }

    std::string to_string() const
    {
      return "LogarithmicCoordinates1D(" + std::to_string(start) + ", " + std::to_string(stop) + ", " +
             std::to_string(bias) + ", " + std::to_string(grid_extent) + ")";
    }

  private:
    const size_t grid_extent;
    const NT gem1, gem1inv;
    NT a, b, c;
  };

  template <typename Coordinates> auto make_grid(const Coordinates &coordinates)
  {
    using ctype = typename Coordinates::ctype;
    using cortype = device::array<ctype, Coordinates::dim>;
    std::vector<cortype> grid(coordinates.size());
    for (size_t i = 0; i < coordinates.size(); ++i) {
      const auto forwarded = coordinates.forward(coordinates.from_linear_index(i));
      for (size_t j = 0; j < Coordinates::dim; ++j) {
        grid[i][j] = forwarded[j];
      }
    }
    return grid;
  }

  template <typename Coordinates> auto make_idx_grid(const Coordinates &coordinates) -> std::vector<double>
  {
    using ctype = typename Coordinates::ctype;
    using cortype = device::array<ctype, Coordinates::dim>;
    std::vector<cortype> grid(coordinates.size());
    for (size_t i = 0; i < coordinates.size(); ++i) {
      const auto forwarded = coordinates.from_linear_index(i);
      for (size_t j = 0; j < Coordinates::dim; ++j) {
        grid[i][j] = forwarded[j];
      }
    }
    return grid;
  }

  template <typename Coordinates> std::vector<typename Coordinates::ctype> dump_grid(const Coordinates &coordinates)
  {
    using ctype = typename Coordinates::ctype;
    std::vector<typename Coordinates::ctype> grid(coordinates.size() * Coordinates::dim);
    for (size_t i = 0; i < coordinates.size(); ++i) {
      for (size_t j = 0; j < Coordinates::dim; ++j) {
        const auto forwarded = coordinates.forward(coordinates.from_linear_index(i));
        grid[i * coordinates.dim + j] = forwarded[j];
      }
    }
    return grid;
  }

  // Definitions of useful combined coordinates
  using LogCoordinates = LogarithmicCoordinates1D<double>;
  using LinCoordinates = LinearCoordinates1D<double>;
  using LogLogCoordinates = CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>>;
  using LogLinCoordinates = CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>;
  using LinLogCoordinates = CoordinatePackND<LinearCoordinates1D<double>, LogarithmicCoordinates1D<double>>;
  using LinLinCoordinates = CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>>;
  using LogLogLinCoordinates =
      CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>;
  using LogLinLinCoordinates =
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>;
  using LinLinLinCoordinates =
      CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>;
} // namespace DiFfRG
