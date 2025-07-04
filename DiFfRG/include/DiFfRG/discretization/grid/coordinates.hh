#pragma once

// standard library
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

// A concept for what is a coordinate class
template <typename T>
concept IsCoordinate = requires(T t) {
  T::ctype;
  T::dim;
  t.size();
};

namespace DiFfRG
{
  /**
   * @brief Utility class for combining multiple coordinate systems into one
   *
   * @tparam Coordinates... the coordinate systems to combine
   */
  template <typename... Coordinates> class CoordinatePackND
  {
    // Assert that all coordinates have the same ctype
    static_assert((std::is_same<typename Coordinates::ctype,
                                typename std::tuple_element<0, std::tuple<Coordinates...>>::type::ctype>::value &&
                   ...));
    static_assert(sizeof...(Coordinates) > 0, "CoordinatePackND requires at least one coordinate system");
    // Assert that all coordinates have the dim 1
    static_assert(((Coordinates::dim == 1) && ...), "CoordinatePackND requires all coordinates to have dim 1");

  public:
    using ctype = typename std::tuple_element<0, std::tuple<Coordinates...>>::type::ctype;
    static constexpr uint dim = sizeof...(Coordinates);

    /**
     * @brief Construct a new CoordinatePackND object
     *
     * @param coordinates the coordinate systems to combine
     */
    CoordinatePackND(Coordinates... coordinates) : coordinates(coordinates...) {}

    template <typename... I> KOKKOS_FORCEINLINE_FUNCTION std::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return forward_impl(std::make_integer_sequence<int, sizeof...(I)>(), std::forward<I>(i)...);
    }

    template <typename... I, int... Is>
    std::array<ctype, dim> KOKKOS_FORCEINLINE_FUNCTION forward_impl(std::integer_sequence<int, Is...>, I &&...i) const
    {
      return {{std::get<Is>(coordinates).forward(std::get<Is>(std::tie(i...)))...}};
    }

    template <typename... I> KOKKOS_FORCEINLINE_FUNCTION std::array<ctype, sizeof...(I)> backward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return backward_impl(std::make_integer_sequence<int, sizeof...(I)>(), std::forward<I>(i)...);
    }

    template <typename... I, int... Is>
    std::array<ctype, sizeof...(I)> KOKKOS_FORCEINLINE_FUNCTION backward_impl(std::integer_sequence<int, Is...>,
                                                                              I &&...i) const
    {
      return {{std::get<Is>(coordinates).backward(std::get<Is>(std::tie(i...)))...}};
    }

    template <uint i> const auto &get_coordinates() const { return std::get<i>(coordinates); }

    uint KOKKOS_FORCEINLINE_FUNCTION size() const
    {
      // multiply all sizes
      uint size = 1;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { size *= std::get<i>(coordinates).size(); });
      return size;
    }

    std::array<uint, sizeof...(Coordinates)> KOKKOS_FORCEINLINE_FUNCTION sizes() const
    {
      std::array<uint, sizeof...(Coordinates)> sizes;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { sizes[i] = std::get<i>(coordinates).size(); });
      return sizes;
    }

    std::array<uint, sizeof...(Coordinates)> KOKKOS_FORCEINLINE_FUNCTION from_continuous_index(size_t s) const
    {
      std::array<uint, sizeof...(Coordinates)> idx;
      // calculate the index for each coordinate system
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) {
        idx[sizeof...(Coordinates) - 1 - i] = s % std::get<sizeof...(Coordinates) - 1 - i>(coordinates).size();
        s = s / std::get<sizeof...(Coordinates) - 1 - i>(coordinates).size();
      });

      return idx;
    }

  protected:
    const std::tuple<Coordinates...> coordinates;
  };

  template<typename Base>
  class SubCoordinates : public Base
  {
  public:
    using ctype = typename Base::ctype;
    static constexpr uint dim = Base::dim;

    SubCoordinates(const Base &base, uint offset, uint size)
        : Base(base), m_size(size)
    {
      if(size == 0) throw std::runtime_error("SubCoordinates: size must be > 0");
      if(offset + size > base.size()) throw std::runtime_error("SubCoordinates: offset + size must be <= base.size()");
      // calculate the offsets and sizes from the continuous index
      offsets = base.from_continuous_index(offset);
      m_sizes = base.from_continuous_index(offset + size);
      for (uint i = 0; i < dim; ++i)
        m_sizes[i] -= offsets[i];
    }

    std::array<uint, dim> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return m_sizes; }

    uint KOKKOS_FORCEINLINE_FUNCTION size() const
    {
      return m_size;
    }
    
    std::array<uint, dim> KOKKOS_FORCEINLINE_FUNCTION from_continuous_index(size_t s) const
    {
      std::array<uint, dim> result{};
      // we do it for m_sizes
      for (uint i = 0; i < dim; ++i) {
        result[dim - 1 - i] = s % m_sizes[dim - 1 - i];
        s = s / m_sizes[dim - 1 - i];
      }
      return result;
    }

    template <typename... I> KOKKOS_FORCEINLINE_FUNCTION std::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == dim);
      return forward({{i...}});
    }

    KOKKOS_FORCEINLINE_FUNCTION std::array<ctype, dim> forward(std::array<uint, dim> i) const
    {
      for (uint j = 0; j < dim; ++j) {
        i[j] += offsets[j];
      }
      return Base::forward(i);
    }

    template<typename... I>
    KOKKOS_FORCEINLINE_FUNCTION std::array<ctype, dim> backward(I &&...i) const
    {
      static_assert(sizeof...(I) == dim);
      return backward({{i...}});
    }
    KOKKOS_FORCEINLINE_FUNCTION std::array<ctype, dim> backward(std::array<uint, dim> i) const
    {
      auto result =  Base::backward(i);
      for (uint j = 0; j < dim; ++j) {
        result[j] -= offsets[j];
      }
      return result;
    }
    
    private:
    std::array<uint, dim> offsets, m_sizes;
    const uint m_size;
  };

  template <typename NT>
    requires std::is_floating_point_v<NT>
  class LinearCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr uint dim = 1;

    LinearCoordinates1D(uint grid_extent, double start, double stop)
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

    std::array<uint, 1> KOKKOS_FORCEINLINE_FUNCTION from_continuous_index(auto i) const
    {
      return std::array<uint, 1>{i};
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const { return start + a * x; }

    template <typename IT> std::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const std::array<IT, 1> &x) const
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

    uint KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    std::array<uint, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    const NT start, stop;

  private:
    const uint grid_extent;
    NT a;
  };

  template <typename NT>
    requires std::is_floating_point_v<NT>
  class LogarithmicCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr uint dim = 1;

    LogarithmicCoordinates1D(uint grid_extent, NT start, NT stop, NT bias)
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

    std::array<uint, 1> KOKKOS_FORCEINLINE_FUNCTION from_continuous_index(auto i) const
    {
      return std::array<uint, 1>{i};
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

    template <typename IT> std::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const std::array<IT, 1> &x) const
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

    uint KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    std::array<uint, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    const NT start, stop, bias;

  private:
    const uint grid_extent;
    const NT gem1, gem1inv;
    NT a, b, c;
  };

  template <typename Coordinates> auto make_grid(const Coordinates &coordinates)
  {
    using ctype = typename Coordinates::ctype;
    if constexpr (Coordinates::dim == 1) {
      std::vector<ctype> grid(coordinates.size());
      for (uint i = 0; i < coordinates.size(); ++i)
        grid[i] = coordinates.forward(i);
      return grid;
    } else if constexpr (Coordinates::dim == 2) {
      std::vector<std::array<ctype, 2>> grid(coordinates.size());
      for (uint i = 0; i < coordinates.sizes()[0]; ++i)
        for (uint j = 0; j < coordinates.sizes()[1]; ++j)
          grid[i * coordinates.sizes()[1] + j] = coordinates.forward(i, j);
      return grid;
    } else if constexpr (Coordinates::dim == 3) {
      std::vector<std::array<ctype, 3>> grid(coordinates.size());
      for (uint i = 0; i < coordinates.sizes()[0]; ++i)
        for (uint j = 0; j < coordinates.sizes()[1]; ++j)
          for (uint k = 0; k < coordinates.sizes()[2]; ++k)
            grid[i * coordinates.sizes()[1] * coordinates.sizes()[2] + j * coordinates.sizes()[2] + k] =
                coordinates.forward(i, j, k);
      return grid;
    } else {
      throw std::runtime_error("make_grid only works for 1D, 2D, and 3D coordinates");
    }
  }

  template <typename Coordinates> auto make_idx_grid(const Coordinates &coordinates) -> std::vector<double>
  {
    if constexpr (Coordinates::dim == 1) {
      std::vector<double> grid(coordinates.size());
      for (uint i = 0; i < coordinates.size(); ++i)
        grid[i] = i;
      return grid;
    } else if constexpr (Coordinates::dim == 2) {
      std::vector<double> grid(coordinates.size());
      for (uint i = 0; i < coordinates.sizes()[0]; ++i)
        for (uint j = 0; j < coordinates.sizes()[1]; ++j)
          grid[i * coordinates.sizes()[1] + j] = i * coordinates.sizes()[1] + j;
      return grid;
    } else if constexpr (Coordinates::dim == 3) {
      std::vector<double> grid(coordinates.size());
      for (uint i = 0; i < coordinates.sizes()[0]; ++i)
        for (uint j = 0; j < coordinates.sizes()[1]; ++j)
          for (uint k = 0; k < coordinates.sizes()[2]; ++k)
            grid[i * coordinates.sizes()[1] * coordinates.sizes()[2] + j * coordinates.sizes()[2] + k] =
                i * coordinates.sizes()[1] * coordinates.sizes()[2] + j * coordinates.sizes()[2] + k;
      return grid;
    } else {
      throw std::runtime_error("make_idx_grid only works for 1D, 2D, and 3D coordinates");
    }
  }
} // namespace DiFfRG
