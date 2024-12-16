#pragma once

// standard library
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/utils.hh>

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

    template <typename... I> __forceinline__ __host__ __device__ std::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return forward_impl(std::make_integer_sequence<int, sizeof...(I)>(), std::forward<I>(i)...);
    }

    template <typename... I, int... Is>
    __forceinline__ __host__ __device__ std::array<ctype, dim> forward_impl(std::integer_sequence<int, Is...>,
                                                                            I &&...i) const
    {
      return {{std::get<Is>(coordinates).forward(std::get<Is>(std::tie(i...)))...}};
    }

    template <typename... I>
    __forceinline__ __host__ __device__ std::array<ctype, sizeof...(I)> backward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return backward_impl(std::make_integer_sequence<int, sizeof...(I)>(), std::forward<I>(i)...);
    }

    template <typename... I, int... Is>
    std::array<ctype, sizeof...(I)> __forceinline__ __host__ __device__ backward_impl(std::integer_sequence<int, Is...>,
                                                                                      I &&...i) const
    {
      return {{std::get<Is>(coordinates).backward(std::get<Is>(std::tie(i...)))...}};
    }

    template <uint i> const auto &get_coordinates() const { return std::get<i>(coordinates); }

    uint size() const
    {
      // multiply all sizes
      uint size = 1;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { size *= std::get<i>(coordinates).size(); });
      return size;
    }

    std::array<uint, sizeof...(Coordinates)> sizes() const
    {
      std::array<uint, sizeof...(Coordinates)> sizes;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { sizes[i] = std::get<i>(coordinates).size(); });
      return sizes;
    }

  protected:
    const std::tuple<Coordinates...> coordinates;
  };

  template <typename NT> class LinearCoordinates1D
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

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    NT __forceinline__ __device__ __host__ forward(const uint &x) const { return start + a * x; }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    NT __forceinline__ __device__ __host__ backward(const NT &y) const { return (y - start) / a; }

    uint size() const { return grid_extent; }

    const NT start, stop;

  private:
    const uint grid_extent;
    NT a;
  };

  template <typename NT> class LogarithmicCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr uint dim = 1;

    LogarithmicCoordinates1D(uint grid_extent, NT start, NT stop, NT bias)
        : start(start), stop(stop), bias(bias), grid_extent(grid_extent), gem1(grid_extent - 1.),
          gem1inv(1. / (grid_extent - 1.))
    {
      if (grid_extent == 0) throw std::runtime_error("LogarithmicCoordinates1D: grid_extent must be > 0");
      using std::expm1;
      a = bias;
      b = (stop - start) / expm1(a);
      c = start;
    }

    template <typename NT2>
    LogarithmicCoordinates1D(const LogarithmicCoordinates1D<NT2> &other)
        : LogarithmicCoordinates1D(other.size(), other.start, other.stop, other.bias)
    {
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    NT __forceinline__ __device__ __host__ forward(const uint &x) const
    {
      using std::expm1;
      return b * expm1(a * x * gem1inv) + c;
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    NT __forceinline__ __device__ __host__ backward(const NT &y) const
    {
      using std::log1p;
      return log1p((y - c) / b) * gem1 / a;
    }

    NT __forceinline__ __device__ __host__ backward_derivative(const NT &y) const { return 1. / (y - c) * gem1 / a; }

    uint size() const { return grid_extent; }

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
