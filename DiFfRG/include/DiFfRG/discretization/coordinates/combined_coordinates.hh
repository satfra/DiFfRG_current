#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/discretization/coordinates/stack_coordinates.hh>

namespace DiFfRG
{
  /**
   * @brief Matsubara frequencies combined with a radial momentum axis.
   *
   * @tparam Radial the coordinate system of the radial axis. Defaults to LogarithmicCoordinates1D,
   * which is what the seven-argument constructor builds; pass e.g. FocusedLogCoordinates1D through
   * the coordinate-object constructor to cluster the radial grid around an interior scale.
   */
  template <typename Idx = int, typename NT = double, typename Radial = LogarithmicCoordinates1D<NT>>
  class BosonicCoordinates1DFiniteT
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 2;
    using radial_type = Radial;

    BosonicCoordinates1DFiniteT(Idx m_start, Idx m_stop, NT m_T, size_t grid_extent, NT p_start, NT p_stop, NT p_bias)
      requires std::is_same_v<Radial, LogarithmicCoordinates1D<NT>>
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(p_start), p_stop(p_stop), grid_extent(grid_extent),
          m_size(m_stop - m_start), matsubara_values(m_start, m_stop, m_T),
          radial_coordinates(grid_extent, p_start, p_stop, p_bias)
    {
    }

    template <typename Idx2, typename NT2, typename Radial2>
    BosonicCoordinates1DFiniteT(const BosonicCoordinates1DFiniteT<Idx2, NT2, Radial2> &other)
        : BosonicCoordinates1DFiniteT(other.radial(), other.m_start, other.m_stop, other.m_T)
    {
    }

    template <typename Radial2>
    BosonicCoordinates1DFiniteT(const Radial2 &radial_coordinates, Idx m_start, Idx m_stop, NT m_T)
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(radial_coordinates.start),
          p_stop(radial_coordinates.stop), grid_extent(radial_coordinates.size()), m_size(m_stop - m_start),
          matsubara_values(m_start, m_stop, m_T), radial_coordinates(radial_coordinates)
    {
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const size_t m, const size_t p) const
    {
      return {matsubara_values.forward(m), radial_coordinates.forward(p)};
    }
    template <typename IT> device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 2> mp) const
    {
      return forward(mp[0], mp[1]);
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    std::tuple<Idx, NT> KOKKOS_FUNCTION backward(const NT m, const NT p) const
    {
      Idx m_idx = matsubara_values.backward(m);
      NT p_idx = 0;
      if (m_idx >= int(m_size)) {
        m_idx = m_size - 1;
        const auto new_p = std::sqrt(powr<2>(p) + powr<2>(m - matsubara_values.forward(m_idx)));
        p_idx = radial_coordinates.backward(new_p);
      } else if (m_idx < 0) {
        m_idx = 0;
        const auto new_p = std::sqrt(powr<2>(p) + powr<2>(m - matsubara_values.forward(m_idx)));
        p_idx = radial_coordinates.backward(new_p);
      } else
        p_idx = radial_coordinates.backward(p);
      return {m_idx, p_idx};
    }

    device::array<size_t, 2> KOKKOS_INLINE_FUNCTION from_linear_index(auto i) const
    {
      device::array<size_t, 2> idx;
      // calculate the index for each coordinate system
      idx[0] = i / grid_extent; // Matsubara index
      idx[1] = i % grid_extent; // Radial index

      return idx;
    }

    size_t size() const { return m_size * grid_extent; }
    device::array<size_t, 2> sizes() const { return {{m_size, grid_extent}}; }

    /**
     * @brief The radial coordinate system, e.g. to rebuild this object with a different Idx or NT.
     */
    const Radial &radial() const { return radial_coordinates; }

    const NT m_start, m_stop, m_T, p_start, p_stop;

    std::string to_string() const
    {
      return "Combined(" + matsubara_values.to_string() + ", " + radial_coordinates.to_string() + ")";
    }

  private:
    const size_t grid_extent, m_size;

    BosonicMatsubaraValues<Idx, NT> matsubara_values;
    Radial radial_coordinates;
  };

  /**
   * @brief Fermionic counterpart of BosonicCoordinates1DFiniteT, see there.
   */
  template <typename Idx = int, typename NT = double, typename Radial = LogarithmicCoordinates1D<NT>>
  class FermionicCoordinates1DFiniteT
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 2;
    using radial_type = Radial;

    FermionicCoordinates1DFiniteT(Idx m_start, Idx m_stop, NT m_T, size_t grid_extent, NT p_start, NT p_stop, NT p_bias)
      requires std::is_same_v<Radial, LogarithmicCoordinates1D<NT>>
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(p_start), p_stop(p_stop), grid_extent(grid_extent),
          m_size(m_stop - m_start), matsubara_values(m_start, m_stop, m_T),
          radial_coordinates(grid_extent, p_start, p_stop, p_bias)
    {
    }

    template <typename Idx2, typename NT2, typename Radial2>
    FermionicCoordinates1DFiniteT(const FermionicCoordinates1DFiniteT<Idx2, NT2, Radial2> &other)
        : FermionicCoordinates1DFiniteT(other.radial(), other.m_start, other.m_stop, other.m_T)
    {
    }

    template <typename Radial2>
    FermionicCoordinates1DFiniteT(const Radial2 &radial_coordinates, Idx m_start, Idx m_stop, NT m_T)
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(radial_coordinates.start),
          p_stop(radial_coordinates.stop), grid_extent(radial_coordinates.size()), m_size(m_stop - m_start),
          matsubara_values(m_start, m_stop, m_T), radial_coordinates(radial_coordinates)
    {
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const size_t m, const size_t p) const
    {
      return {matsubara_values.forward(m), radial_coordinates.forward(p)};
    }
    template <typename IT> device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 2> mp) const
    {
      return forward(mp[0], mp[1]);
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    std::tuple<Idx, NT> KOKKOS_FUNCTION backward(const NT m, const NT p) const
    {
      Idx m_idx = matsubara_values.backward(m);
      NT p_idx = 0;
      if (m_idx >= int(m_size)) {
        m_idx = m_size - 1;
        const auto new_p = std::sqrt(powr<2>(p) + powr<2>(m - matsubara_values.forward(m_idx)));
        p_idx = radial_coordinates.backward(new_p);
      } else if (m_idx < 0) {
        m_idx = 0;
        const auto new_p = std::sqrt(powr<2>(p) + powr<2>(m - matsubara_values.forward(m_idx)));
        p_idx = radial_coordinates.backward(new_p);
      } else
        p_idx = radial_coordinates.backward(p);
      return {m_idx, p_idx};
    }

    device::array<size_t, 2> KOKKOS_INLINE_FUNCTION from_linear_index(auto i) const
    {
      device::array<size_t, 2> idx;
      // calculate the index for each coordinate system
      idx[0] = i / grid_extent; // Matsubara index
      idx[1] = i % grid_extent; // Radial index

      return idx;
    }

    size_t size() const { return m_size * grid_extent; }
    device::array<size_t, 2> sizes() const { return {{m_size, grid_extent}}; }

    /**
     * @brief The radial coordinate system, e.g. to rebuild this object with a different Idx or NT.
     */
    const Radial &radial() const { return radial_coordinates; }

    const NT m_start, m_stop, m_T, p_start, p_stop;

    std::string to_string() const
    {
      return "Combined(" + matsubara_values.to_string() + ", " + radial_coordinates.to_string() + ")";
    }

  private:
    const size_t grid_extent, m_size;

    FermionicMatsubaraValues<Idx, NT> matsubara_values;
    Radial radial_coordinates;
  };

  // Finite-T coordinates whose radial axis is focused on an interior scale
  using FocusedBosonicCoordinates1DFiniteT = BosonicCoordinates1DFiniteT<int, double, FocusedLogCoordinates1D<double>>;
  using FocusedFermionicCoordinates1DFiniteT =
      FermionicCoordinates1DFiniteT<int, double, FocusedLogCoordinates1D<double>>;
} // namespace DiFfRG
