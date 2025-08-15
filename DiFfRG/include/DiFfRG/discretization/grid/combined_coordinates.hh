#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/discretization/grid/stack_coordinates.hh>

namespace DiFfRG
{
  template <typename Idx = int, typename NT = double> class BosonicCoordinates1DFiniteT
  {
  public:
    using ctype = NT;
    static constexpr uint dim = 2;

    BosonicCoordinates1DFiniteT(Idx m_start, Idx m_stop, NT m_T, uint grid_extent, NT p_start, NT p_stop, NT p_bias)
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(p_start), p_stop(p_stop), p_bias(p_bias),
          grid_extent(grid_extent), m_size(m_stop - m_start), matsubara_values(m_start, m_stop, m_T),
          logarithmic_coordinates(grid_extent, p_start, p_stop, p_bias)
    {
    }

    template <typename Idx2, typename NT2>
    BosonicCoordinates1DFiniteT(const BosonicCoordinates1DFiniteT<Idx2, NT2> &other)
        : BosonicCoordinates1DFiniteT(other.m_start, other.m_stop, other.m_T, other.grid_extent, other.p_start,
                                      other.p_stop, other.p_bias)
    {
    }

    template <typename NT2>
    BosonicCoordinates1DFiniteT(const LogarithmicCoordinates1D<NT2> &logarithmic_coordinates, Idx m_start, Idx m_stop,
                                NT m_T)
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(logarithmic_coordinates.start),
          p_stop(logarithmic_coordinates.stop), p_bias(logarithmic_coordinates.bias),
          grid_extent(logarithmic_coordinates.size()), m_size(m_stop - m_start), matsubara_values(m_start, m_stop, m_T),
          logarithmic_coordinates(logarithmic_coordinates)
    {
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const uint m, const uint p) const
    {
      return {matsubara_values.forward(m), logarithmic_coordinates.forward(p)};
    }
    device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<uint, 2> mp) const
    {
      return forward(mp[0], mp[1]);
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    std::tuple<Idx, NT> KOKKOS_FORCEINLINE_FUNCTION backward(const NT m, const NT p) const
    {
      Idx m_idx = matsubara_values.backward(m);
      NT p_idx = 0;
      if (m_idx >= int(m_size)) {
        m_idx = m_size - 1;
        const auto new_p = std::sqrt(powr<2>(p) + powr<2>(m - matsubara_values.forward(m_idx)));
        p_idx = logarithmic_coordinates.backward(new_p);
      } else if (m_idx < 0) {
        m_idx = 0;
        const auto new_p = std::sqrt(powr<2>(p) + powr<2>(m - matsubara_values.forward(m_idx)));
        p_idx = logarithmic_coordinates.backward(new_p);
      } else
        p_idx = logarithmic_coordinates.backward(p);
      return {m_idx, p_idx};
    }

    device::array<uint, 2> KOKKOS_FORCEINLINE_FUNCTION from_continuous_index(auto i) const
    {
      device::array<uint, 2> idx;
      // calculate the index for each coordinate system
      idx[0] = i / grid_extent; // Matsubara index
      idx[1] = i % grid_extent; // Logarithmic index

      return idx;
    }

    uint size() const { return m_size * grid_extent; }
    device::array<uint, 2> sizes() const { return {{m_size, grid_extent}}; }

    const NT m_start, m_stop, m_T, p_start, p_stop, p_bias;

  private:
    const uint grid_extent, m_size;

    BosonicMatsubaraValues<Idx, NT> matsubara_values;
    LogarithmicCoordinates1D<NT> logarithmic_coordinates;
  };

  template <typename Idx = int, typename NT = double> class FermionicCoordinates1DFiniteT
  {
  public:
    using ctype = NT;
    static constexpr uint dim = 2;

    FermionicCoordinates1DFiniteT(Idx m_start, Idx m_stop, NT m_T, uint grid_extent, NT p_start, NT p_stop, NT p_bias)
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(p_start), p_stop(p_stop), p_bias(p_bias),
          grid_extent(grid_extent), m_size(m_stop - m_start), matsubara_values(m_start, m_stop, m_T),
          logarithmic_coordinates(grid_extent, p_start, p_stop, p_bias)
    {
    }

    template <typename Idx2, typename NT2>
    FermionicCoordinates1DFiniteT(const FermionicCoordinates1DFiniteT<Idx2, NT2> &other)
        : FermionicCoordinates1DFiniteT(other.m_start, other.m_stop, other.m_T, other.grid_extent, other.p_start,
                                        other.p_stop, other.p_bias)
    {
    }

    template <typename NT2>
    FermionicCoordinates1DFiniteT(const LogarithmicCoordinates1D<NT2> &logarithmic_coordinates, Idx m_start, Idx m_stop,
                                  NT m_T)
        : m_start(m_start), m_stop(m_stop), m_T(m_T), p_start(logarithmic_coordinates.start),
          p_stop(logarithmic_coordinates.stop), p_bias(logarithmic_coordinates.bias),
          grid_extent(logarithmic_coordinates.size()), m_size(m_stop - m_start), matsubara_values(m_start, m_stop, m_T),
          logarithmic_coordinates(logarithmic_coordinates)
    {
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const uint m, const uint p) const
    {
      return {matsubara_values.forward(m), logarithmic_coordinates.forward(p)};
    }
    device::array<NT, 2> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<uint, 2> mp) const
    {
      return forward(mp[0], mp[1]);
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    std::tuple<Idx, NT> KOKKOS_FORCEINLINE_FUNCTION backward(const NT m, const NT p) const
    {
      Idx m_idx = matsubara_values.backward(m);
      NT p_idx = 0;
      if (m_idx >= int(m_size)) {
        m_idx = m_size - 1;
        const auto new_p = std::sqrt(powr<2>(p) + powr<2>(m - matsubara_values.forward(m_idx)));
        p_idx = logarithmic_coordinates.backward(new_p);
      } else if (m_idx < 0) {
        m_idx = 0;
        const auto new_p = std::sqrt(powr<2>(p) + powr<2>(m - matsubara_values.forward(m_idx)));
        p_idx = logarithmic_coordinates.backward(new_p);
      } else
        p_idx = logarithmic_coordinates.backward(p);
      return {m_idx, p_idx};
    }

    device::array<uint, 2> KOKKOS_FORCEINLINE_FUNCTION from_continuous_index(auto i) const
    {
      device::array<uint, 2> idx;
      // calculate the index for each coordinate system
      idx[0] = i / grid_extent; // Matsubara index
      idx[1] = i % grid_extent; // Logarithmic index

      return idx;
    }

    uint size() const { return m_size * grid_extent; }
    device::array<uint, 2> sizes() const { return {{m_size, grid_extent}}; }

    const NT m_start, m_stop, m_T, p_start, p_stop, p_bias;

  private:
    const uint grid_extent, m_size;

    FermionicMatsubaraValues<Idx, NT> matsubara_values;
    LogarithmicCoordinates1D<NT> logarithmic_coordinates;
  };
} // namespace DiFfRG
