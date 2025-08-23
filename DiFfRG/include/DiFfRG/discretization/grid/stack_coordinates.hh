#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>

namespace DiFfRG
{
  template <typename Idx> class IndexStack
  {
  public:
    using ctype = Idx;
    static constexpr size_t dim = 1;

    IndexStack(Idx start, Idx stop) : start(start), stop(stop), m_size(stop - start)
    {
      if (start > stop)
        throw std::invalid_argument("start must be smaller than stop; start = " + std::to_string(start) +
                                    ", stop = " + std::to_string(stop));
    }

    template <typename Idx2> IndexStack(const LinearCoordinates1D<Idx2> &other) : IndexStack(other.start, other.stop) {}

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    Idx KOKKOS_FORCEINLINE_FUNCTION forward(const Idx x) const { return x + start; }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    Idx KOKKOS_FORCEINLINE_FUNCTION backward(const Idx y) const { return y - start; }

    size_t size() const { return m_size; }

    const Idx start, stop;

    std::string to_string() const { return "IndexStack(" + std::to_string(start) + ", " + std::to_string(stop) + ")"; }

  private:
    const size_t m_size;
  };

  template <typename Idx = int, typename NT = double> class BosonicMatsubaraValues
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 1;

    BosonicMatsubaraValues(Idx start, Idx stop, NT T) : start(start), stop(stop), T(T), m_size(stop - start)
    {
      if (start > stop)
        throw std::invalid_argument("start must be smaller than stop; start = " + std::to_string(start) +
                                    ", stop = " + std::to_string(stop));
    }

    template <typename Idx2, typename NT2>
    BosonicMatsubaraValues(const BosonicMatsubaraValues<Idx2, NT2> &other)
        : BosonicMatsubaraValues(other.start, other.stop, other.T)
    {
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    NT KOKKOS_FORCEINLINE_FUNCTION forward(const Idx &x) const { return NT(x + start) * 2. * M_PI * T; }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    Idx KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const { return Idx(std::round(y / (2. * M_PI * T))) - start; }

    size_t size() const { return m_size; }

    const Idx start, stop;
    const NT T;

    std::string to_string() const
    {
      return "BosonicMatsubaraValues(" + std::to_string(start) + ", " + std::to_string(stop) + ", " +
             std::to_string(T) + ")";
    }

  private:
    const size_t m_size;
  };

  template <typename Idx = int, typename NT = double> class FermionicMatsubaraValues
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 1;

    FermionicMatsubaraValues(Idx start, Idx stop, NT T) : start(start), stop(stop), T(T), m_size(stop - start)
    {
      if (start > stop)
        throw std::invalid_argument("start must be smaller than stop; start = " + std::to_string(start) +
                                    ", stop = " + std::to_string(stop));
    }

    template <typename Idx2, typename NT2>
    FermionicMatsubaraValues(const FermionicMatsubaraValues<Idx2, NT2> &other)
        : FermionicMatsubaraValues(other.start, other.stop, other.T)
    {
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    NT KOKKOS_FORCEINLINE_FUNCTION forward(const Idx &x) const { return (NT(x + start) + 0.5) * 2. * M_PI * T; }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    Idx KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const
    {
      return Idx(std::round((y - M_PI * T) / (2. * M_PI * T))) - start;
    }

    size_t size() const { return m_size; }

    const Idx start, stop;
    const NT T;

    std::string to_string() const
    {
      return "FermionicMatsubaraValues(" + std::to_string(start) + ", " + std::to_string(stop) + ", " +
             std::to_string(T) + ")";
    }

  private:
    const size_t m_size;
  };
} // namespace DiFfRG
