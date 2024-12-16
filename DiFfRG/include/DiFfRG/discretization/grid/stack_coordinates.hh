#pragma once

// standard library
#include <cmath>
#include <stdexcept>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>

namespace DiFfRG
{
  template <typename Idx> class IndexStack
  {
  public:
    using ctype = Idx;
    static constexpr uint dim = 1;

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
    Idx __forceinline__ __device__ __host__ forward(const Idx x) const { return x + start; }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    Idx __forceinline__ __device__ __host__ backward(const Idx y) const { return y - start; }

    uint size() const { return m_size; }

    const Idx start, stop;

  private:
    const uint m_size;
  };

  template <typename Idx, typename NT> class BosonicMatsubaraValues
  {
  public:
    using ctype = NT;
    static constexpr uint dim = 1;

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
    NT __forceinline__ __device__ __host__ forward(const Idx &x) const { return NT(x + start) * 2. * M_PI * T; }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    Idx __forceinline__ __device__ __host__ backward(const NT &y) const
    {
      return Idx(std::round(y / (2. * M_PI * T))) - start;
    }

    uint size() const { return m_size; }

    const Idx start, stop;
    const NT T;

  private:
    const uint m_size;
  };

  template <typename Idx, typename NT> class FermionicMatsubaraValues
  {
  public:
    using ctype = NT;
    static constexpr uint dim = 1;

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
    NT __forceinline__ __device__ __host__ forward(const Idx &x) const { return (NT(x + start) + 0.5) * 2. * M_PI * T; }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    Idx __forceinline__ __device__ __host__ backward(const NT &y) const
    {
      return Idx(std::round((y - M_PI * T) / (2. * M_PI * T))) - start;
    }

    uint size() const { return m_size; }

    const Idx start, stop;
    const NT T;

  private:
    const uint m_size;
  };
} // namespace DiFfRG
