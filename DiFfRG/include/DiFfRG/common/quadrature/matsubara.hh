#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>

// C++ standard library
#include <vector>

namespace DiFfRG
{
  /**
   * @brief A quadrature rule for (bosonic) Matsubara frequencies, based on the method of Monien [1]. This class
   * provides nodes and weights for the summation
   * \f[
   * T \sum_{n=\in \mathbb{Z}} f(2\pi n T) \approx \sum_{n=1}^{N} w_i (f(x_i) + f(-x_i)) + T f(0)
   * \f]
   *
   * [1] H. Monien, "Gaussian quadrature for sums: a rapidly convergent summation scheme", Math. Comp. 79, 857 (2010).
   * doi:10.1090/S0025-5718-09-02289-3
   *
   * @tparam NT numeric type to be used for all calculations
   */
  template <typename NT> class MatsubaraQuadrature
  {
  public:
    /**
     * @brief Calculate the number of nodes needed for a given temperature and typical energy scale.
     *
     * @param T The temperature.
     * @param typical_E A typical energy scale.
     * @param step The step size of considered node sizes (e.g. step=2 implies only even numbers of nodes).
     * @return int The number of nodes needed. If the number is negative, the T=0-limit has been reached (usually if
     * typical_E / T > 4.4e+2, which is 64 nodes).
     */
    int predict_size(const NT T, const NT typical_E = 1., const int step = 2);

    /**
     * @brief Create a new quadrature rule for Matsubara frequencies.
     *
     * @param T The temperature.
     * @param typical_E A typical energy scale, which determines the number of nodes in the quadrature rule.
     * @param step The step size of considered node sizes (e.g. step=2 implies only even numbers of nodes).
     * @param min_size Minimum number of nodes.
     * @param max_size Maximum number of nodes.
     */
    MatsubaraQuadrature(const NT T, const NT typical_E = 1., const int step = 2, const int min_size = 0,
                        const int max_size = 256, const int vacuum_quad_size = 48, const int precision_factor = 1);

    MatsubaraQuadrature();

    /**
     * @brief Update the quadrature rule with new parameters.
     *
     * @param T  The temperature.
     * @param typical_E A typical energy scale, which determines the number of nodes in the quadrature rule.
     * @param step The step size of considered node sizes (e.g. step=2 implies only even numbers of nodes).
     * @param min_size Minimum number of nodes.
     * @param max_size Maximum number of nodes.
     */
    void reinit(const NT T, const NT typical_E = 1., const int step = 2, const int min_size = 0,
                const int max_size = 256, const int vacuum_quad_size = 48, const int precision_factor = 1);

    /**
     * @brief Get the size of the quadrature rule.
     */
    size_t size() const;

    /**
     * @brief Get the temperature of the quadrature rule.
     */
    NT get_T() const;

    /**
     * @brief Get the typical energy scale of the quadrature rule.
     */
    NT get_typical_E() const;

    /**
     * @brief Compute a matsubara sum of a given function.
     *
     * @tparam F The function type.
     * @param f The function to be summed. Must have signature NT f(NT x).
     * @return NT The Matsubara sum of the function.
     */
    template <typename F> auto sum(const F &f) const
    {
      auto sum = T * f(static_cast<NT>(0));
      for (int i = 0; i < m_size; ++i)
        sum += host_weights[i] * (f(host_nodes[i]) + f(-host_nodes[i]));
      return sum;
    }

    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> nodes() const
    {
      if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultExecutionSpace::memory_space>) {
        return device_nodes;
      } else if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultHostExecutionSpace::memory_space>) {
        return host_nodes;
      } else {
        throw std::runtime_error("Invalid memory space");
      }
    }

    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> weights() const
    {
      if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultExecutionSpace::memory_space>) {
        return device_weights;
      } else if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultHostExecutionSpace::memory_space>) {
        return host_weights;
      } else {
        throw std::runtime_error("Invalid memory space");
      }
    }

  private:
    NT T, typical_E;

    Kokkos::View<NT *, GPU_memory> device_nodes;
    Kokkos::View<NT *, GPU_memory> device_weights;

    Kokkos::View<NT *, CPU_memory> host_nodes;
    Kokkos::View<NT *, CPU_memory> host_weights;

    /**
     * @brief The number of nodes in the quadrature rule.
     */
    int m_size;

    void write_data(const std::vector<NT> &x, const std::vector<NT> &w);

    /**
     * @brief Construct a quadrature rule for T=0.
     */
    void reinit_0();

    int vacuum_quad_size;
    int precision_factor;
  };
} // namespace DiFfRG