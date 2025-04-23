#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  /**
   * @brief Integration of an arbitrary 1D function from qx_min to qx_max using TBB.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <typename NT, typename KERNEL> class Integrator1DCartesianTBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    /**
     * @brief Construct a new Integrator1DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_size The number of grid points in x direction.
     * @param x_extent The extent of the x integration range. This argument is not used, but kept for compatibility with
     * flow classes.
     * @param qx_min The minimum value of the x integration range.
     * @param qx_max The maximum value of the x integration range.
     */
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent = 0., const uint max_block_size = 0)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0])
    {
    }

    /**
     * @brief Construct a new Integrator1DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_size The number of grid points in x direction.
     * @param x_extent This argument is not used, but kept for compatibility with flow classes.
     * @param max_block_size The maximum block size to use on GPU. This argument is not used, but kept for
     * compatibility.
     * @param qx_min The minimum value of the x integration range.
     * @param qx_max The maximum value of the x integration range.
     */
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const JSONValue &)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0], 0.)
    {
    }

    /**
     * @brief Construct a new Integrator1DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_size The number of grid points in x direction.
     * @param x_extent This argument is not used, but kept for compatibility with flow classes.
     * @param qx_min The minimum value of the x integration range.
     * @param qx_max The maximum value of the x integration range.
     */
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size, const ctype,
                             const JSONValue &)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0])
    {
    }

    /**
     * @brief Construct a new Integrator1DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_size The number of grid points in x direction.
     * @param x_extent This argument is not used, but kept for compatibility with flow classes.
     * @param qx_min The minimum value of the x integration range.
     * @param qx_max The maximum value of the x integration range.
     */
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const uint grid_size) : grid_size(grid_size) {}

    /**
     * @brief Get the integration result.
     *
     * @tparam T The types of the additional arguments to the kernel.
     * @param k The current RG scale.
     * @param t The additional arguments of the kernel.
     * @return NT The result of the integration.
     */
    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      constexpr ctype int_element = 1.;

      return KERNEL::constant(k, t...) + tbb::parallel_reduce(
                                             tbb::blocked_range<uint>(0, grid_size), NT(0),
                                             [&](const tbb::blocked_range<uint> &r, NT value) -> NT {
                                               for (uint idx_x = r.begin(); idx_x != r.end(); ++idx_x) {
                                                 value += int_element * KERNEL::kernel(idx_x, k, t...);
                                               }
                                               return value;
                                             },
                                             [&](NT x, NT y) -> NT { return x + y; });
    }

    /**
     * @brief Get the integration result asynchronously.
     *
     * @tparam T The types of the additional arguments to the kernel.
     * @param k The current RG scale.
     * @param t The additional arguments of the kernel.
     * @return An std::future<NT> which returns the result of the integration.
     */
    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    const uint grid_size;
  };
} // namespace DiFfRG