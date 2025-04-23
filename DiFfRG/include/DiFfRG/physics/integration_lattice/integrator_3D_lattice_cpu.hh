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
   * @brief Integration of an arbitrary 3D function from (qx_min, qy_min) to (qx_max, qy_max) using TBB.
   *
   * @tparam NT The numerical type of the result.
   * @tparam KERNEL The kernel to integrate.
   */
  template <typename NT, typename KERNEL> class Integrator3DCartesianTBB
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;

    /**
     * @brief Construct a new Integrator3DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_sizes The number of grid points in x and y direction.
     * @param x_extent The extent of the x integration range. This argument is not used, but kept for compatibility with
     * flow classes.
     * @param json The JSON object to read additional parameters from.
     */
    Integrator3DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 3> grid_sizes,
                             const ctype x_extent, const JSONValue &)
        : Integrator3DCartesianTBB(quadrature_provider, grid_sizes)
    {
    }

    /**
     * @brief Construct a new Integrator3DCartesianTBB object
     *
     * @param quadrature_provider The quadrature provider to use.
     * @param grid_sizes The number of grid points in x and y direction.
     * @param x_extent The extent of the x integration range. This argument is not used, but kept for compatibility with
     * flow classes.
     * @param max_block_size The maximum block size to use on GPU. This argument is not used, but kept for
     * compatibility.
     * @param qx_min The minimum value of the x integration range.
     * @param qy_min The minimum value of the y integration range.
     * @param qx_max The maximum value of the x integration range.
     * @param qy_max The maximum value of the y integration range.
     */
    Integrator3DCartesianTBB(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes,
                             const ctype x_extent = 0., const uint max_block_size = 0)
        : grid_sizes(grid_sizes)
    {
      (void)max_block_size;
      (void)x_extent;
    }

    /**
     * @brief Get the value of the integral.
     *
     * @param k The current RG scale.
     * @param t The additional parameters for the kernel.
     * @return The value of the integral.
     */
    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      constexpr ctype int_element = 1.;

      const auto constant = KERNEL::constant(k, t...);
      return constant +
             tbb::parallel_reduce(
                 tbb::blocked_range3d<uint, uint, uint>(0, grid_sizes[0], 0, grid_sizes[1], 0, grid_sizes[2]), NT(0.),
                 [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                   for (uint idx_x = r.pages().begin(); idx_x != r.pages().end(); ++idx_x) {
                     for (uint idx_y = r.rows().begin(); idx_y != r.rows().end(); ++idx_y) {
                       for (uint idx_z = r.cols().begin(); idx_z != r.cols().end(); ++idx_z) {
                         value += int_element * KERNEL::kernel(idx_x, idx_y, idx_z, k, t...);
                       }
                     }
                   }
                   return value;
                 },
                 [&](NT x, NT y) -> NT { return x + y; });
    }

    /**
     * @brief Get the value of the integral asynchronously.
     *
     * @param k The current RG scale.
     * @param t The additional parameters for the kernel.
     * @return An std::future<NT> which returns the value of the integral.
     */
    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    const std::array<uint, 3> grid_sizes;
  };
} // namespace DiFfRG