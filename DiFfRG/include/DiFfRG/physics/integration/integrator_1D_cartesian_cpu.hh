#pragma once

// standard library
#include <future>

// external libraries
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator1DCartesianTBB
  {
  public:
    using ctype = typename get_type::ctype<NT>;
    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent = 0., const uint max_block_size = 0, const ctype qx_min = -M_PI,
                             const ctype qx_max = M_PI)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0], x_extent, qx_min, qx_max)
    {
    }

    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const JSONValue &json)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0], 0.,
                                   json.get_double("/discretization/integration/qx_min", -M_PI),
                                   json.get_double("/discretization/integration/qx_max", M_PI))
    {
    }

    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent, const JSONValue &json)
        : Integrator1DCartesianTBB(quadrature_provider, grid_size[0], x_extent,
                                   json.get_double("/discretization/integration/qx_min", -M_PI),
                                   json.get_double("/discretization/integration/qx_max", M_PI))
    {
    }

    Integrator1DCartesianTBB(QuadratureProvider &quadrature_provider, const uint grid_size, const ctype x_extent = 0.,
                             const ctype qx_min = -M_PI, const ctype qx_max = M_PI)
        : grid_size(grid_size), x_quadrature_p(quadrature_provider.get_points<ctype>(grid_size)),
          x_quadrature_w(quadrature_provider.get_weights<ctype>(grid_size))
    {
      this->qx_min = qx_min;
      this->qx_extent = qx_max - qx_min;
    }

    /**
     * @brief Set the minimum value of the qx integration range.
     */
    void set_qx_min(const ctype qx_min)
    {
      this->qx_extent = this->qx_extent - qx_min + this->qx_min;
      this->qx_min = qx_min;
    }

    /**
     * @brief Set the maximum value of the qx integration range.
     */
    void set_qx_max(const ctype qx_max) { this->qx_extent = qx_max - qx_min; }

    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      constexpr int d = 1;
      constexpr ctype int_element = powr<-d>(2 * (ctype)M_PI); // fourier factor

      return KERNEL::constant(k, t...) + tbb::parallel_reduce(
                                             tbb::blocked_range<uint>(0, grid_size), NT(0),
                                             [&](const tbb::blocked_range<uint> &r, NT value) -> NT {
                                               for (uint idx_x = r.begin(); idx_x != r.end(); ++idx_x) {
                                                 const ctype q = qx_min + qx_extent * x_quadrature_p[idx_x];
                                                 const ctype weight = qx_extent * x_quadrature_w[idx_x];
                                                 value += int_element * weight * KERNEL::kernel(q, k, t...);
                                               }
                                               return value;
                                             },
                                             [&](NT x, NT y) -> NT { return x + y; });
    }

    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    const uint grid_size;

    ctype qx_min = -M_PI;
    ctype qx_extent = 2. * M_PI;

    const std::vector<ctype> &x_quadrature_p;
    const std::vector<ctype> &x_quadrature_w;
  };
} // namespace DiFfRG