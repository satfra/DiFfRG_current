#pragma once

#ifdef __CUDACC__

// standard library
#include <future>

// external libraries
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator1DCartesianGPU
  {
  public:
    using ctype = typename get_type::ctype<NT>;
    template <typename... T> struct functor {
    public:
      functor(const ctype *x_quadrature_p, const ctype *x_quadrature_w, const ctype qx_min, const ctype qx_extent,
              const ctype k, T... t)
          : x_quadrature_p(x_quadrature_p), x_quadrature_w(x_quadrature_w), qx_min(qx_min), qx_extent(qx_extent), k(k),
            t(t...)
      {
      }

      __device__ NT operator()(const uint idx) const
      {
        constexpr int d = 1;
        constexpr ctype int_element = powr<-d>(2 * (ctype)M_PI); // fourier factor

        const ctype q = qx_min + qx_extent * x_quadrature_p[idx];
        const ctype weight = qx_extent * x_quadrature_w[idx];

        const NT res = std::apply([&](auto &&...args) { return KERNEL::kernel(q, k, args...); }, t);
        return int_element * res * weight;
      }

    private:
      const ctype *x_quadrature_p;
      const ctype *x_quadrature_w;
      const ctype qx_min, qx_extent;
      const ctype k;
      const std::tuple<T...> t;
    };

    Integrator1DCartesianGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent = 0., const uint max_block_size = 0, const ctype qx_min = -M_PI,
                             const ctype qx_max = M_PI)
        : Integrator1DCartesianGPU(quadrature_provider, grid_size[0], x_extent, qx_min, qx_max)
    {
    }

    Integrator1DCartesianGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const JSONValue &json)
        : Integrator1DCartesianGPU(quadrature_provider, grid_size[0], 0.,
                                   json.get_double("/discretization/integration/qx_min", -M_PI),
                                   json.get_double("/discretization/integration/qx_max", M_PI))
    {
    }

    Integrator1DCartesianGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent, const JSONValue &json)
        : Integrator1DCartesianGPU(quadrature_provider, grid_size[0], x_extent,
                                   json.get_double("/discretization/integration/qx_min", -M_PI),
                                   json.get_double("/discretization/integration/qx_max", M_PI))
    {
    }

    Integrator1DCartesianGPU(QuadratureProvider &quadrature_provider, const uint grid_size, const ctype x_extent = 0.,
                             const ctype qx_min = -M_PI, const ctype qx_max = M_PI)
        : grid_size(grid_size)
    {
      ptr_x_quadrature_p = quadrature_provider.get_device_points<ctype>(grid_size);
      ptr_x_quadrature_w = quadrature_provider.get_device_weights<ctype>(grid_size);

      this->qx_min = qx_min;
      this->qx_extent = qx_max - qx_min;
    }

    Integrator1DCartesianGPU(const Integrator1DCartesianGPU &other)
        : grid_size(other.grid_size), ptr_x_quadrature_p(other.ptr_x_quadrature_p),
          ptr_x_quadrature_w(other.ptr_x_quadrature_w)
    {
      qx_min = other.qx_min;
      qx_extent = other.qx_extent;
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
      return KERNEL::constant(k, t...) +
             thrust::transform_reduce(thrust::cuda::par.on(rmm::cuda_stream_per_thread.value()),
                                      thrust::make_counting_iterator<uint>(0),
                                      thrust::make_counting_iterator<uint>(grid_size),
                                      functor<T...>(ptr_x_quadrature_p, ptr_x_quadrature_w, qx_min, qx_extent, k, t...),
                                      NT(0), thrust::plus<NT>());
    }

    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }

  private:
    const uint grid_size;

    ctype qx_min = -M_PI;
    ctype qx_extent = 2. * M_PI;

    const ctype *ptr_x_quadrature_p;
    const ctype *ptr_x_quadrature_w;
  };
} // namespace DiFfRG

#else

#ifdef USE_CUDA

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator1DCartesianGPU;
}

#else

#include <DiFfRG/physics/integration/integrator_1D_cartesian_cpu.hh>

namespace DiFfRG
{
  template <typename NT, typename KERNEL> class Integrator1DCartesianGPU : public IntegratorTBB<d, NT, KERNEL>
  {
  public:
    using ctype = typename get_type::ctype<NT>;

    Integrator1DCartesianGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent, const uint max_block_size = 256, const ctype qx_min = -M_PI,
                             const ctype qx_max = M_PI)
        : Integrator1DCartesianTBB<NT, KERNEL>(quadrature_provider, grid_size[0], x_extent, qx_min, qx_max)
    {
      (void)max_block_size;
    }

    Integrator1DCartesianGPU(QuadratureProvider &quadrature_provider, const uint grid_size, const ctype x_extent,
                             const uint max_block_size = 256, const ctype qx_min = -M_PI, const ctype qx_max = M_PI)
        : Integrator1DCartesianTBB<NT, KERNEL>(quadrature_provider, grid_size, x_extent, x_extent, qx_min, qx_max)
    {
      (void)max_block_size;
    }

    Integrator1DCartesianGPU(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                             const ctype x_extent, const JSONValue &json)
        : Integrator1DCartesianTBB<NT, KERNEL>(quadrature_provider, grid_size, x_extent, json)
    {
    }
  };
} // namespace DiFfRG

#endif

#endif