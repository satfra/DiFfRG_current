#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  namespace internal
  {
    template <typename NT, typename KERNEL> class Transform_p2_4D_2ang
    {
    public:
      using ctype = typename get_type::ctype<NT>;

      static constexpr ctype int_prefactor = 2 * M_PI                    // phi integral
                                             / powr<4>((ctype)2 * M_PI); // fourier factor

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q2, const ctype cos1, const ctype cos2, const T &...t)
        requires provides_kernel<NT, KERNEL, ctype, 3, T...>
      {
        using namespace DiFfRG::compute;

        const ctype q = sqrt(q2);
        const ctype int_element = powr<4 - 2>(q) / (ctype)2; // from p^2 integral

        // sqrt(1. - powr<2>(cos1)); // the cos1 integral jacobian is already included
        // in the chebyshev2 quadrature

        const NT result = KERNEL::kernel(q, cos1, cos2, t...);

        return int_prefactor * int_element * result;
      }

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT constant(const T &...t)
        requires provides_constant<NT, KERNEL, T...>
      {
        return KERNEL::constant(t...);
      }
    };

  } // namespace internal

  /**
   * @brief Integrator_p2_1ang integrates a kernel \f$K(p,\cos_1,\cos_2,\ldots)\f$ depending on the radial momentum
   * \f$p\f$ and two angles on \f$[0,\pi]\f$ as
   * $$
   * \frac{2\pi}{(2\pi)^{d}} \,\int_0^\pi d\cos_1\,\int_0^\pi d\cos_2\,\int_0^\infty dp^2 p^{d-2}
   * K(p,\cos_1,\cos_2,\ldots)
   * $$
   * where \f$S_d\f$ is the solid angle in \f$d\f$ dimensions.
   *
   * @tparam dim dimension of the momentum space, i.e. $d$ in the above equation
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, which must provide the static methods `kernel` and `constant`
   * @tparam ExecutionSpace can be any execution space, e.g. GPU_exec, TBB_exec, or OpenMP_exec.
   */
  template <typename NT, typename KERNEL, typename ExecutionSpace>
  class Integrator_p2_4D_2ang
      : public QuadratureIntegrator<3, NT, internal::Transform_p2_4D_2ang<NT, KERNEL>, ExecutionSpace>
  {
    using Base = QuadratureIntegrator<3, NT, internal::Transform_p2_4D_2ang<NT, KERNEL>, ExecutionSpace>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    Integrator_p2_4D_2ang(QuadratureProvider &quadrature_provider, const std::array<uint, 3> grid_size,
                          ctype x_extent = 2.)
        : Base(quadrature_provider, grid_size, {0, 0, -1.}, {x_extent, 1, 1},
               {QuadratureType::legendre, QuadratureType::chebyshev2, QuadratureType::legendre}),
          x_extent(x_extent), k(1.)
    {
    }

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, 0, -1}, {x_extent * powr<2>(k), 1, 1});
    }

    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, 0, -1}, {x_extent * powr<2>(k), 1, 1});
    }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
