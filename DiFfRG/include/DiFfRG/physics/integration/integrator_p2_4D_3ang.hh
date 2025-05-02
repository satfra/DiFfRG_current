#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/integrator_4D.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  namespace internal
  {
    template <typename NT, typename KERNEL> class Transform_p2_4D_3ang
    {
    public:
      using ctype = typename get_type::ctype<NT>;

      static constexpr ctype int_prefactor = powr<-4>((ctype)2 * M_PI); // fourier factor

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q2, const ctype cos1, const ctype cos2, const ctype phi,
                                                   const T &...t)
      {
        using namespace DiFfRG::compute;

        const ctype q = sqrt(q2);
        const ctype int_element = powr<4 - 2>(q) / (ctype)2; // from p^2 integral

        // sqrt(1. - powr<2>(cos1)); // the cos1 integral jacobian is already included
        // in the chebyshev2 quadrature

        const NT result = KERNEL::kernel(q, cos1, cos2, phi, t...);

        return int_prefactor * int_element * result;
      }

      template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION NT constant(const T &...t)
      {
        return KERNEL::constant(t...);
      }
    };

  } // namespace internal

  template <typename NT, typename KERNEL, typename ExecutionSpace>
  class Integrator_p2_4D_3ang : public Integrator4D<NT, internal::Transform_p2_4D_3ang<NT, KERNEL>, ExecutionSpace>
  {
    using Base = Integrator4D<NT, internal::Transform_p2_4D_3ang<NT, KERNEL>, ExecutionSpace>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    Integrator_p2_4D_3ang(QuadratureProvider &quadrature_provider, const std::array<uint, 4> grid_size,
                          ctype x_extent = 2.)
        : Base(quadrature_provider, grid_size, {0, 0, -1, 0}, {x_extent, 1, 1, 2 * M_PI},
               {QuadratureType::legendre, QuadratureType::chebyshev2, QuadratureType::legendre,
                QuadratureType::legendre}),
          x_extent(x_extent), k(1.)
    {
    }

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, 0, -1, 0}, {x_extent * powr<2>(k), 1, 1, 2 * M_PI});
    }

    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, 0, -1, 0}, {x_extent * powr<2>(k), 1, 1, 2 * M_PI});
    }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
