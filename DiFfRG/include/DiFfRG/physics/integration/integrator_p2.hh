#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/integrator_1D.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  namespace internal
  {
    template <int dim, typename NT, typename KERNEL> class Transform_p2
    {
    public:
      using ctype = typename get_type::ctype<NT>;

      static constexpr ctype int_prefactor = S_d_prec<ctype>(dim)          // solid nd angle
                                             / powr<dim>(2 * (ctype)M_PI); // fourier factor

      template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q2, const T &...t)
      {
        using namespace DiFfRG::compute;

        const ctype q = sqrt(q2);
        const ctype int_element = (powr<dim - 2>(q) / (ctype)2); // from p^2 integral
        const NT result = KERNEL::kernel(q, t...);

        return int_prefactor * int_element * result;
      }

      template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION NT constant(const T &...t)
      {
        return KERNEL::constant(t...);
      }
    };

  } // namespace internal

  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
  class Integrator_p2 : public Integrator1D<NT, internal::Transform_p2<dim, NT, KERNEL>, ExecutionSpace>
  {
    using Base = Integrator1D<NT, internal::Transform_p2<dim, NT, KERNEL>, ExecutionSpace>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    Integrator_p2(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size, ctype x_extent = 2.)
        : Base(quadrature_provider, grid_size, {0}, {x_extent}), x_extent(x_extent), k(1.)
    {
    }

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0}, {x_extent * powr<2>(k)});
    }

    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0}, {x_extent * powr<2>(k)});
    }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
