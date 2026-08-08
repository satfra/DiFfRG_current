#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

namespace DiFfRG
{
  namespace internal
  {
    template <int dim, typename NT, typename KERNEL> class Transform_fT_p2_1ang
    {
    public:
      static constexpr int sdim = dim - 1; // spatial dimension
      using ctype = typename get_type::ctype<NT>;

      // The polar angle (spatial loop vs. external momentum) of a 2-point loop lives in the sdim
      // spatial dimensions and carries the zonal measure (1-c^2)^{(sdim-3)/2} dc, supplied by the
      // Gauss-Jacobi(alpha=beta=(sdim-3)/2) angular quadrature (see the class below). The remaining
      // (sdim-2)-sphere gives S_{sdim-1} = S_d_prec(sdim-1).
      static constexpr ctype int_prefactor = S_d_prec<ctype>(sdim - 1)     // remaining solid angle after the polar angle
                                             / powr<sdim>(2 * (ctype)M_PI); // fourier factor

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q, const ctype cos, const ctype q0, const T &...t)
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<sdim - 1>(q); // from p integral
        const NT result = KERNEL::kernel(q, cos, q0, t...);

        return int_prefactor * int_element * result;
      }

      template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION NT constant(const T &...t)
      {
        return KERNEL::constant(t...);
      }
    };

  } // namespace internal

  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim >= 3)
  class Integrator_fT_p2_1ang
      : public QuadratureIntegrator_fT<3, NT, internal::Transform_fT_p2_1ang<dim, NT, KERNEL>, ExecutionSpace>
  {
    using Base = QuadratureIntegrator_fT<3, NT, internal::Transform_fT_p2_1ang<dim, NT, KERNEL>, ExecutionSpace>;

    static constexpr int sdim = dim - 1; // spatial dimension

    // Gauss-Jacobi quadrature on [-1,1] with weight (1-c^2)^{(sdim-3)/2} for the spatial polar angle.
    // Dimension-agnostic: reduces to Chebyshev-1 (sdim=2), Legendre (sdim=3), Chebyshev-2 (sdim=4), ...
    static QuadratureType polar_quadrature()
    {
      QuadratureType t(QuadratureType::jacobi);
      t.a = static_cast<double>(-1);
      t.b = static_cast<double>(1);
      t.alpha = (static_cast<double>(sdim) - 3.) / 2.;
      t.beta = t.alpha;
      return t;
    }

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    Integrator_fT_p2_1ang(QuadratureProvider &quadrature_provider, const ConfigTree &json)
      requires provides_regulator<KERNEL>
        : Integrator_fT_p2_1ang(quadrature_provider, internal::make_int_grid<2, NT>(json, {"x_order", "cos1_order"}),
                                optimize_x_extent<typename KERNEL::Regulator>(json), json.get_double("/physical/T", 1.0))
    {
    }

    Integrator_fT_p2_1ang(QuadratureProvider &quadrature_provider, const std::array<size_t, 2> grid_size,
                          ctype x_extent = 2., ctype T = 1, ctype typical_E = 1)
        : Base(quadrature_provider, grid_size, {0, 0.}, {std::sqrt(x_extent), 1.},
               {QuadratureType(QuadratureType::legendre), polar_quadrature()}, T, typical_E),
          x_extent(x_extent), k(1.)
    {
    }

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, 0.}, {std::sqrt(x_extent) * k, 1.});
    }

    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, 0.}, {std::sqrt(x_extent) * k, 1.});
      Base::set_typical_E(k); // update typical energy
    }

    void set_typical_E(ctype typical_E) { Base::set_typical_E(typical_E); }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
