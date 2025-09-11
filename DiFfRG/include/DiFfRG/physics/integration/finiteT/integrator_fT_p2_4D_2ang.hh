#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

namespace DiFfRG
{
  namespace internal
  {
    template <typename NT, typename KERNEL> class Transform_fT_p2_4D_2ang
    {
    public:
      using ctype = typename get_type::ctype<NT>;

      static constexpr ctype int_prefactor = powr<-3>((ctype)2 * M_PI); // fourier factor

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q2, const ctype cos1, const ctype phi, const ctype q0,
                                                   const T &...t)
      {
        using namespace DiFfRG::compute;

        const ctype q = sqrt(q2);
        const ctype int_element = powr<3 - 2>(q) / (ctype)2; // from p^2 integral

        const NT result = KERNEL::kernel(q, cos1, phi, q0, t...);

        return int_prefactor * int_element * result;
      }

      template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION NT constant(const T &...t)
      {
        return KERNEL::constant(t...);
      }
    };
  } // namespace internal

  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim == 4)
  class Integrator_fT_p2_4D_2ang
      : public QuadratureIntegrator_fT<4, NT, internal::Transform_fT_p2_4D_2ang<NT, KERNEL>, ExecutionSpace>
  {
    using Base = QuadratureIntegrator_fT<4, NT, internal::Transform_fT_p2_4D_2ang<NT, KERNEL>, ExecutionSpace>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    /**
     * @brief Execution space to be used for the integration, e.g. GPU_exec, TBB_exec.
     */
    using execution_space = ExecutionSpace;

    Integrator_fT_p2_4D_2ang(QuadratureProvider &quadrature_provider, const JSONValue &json)
      requires provides_regulator<KERNEL>
        : Integrator_fT_p2_4D_2ang(quadrature_provider,
                                   internal::make_int_grid<3>(json, {"x_order", "cos1_order", "phi_order"}),
                                   optimize_x_extent<typename KERNEL::Regulator>(json), json.get_double("T", 1.0))
    {
    }

    Integrator_fT_p2_4D_2ang(QuadratureProvider &quadrature_provider, const std::array<size_t, 3> grid_size,
                             ctype x_extent = 2., ctype T = 1, ctype typical_E = 1)
        : Base(quadrature_provider, grid_size, {0, -1, 0}, {x_extent, 1, 2 * M_PI},
               {QuadratureType::legendre, QuadratureType::legendre, QuadratureType::legendre}, T, typical_E),
          x_extent(x_extent), k(1.)
    {
    }

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, -1, 0}, {x_extent * powr<2>(k), 1, 2 * M_PI});
    }

    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, -1, 0}, {x_extent * powr<2>(k), 1, 2 * M_PI});
      Base::set_typical_E(k); // update typical energy
    }

    void set_typical_E(ctype typical_E) { Base::set_typical_E(typical_E); }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
