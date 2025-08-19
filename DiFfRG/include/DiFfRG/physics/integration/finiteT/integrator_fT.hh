
#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

namespace DiFfRG
{
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim == 1)
  class Integrator_fT : public QuadratureIntegrator_fT<1, NT, KERNEL, ExecutionSpace>
  {
    using Base = QuadratureIntegrator_fT<1, NT, KERNEL, ExecutionSpace>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    /**
     * @brief Execution space to be used for the integration, e.g. GPU_exec, TBB_exec, or OpenMP_exec.
     */
    using execution_space = ExecutionSpace;

    Integrator_fT(QuadratureProvider &quadrature_provider, const JSONValue &json)
      requires provides_regulator<KERNEL>
        : Integrator_fT(quadrature_provider, internal::make_int_grid<2>(json, {"x_order", "cos1_order"}),
                        json.get_double("T", 1.0))
    {
    }

    Integrator_fT(QuadratureProvider &quadrature_provider, ctype T = 1, ctype typical_E = 1)
        : Base(quadrature_provider, {}, {}, {}, {}, T, typical_E)
    {
    }

    void set_k(ctype k)
    {
      Base::set_typical_E(k); // update typical energy
    }

    void set_typical_E(ctype typical_E) { Base::set_typical_E(typical_E); }
  };
} // namespace DiFfRG
