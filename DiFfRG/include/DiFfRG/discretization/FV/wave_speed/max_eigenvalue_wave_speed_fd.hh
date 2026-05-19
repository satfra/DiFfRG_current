#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include <DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      /**
       * @brief Wave-speed strategy that uses the analytical (eigenvector-perturbation)
       * derivative formula from MaxEigenvalueWaveSpeed but obtains the Hessian H via
       * **finite differences** on `Real<1, double>`-AD Jacobians instead of via
       * `Real<2, double>` forward-mode AD seeded through the integrator.
       *
       * Rationale: identical to LDG's approach. The Real<2, double> AD path propagates
       * (val, ε, ε²) triples through every operation in the flux integrand. For
       * quadrature-based fluxes the ε² coefficient accumulates roundoff that also
       * contaminates the extracted ε coefficient (i.e. J), even though the AD math is
       * formally exact. Replacing the AD H with a central FD on the Real<1>-AD J keeps
       * the full numerical-flux Jacobian (so we don't lose the wave-speed derivative
       * contribution to da, unlike MaxEigenvalueWaveSpeedZeroDeriv) but uses only
       * first-order AD on the model's flux function.
       *
       * compute_speeds, compute_speed_derivatives, compute_selected_speed_derivatives
       * delegate to MaxEigenvalueWaveSpeed — only the H source differs, which is
       * handled by compute_kt_numflux_jacobian dispatching on `hessian_via_fd`.
       */
      struct MaxEigenvalueWaveSpeedFD {
        static constexpr bool needs_hessian = true;
        static constexpr bool hessian_via_fd = true;

        template <typename NumberType, int dim, size_t n_components>
        static std::array<NumberType, dim>
        compute_speeds(const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
                       const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus)
        {
          return MaxEigenvalueWaveSpeed::compute_speeds<NumberType, dim, n_components>(J_plus, J_minus);
        }

        template <typename NumberType, int dim, size_t n_components>
        static std::pair<std::array<std::array<NumberType, n_components>, dim>,
                         std::array<std::array<NumberType, n_components>, dim>>
        compute_speed_derivatives(
            const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
            const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus,
            const internal::HessianTensor<NumberType, dim, n_components> &H_plus,
            const internal::HessianTensor<NumberType, dim, n_components> &H_minus)
        {
          return MaxEigenvalueWaveSpeed::compute_speed_derivatives<NumberType, dim, n_components>(
              J_plus, J_minus, H_plus, H_minus);
        }

        template <typename NumberType, int dim, size_t n_components>
        static std::pair<std::array<std::array<NumberType, n_components>, dim>,
                         std::array<std::array<NumberType, n_components>, dim>>
        compute_selected_speed_derivatives(
            const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
            const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus,
            const internal::HessianTensor<NumberType, dim, n_components> &H_plus,
            const internal::HessianTensor<NumberType, dim, n_components> &H_minus)
        {
          return MaxEigenvalueWaveSpeed::compute_selected_speed_derivatives<NumberType, dim, n_components>(
              J_plus, J_minus, H_plus, H_minus);
        }
      };

    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
