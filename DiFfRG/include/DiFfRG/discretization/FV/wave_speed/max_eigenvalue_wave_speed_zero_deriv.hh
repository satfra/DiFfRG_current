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
       * @brief Wave-speed strategy with zero derivative (ignores da/du terms in the Jacobian).
       *
       * compute_speeds: identical to MaxEigenvalueWaveSpeed.
       * compute_speed_derivatives: returns da = 0.
       * compute_selected_speed_derivatives: returns da = 0.
       */
      struct MaxEigenvalueWaveSpeedZeroDeriv {

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
            [[maybe_unused]] const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
            [[maybe_unused]] const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus,
            [[maybe_unused]] const internal::HessianTensor<NumberType, dim, n_components> &H_plus,
            [[maybe_unused]] const internal::HessianTensor<NumberType, dim, n_components> &H_minus)
        {
          return {};
        }

        template <typename NumberType, int dim, size_t n_components>
        static std::pair<std::array<std::array<NumberType, n_components>, dim>,
                         std::array<std::array<NumberType, n_components>, dim>>
        compute_selected_speed_derivatives(
            [[maybe_unused]] const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
            [[maybe_unused]] const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus,
            [[maybe_unused]] const internal::HessianTensor<NumberType, dim, n_components> &H_plus,
            [[maybe_unused]] const internal::HessianTensor<NumberType, dim, n_components> &H_minus)
        {
          return {};
        }
      };

    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
