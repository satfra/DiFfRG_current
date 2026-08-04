#pragma once

#include <array>
#include <autodiff/forward/real/real.hpp>
#include <cstddef>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <tuple>

#include <DiFfRG/discretization/FV/assembler/flux_ties.hh>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        template <typename NumberType, size_t n_components>
        using JacobianMatrix = std::array<std::array<NumberType, n_components>, n_components>;

        template <typename NumberType, int dim, size_t n_components>
        using HessianTensor =
            std::array<std::array<std::array<std::array<NumberType, n_components>, n_components>, n_components>, dim>;

        /**
         * @brief Derivative of every flux component/direction with respect to every component/direction of grad(u).
         *
         * Indexed as [flux_component][gradient_component][flux_direction][gradient_direction].
         */
        template <typename NumberType, int dim, size_t n_components>
        using FluxGradientJacobian =
            std::array<std::array<dealii::Tensor<2, dim, NumberType>, n_components>, n_components>;

        template <typename NumberType, int dim, size_t n_components>
        using MixedHessianTensor = std::array<HessianTensor<NumberType, dim, n_components>, dim>;

        template <typename NumberType, int dim, size_t n_components> struct FluxDerivativeData {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F{};
          std::array<JacobianMatrix<NumberType, n_components>, dim> J{};
          HessianTensor<NumberType, dim, n_components> H{};
          FluxGradientJacobian<NumberType, dim, n_components> grad_J{};
          MixedHessianTensor<NumberType, dim, n_components> mixed_H{};
        };

        /**
         * @brief Compute F, dF/du, d2F/du2, dF/dgrad(u), and d2F/(du dgrad(u)) with second-order forward AD.
         *
         * The gradient is held fixed while extracting dF/du. The Hessian and mixed derivatives are used by the
         * wave-speed strategy to differentiate the physical KT speed.
         */
        template <typename Model, typename NumberType, int dim, size_t n_components>
        FluxDerivativeData<NumberType, dim, n_components>
        compute_flux_derivatives_ad(const std::array<NumberType, n_components> &u,
                                    const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &grad_u,
                                    const dealii::Point<dim> &x_q, const Model &model)
        {
          using ADNumberType = autodiff::Real<2, NumberType>;
          using autodiff::detail::derivative;
          using autodiff::detail::seed;

          auto unseed = [](ADNumberType &x) { seed<1>(x, NumberType(0)); };

          std::array<ADNumberType, n_components> u_AD{};
          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> grad_u_AD{};
          for (size_t c = 0; c < n_components; ++c) {
            u_AD[c] = ADNumberType(u[c]);
            for (size_t d = 0; d < dim; ++d)
              grad_u_AD[c][d] = ADNumberType(grad_u[c][d]);
          }

          FluxDerivativeData<NumberType, dim, n_components> result{};
          FluxGradientJacobian<NumberType, dim, n_components> grad_diagonal_H{};
          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> F_AD{};

          model.KurganovTadmor_advection_flux(F_AD, x_q, flux_tie(u_AD, grad_u_AD));
          for (size_t i = 0; i < n_components; ++i)
            for (size_t d_out = 0; d_out < dim; ++d_out)
              result.F[i][d_out] = F_AD[i][d_out].val();

          // Diagonal u passes provide J and the diagonal u-Hessian.
          for (size_t j = 0; j < n_components; ++j) {
            seed<1>(u_AD[j], NumberType(1));
            F_AD = {};
            model.KurganovTadmor_advection_flux(F_AD, x_q, flux_tie(u_AD, grad_u_AD));
            for (size_t i = 0; i < n_components; ++i)
              for (size_t d_out = 0; d_out < dim; ++d_out) {
                result.J[d_out][i][j] = derivative<1>(F_AD[i][d_out]);
                result.H[d_out][i][j][j] = derivative<2>(F_AD[i][d_out]);
              }
            unseed(u_AD[j]);
          }

          // Off-diagonal u-Hessian entries via polarization.
          for (size_t j = 0; j < n_components; ++j)
            for (size_t c = j + 1; c < n_components; ++c) {
              seed<1>(u_AD[j], NumberType(1));
              seed<1>(u_AD[c], NumberType(1));
              F_AD = {};
              model.KurganovTadmor_advection_flux(F_AD, x_q, flux_tie(u_AD, grad_u_AD));
              for (size_t i = 0; i < n_components; ++i)
                for (size_t d_out = 0; d_out < dim; ++d_out) {
                  const NumberType cross =
                      (derivative<2>(F_AD[i][d_out]) - result.H[d_out][i][j][j] -
                       result.H[d_out][i][c][c]) /
                      NumberType(2);
                  result.H[d_out][i][j][c] = result.H[d_out][i][c][j] = cross;
                }
              unseed(u_AD[j]);
              unseed(u_AD[c]);
            }

          // Gradient diagonal passes provide dF/dgrad(u) and the diagonal terms used by mixed polarization.
          for (size_t c = 0; c < n_components; ++c)
            for (size_t d_in = 0; d_in < dim; ++d_in) {
              seed<1>(grad_u_AD[c][d_in], NumberType(1));
              F_AD = {};
              model.KurganovTadmor_advection_flux(F_AD, x_q, flux_tie(u_AD, grad_u_AD));
              for (size_t i = 0; i < n_components; ++i)
                for (size_t d_out = 0; d_out < dim; ++d_out) {
                  result.grad_J[i][c][d_out][d_in] = derivative<1>(F_AD[i][d_out]);
                  grad_diagonal_H[i][c][d_out][d_in] = derivative<2>(F_AD[i][d_out]);
                }
              unseed(grad_u_AD[c][d_in]);
            }

          // Mixed d2F/(du_j dgrad(u_c)_d_in) entries via polarization.
          for (size_t j = 0; j < n_components; ++j)
            for (size_t c = 0; c < n_components; ++c)
              for (size_t d_in = 0; d_in < dim; ++d_in) {
                seed<1>(u_AD[j], NumberType(1));
                seed<1>(grad_u_AD[c][d_in], NumberType(1));
                F_AD = {};
                model.KurganovTadmor_advection_flux(F_AD, x_q, flux_tie(u_AD, grad_u_AD));
                for (size_t i = 0; i < n_components; ++i)
                  for (size_t d_out = 0; d_out < dim; ++d_out)
                    result.mixed_H[d_in][d_out][i][j][c] =
                        (derivative<2>(F_AD[i][d_out]) - result.H[d_out][i][j][j] -
                         grad_diagonal_H[i][c][d_out][d_in]) /
                        NumberType(2);
                unseed(u_AD[j]);
                unseed(grad_u_AD[c][d_in]);
              }

          return result;
        }

        /**
         * @brief Backward-compatible state-only view of the full AD derivative extraction.
         *
         * This is intentionally not a second derivative strategy: it calls
         * compute_flux_derivatives_ad with a zero gradient and returns its F/J/H subset.
         */
        template <typename Model, typename NumberType, int dim, size_t n_components>
        auto compute_flux_jacobian_and_hessian(const std::array<NumberType, n_components> &u,
                                               const dealii::Point<dim> &x_q, const Model &model)
        {
          const std::array<dealii::Tensor<1, dim, NumberType>, n_components> grad_u{};
          const auto derivatives =
              compute_flux_derivatives_ad<Model, NumberType, dim, n_components>(u, grad_u, x_q, model);
          return std::make_tuple(derivatives.F, derivatives.J, derivatives.H);
        }

      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
