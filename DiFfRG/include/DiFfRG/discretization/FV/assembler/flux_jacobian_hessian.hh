#pragma once

#include <array>
#include <autodiff/forward/real/real.hpp>
#include <cstddef>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <tuple>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        template <typename... T> auto advection_flux_tie(T &&...t);

        template <typename NumberType, size_t n_components>
        using JacobianMatrix = std::array<std::array<NumberType, n_components>, n_components>;

        template <typename NumberType, int dim, size_t n_components>
        using HessianTensor =
            std::array<std::array<std::array<std::array<NumberType, n_components>, n_components>, n_components>, dim>;

        /// Compute the Jacobian and full Hessian of the advection flux at a given state u
        /// using second-order forward-mode autodiff with the polarization identity for off-diagonal entries.
        template <typename Model, typename NumberType, int dim, size_t n_components>
        auto compute_flux_jacobian_and_hessian(const std::array<NumberType, n_components> &u,
                                               const dealii::Point<dim> &x_q, const Model &model)
        {
          using ADNumberType = autodiff::Real<2, NumberType>;
          using autodiff::detail::seed;
          using autodiff::detail::derivative;

          auto unseed = [](ADNumberType &x) { seed<1>(x, NumberType(0)); };

          std::array<ADNumberType, n_components> u_AD{};
          for (size_t i = 0; i < n_components; ++i)
            u_AD[i] = ADNumberType(u[i]);

          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F{};
          std::array<JacobianMatrix<NumberType, n_components>, dim> J{};
          HessianTensor<NumberType, dim, n_components> H{};

          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> F_AD{};

          // Pass 0: Evaluate the flux with clean (unseeded) AD variables to get the plain flux values.
          model.KurganovTadmor_advection_flux(F_AD, x_q, advection_flux_tie(u_AD));
          for (size_t i = 0; i < n_components; ++i)
            for (size_t d = 0; d < dim; ++d)
              F[i][d] = autodiff::detail::val(F_AD[i][d]);

          // Pass 1: Compute Jacobian columns and diagonal Hessian entries H[j][j].
          // Seed one variable at a time.
          for (size_t j = 0; j < n_components; ++j) {
            seed<1>(u_AD[j], NumberType(1));

            model.KurganovTadmor_advection_flux(F_AD, x_q, advection_flux_tie(u_AD));

            for (size_t d = 0; d < dim; ++d)
              for (size_t i = 0; i < n_components; ++i) {
                J[d][i][j] = derivative<1>(F_AD[i][d]);
                H[d][i][j][j] = derivative<2>(F_AD[i][d]);
              }

            unseed(u_AD[j]);
          }

          // Pass 2: Compute off-diagonal Hessian entries (c > j) via the polarization identity:
          //   H[j][c] = (d²F/(e_j+e_c)² - H[j][j] - H[c][c]) / 2,  H[c][j] = H[j][c]
          // All diagonal entries H[j][j] and H[c][c] are available from pass 1.
          for (size_t j = 0; j < n_components; ++j) {
            for (size_t c = j + 1; c < n_components; ++c) {
              seed<1>(u_AD[j], NumberType(1));
              seed<1>(u_AD[c], NumberType(1));

              model.KurganovTadmor_advection_flux(F_AD, x_q, advection_flux_tie(u_AD));

              for (size_t d = 0; d < dim; ++d)
                for (size_t i = 0; i < n_components; ++i) {
                  const NumberType cross =
                      (derivative<2>(F_AD[i][d]) - H[d][i][j][j] - H[d][i][c][c]) / NumberType(2);
                  H[d][i][j][c] = H[d][i][c][j] = cross;
                }

              unseed(u_AD[j]);
              unseed(u_AD[c]);
            }
          }

          return std::make_tuple(F, J, H);
        }

      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
