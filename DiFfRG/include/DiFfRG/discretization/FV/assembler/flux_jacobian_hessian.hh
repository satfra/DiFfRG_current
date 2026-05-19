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

        // Forward declaration; defined below.
        template <typename Model, typename NumberType, int dim, size_t n_components>
        auto compute_flux_jacobian_only(const std::array<NumberType, n_components> &u,
                                        const dealii::Point<dim> &x_q, const Model &model);

        /// LDG-style alternative to compute_flux_jacobian_and_hessian: extracts J via
        /// first-order forward-mode AD and computes H by central finite differences on
        /// J(u ± ε·e_k) — avoiding any Real<2, double> AD seeding through the
        /// (potentially noisy) integrator chain. Returns (F, J, H_fd).
        ///
        /// Cost: 1 (F+J at u) + 2·n_components (F+J at u±ε·e_k for each component) calls
        /// to compute_flux_jacobian_only. For scalar n_components=1 that's 3 evaluations
        /// vs compute_flux_jacobian_and_hessian's 2 — roughly 1.5× the work, but with a
        /// much cleaner H because the FD step uses a derived (rather than seeded)
        /// numerical derivative on the well-conditioned Real<1> J.
        template <typename Model, typename NumberType, int dim, size_t n_components>
        auto compute_flux_jacobian_and_hessian_fd(const std::array<NumberType, n_components> &u,
                                                  const dealii::Point<dim> &x_q, const Model &model)
        {
          // 1. F and J at the centre point via Real<1>.
          auto [F, J, _H_dummy] =
              compute_flux_jacobian_only<Model, NumberType, dim, n_components>(u, x_q, model);

          // 2. Central FD per component to fill H[d][i][j][k] = ∂J[d][i][j]/∂u[k].
          HessianTensor<NumberType, dim, n_components> H{};
          for (size_t k = 0; k < n_components; ++k) {
            // Step size: max(1e-6 · |u_k|, 1e-10). Matches LDG's choice for
            // well-scaled solution components; small enough to capture local curvature,
            // large enough to stay above ~ulp(J) noise after the cancellation in (J+ - J-).
            const NumberType eps =
                std::max(NumberType(1.0e-6) * std::abs(u[k]), NumberType(1.0e-10));

            auto u_plus_eps = u;
            u_plus_eps[k] += eps;
            auto u_minus_eps = u;
            u_minus_eps[k] -= eps;

            auto [F_p, J_p, _Hp] =
                compute_flux_jacobian_only<Model, NumberType, dim, n_components>(u_plus_eps, x_q, model);
            auto [F_m, J_m, _Hm] =
                compute_flux_jacobian_only<Model, NumberType, dim, n_components>(u_minus_eps, x_q, model);

            const NumberType inv_2eps = NumberType(1) / (NumberType(2) * eps);
            for (size_t d = 0; d < dim; ++d)
              for (size_t i = 0; i < n_components; ++i)
                for (size_t j = 0; j < n_components; ++j)
                  H[d][i][j][k] = (J_p[d][i][j] - J_m[d][i][j]) * inv_2eps;
          }

          return std::make_tuple(F, J, H);
        }

        /// Compute only the Jacobian (no Hessian) of the advection flux at a given state u
        /// using first-order forward-mode autodiff. Companion to
        /// compute_flux_jacobian_and_hessian, used when the wave-speed strategy does NOT
        /// need the Hessian (see WaveSpeedStrategy::needs_hessian). Returns a zero-filled
        /// H as the third tuple element so the calling code's structured binding still
        /// works unchanged.
        ///
        /// Rationale: Real<2, double> AD propagates (val, ε, ε²) triples through every
        /// operation. The ε² bookkeeping is sensitive to roundoff in operations like
        /// 1/sqrt(...) — and that noise contaminates the ε coefficient too. The J
        /// extracted from a Real<2> chain is therefore measurably noisier than the J
        /// from a Real<1> chain. When the strategy does not need H, switching to this
        /// Real<1>-only path both halves the AD work and produces a noticeably cleaner J.
        template <typename Model, typename NumberType, int dim, size_t n_components>
        auto compute_flux_jacobian_only(const std::array<NumberType, n_components> &u,
                                        const dealii::Point<dim> &x_q, const Model &model)
        {
          using ADNumberType = autodiff::Real<1, NumberType>;
          using autodiff::detail::seed;

          auto unseed = [](ADNumberType &x) { seed<1>(x, NumberType(0)); };

          std::array<ADNumberType, n_components> u_AD{};
          for (size_t i = 0; i < n_components; ++i)
            u_AD[i] = ADNumberType(u[i]);

          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F{};
          std::array<JacobianMatrix<NumberType, n_components>, dim> J{};
          HessianTensor<NumberType, dim, n_components> H{};  // zero — strategy does not use it

          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> F_AD{};

          // Pass 0: extract F values with clean (unseeded) AD variables.
          model.KurganovTadmor_advection_flux(F_AD, x_q, advection_flux_tie(u_AD));
          for (size_t i = 0; i < n_components; ++i)
            for (size_t d = 0; d < dim; ++d)
              F[i][d] = autodiff::detail::val(F_AD[i][d]);

          // Pass 1: extract Jacobian columns one variable at a time.
          for (size_t j = 0; j < n_components; ++j) {
            seed<1>(u_AD[j], NumberType(1));

            model.KurganovTadmor_advection_flux(F_AD, x_q, advection_flux_tie(u_AD));

            for (size_t d = 0; d < dim; ++d)
              for (size_t i = 0; i < n_components; ++i)
                J[d][i][j] = autodiff::derivative<1>(F_AD[i][d]);

            unseed(u_AD[j]);
          }

          return std::make_tuple(F, J, H);
        }

      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
