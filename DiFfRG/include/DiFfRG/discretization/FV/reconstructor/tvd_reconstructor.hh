#pragma once

// DiFfRG
#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/abstract_reconstructor.hh>

// deal.II
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// autodiff
#include <autodiff/forward/real/real.hpp>

// standard library
#include <array>
#include <cstddef>
#include <utility>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief TVD gradient reconstructor parameterised by a slope limiter.
     *
     * Computes piecewise-linear cell gradients on a rectangular mesh using
     * two one-sided finite-difference slopes per dimension, each pair
     * limited by the injected @c Limiter.
     *
     * When the limiter satisfies the TVD property (e.g. @c MinModLimiter),
     * the resulting reconstruction is Total Variation Diminishing.
     *
     * @tparam Limiter      A type satisfying the @c HasSlopeLimiter concept
     *                      (e.g. @c MinModLimiter).
     * @tparam NumberType   The numeric type used for computations
     *                      (e.g. @c double, @c float).
     *
     * Usage with the KT assembler:
     * @code
     * using Assembler = FV::KurganovTadmor::Assembler<
     *     Discretization, Model, def::TVDReconstructor<def::MinModLimiter, double>>;
     * @endcode
     */
    template <HasSlopeLimiter Limiter, typename NumberType> class TVDReconstructor
    {
      using ADNumberType = autodiff::Real<1, NumberType>;

    public:
      /**
       * @brief Compute the gradient of u using the injected slope limiter.
       *
       * For every component and every spatial dimension the method computes
       * two one-sided slopes from the cell centre to its axis-aligned
       * neighbours and returns the limiter-processed gradient.
       *
       * @tparam dim          the spatial dimension
       * @tparam n_components the number of components in u
       *
       * @param center_pos  position of the cell centre
       * @param u_center    solution values at the cell centre
       * @param x_n         positions of the 2*dim neighbouring cell centres,
       *                    ordered as pairs (left, right) for dim 0, then dim 1, …
       * @param u_n         solution values at those neighbours
       *
       * @return per-component gradient vector
       */
      template <int dim, int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient(const dealii::Point<dim> &center_pos, const std::array<NumberType, n_components> &u_center,
                       const std::array<dealii::Point<dim>, 2 * dim> &x_n,
                       const std::array<std::array<NumberType, n_components>, 2 * dim> &u_n)
      {
        GradientType<dim, NumberType, n_components> u_grad{};
        for (size_t c = 0; c < static_cast<size_t>(n_components); c++) {
          const NumberType &u_val = u_center[c];
          for (int d = 0, i_n_1 = 0, i_n_2 = 1; d < dim; d++, i_n_1 += 2, i_n_2 += 2) {

            const auto &u_n_1 = u_n[i_n_1][c];
            const auto &u_n_2 = u_n[i_n_2][c];

            const auto dx_1 = x_n[i_n_1] - center_pos;
            const NumberType du_1 = (u_n_1 - u_val) / dx_1[d];
            const auto dx_2 = x_n[i_n_2] - center_pos;
            const NumberType du_2 = (u_n_2 - u_val) / dx_2[d];

            u_grad[c][d] = Limiter::slope_limit(du_1, du_2);
          }
        }

        return u_grad;
      }

      /**
       * @brief Compute the derivative of the limited gradient w.r.t. a single stencil DOF.
       *
       * Uses the same @c std::pair<NumberType,bool> seeding convention as
       * @c compute_reconstruction_derivative.  Exactly one entry across
       * @p u_center and @p u_n must have its flag set to @c true.
       *
       * Unlike @c compute_reconstruction_derivative, this method does not
       * evaluate the reconstruction at a face point — it returns the
       * per-component derivative of the gradient itself.
       *
       * @tparam dim          the spatial dimension
       * @tparam n_components the number of solution components
       *
       * @param center_pos  position of the cell centre
       * @param u_center    cell-centre values as (value, is_derivative_target) pairs
       * @param x_n         positions of the 2*dim neighbouring cell centres
       * @param u_n         neighbour values as (value, is_derivative_target) pairs
       *
       * @return per-component gradient-derivative tensor  d(grad u_c) / d(u_target)
       */
      template <int dim, int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient_derivative(const dealii::Point<dim> &center_pos,
                                  const std::array<std::pair<NumberType, bool>, n_components> &u_center,
                                  const std::array<dealii::Point<dim>, 2 * dim> &x_n,
                                  const std::array<std::array<std::pair<NumberType, bool>, n_components>, 2 * dim> &u_n)
      {
        // Verify that exactly one boolean flag is set across all stencil entries
        {
          size_t n_seeds = 0;
          for (size_t c = 0; c < n_components; ++c)
            if (u_center[c].second) ++n_seeds;
          for (size_t k = 0; k < 2 * dim; ++k)
            for (size_t c = 0; c < n_components; ++c)
              if (u_n[k][c].second) ++n_seeds;
          if (n_seeds != 1)
            throw std::runtime_error("compute_gradient_derivative: exactly one derivative target "
                                     "(bool == true) must be set across u_center and u_n, but " +
                                     std::to_string(n_seeds) + " were found.");
        }

        // Promote stencil values to AD type, seeding the flagged entry
        std::array<ADNumberType, n_components> u_center_AD{};
        for (size_t c = 0; c < n_components; ++c) {
          u_center_AD[c] = ADNumberType(u_center[c].first);
          if (u_center[c].second) seed<1>(u_center_AD[c], NumberType(1));
        }

        std::array<std::array<ADNumberType, n_components>, 2 * dim> u_n_AD{};
        for (size_t k = 0; k < 2 * dim; ++k)
          for (size_t c = 0; c < n_components; ++c) {
            u_n_AD[k][c] = ADNumberType(u_n[k][c].first);
            if (u_n[k][c].second) seed<1>(u_n_AD[k][c], NumberType(1));
          }

        // Evaluate gradient reconstruction with AD types
        const auto u_grad_AD = TVDReconstructor<Limiter, ADNumberType>::template compute_gradient<dim, n_components>(
            center_pos, u_center_AD, x_n, u_n_AD);

        // Extract per-component gradient derivatives
        GradientType<dim, NumberType, n_components> result{};
        for (size_t c = 0; c < n_components; ++c)
          for (int d = 0; d < dim; ++d)
            result[c][d] = autodiff::derivative(u_grad_AD[c][d]);

        return result;
      }
    };

    // Verify the default instantiation satisfies the concept.
    static_assert(HasReconstructor<TVDReconstructor<MinModLimiter, double>>);

  } // namespace def
} // namespace DiFfRG
