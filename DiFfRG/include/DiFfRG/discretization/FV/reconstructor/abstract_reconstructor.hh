#pragma once

// deal.II
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// standard library
#include <array>
#include <concepts>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Concept that any gradient-reconstruction strategy must satisfy.
     *
     * A reconstructor class must provide a static @c compute_gradient method
     * that, given a cell centre, its solution value, and the neighbouring cell
     * centres / values, returns a per-component gradient vector.
     *
     * Strategies that need a slope limiter (e.g. TVD reconstructors) inject it
     * as a template parameter of the concrete class — the concept itself is
     * limiter-agnostic:
     * @code
     * class MyReconstructor
     * {
     * public:
     *   template <typename NumberType, int dim, int n_components>
     *   static std::array<dealii::Tensor<1, dim, NumberType>, n_components>
     *   compute_gradient(const dealii::Point<dim> &center_pos,
     *                    const std::array<NumberType, n_components> &u_center,
     *                    const std::array<dealii::Point<dim>, 2 * dim> &x_n,
     *                    const std::array<std::array<NumberType, n_components>, 2 * dim> &u_n);
     * };
     * static_assert(HasReconstructor<MyReconstructor>);
     * @endcode
     */
    template <typename T>
    concept HasReconstructor =
        requires(const dealii::Point<1> &center, const std::array<double, 1> &u_center,
                 const std::array<dealii::Point<1>, 2> &x_n, const std::array<std::array<double, 1>, 2> &u_n) {
          {
            T::template compute_gradient<double, 1, 1>(center, u_center, x_n, u_n)
          } -> std::same_as<std::array<dealii::Tensor<1, 1, double>, 1>>;
        } &&
        requires(const dealii::Point<2> &center, const std::array<double, 2> &u_center,
                 const std::array<dealii::Point<2>, 4> &x_n, const std::array<std::array<double, 2>, 4> &u_n) {
          {
            T::template compute_gradient<double, 2, 2>(center, u_center, x_n, u_n)
          } -> std::same_as<std::array<dealii::Tensor<1, 2, double>, 2>>;
        };

  } // namespace def
} // namespace DiFfRG
