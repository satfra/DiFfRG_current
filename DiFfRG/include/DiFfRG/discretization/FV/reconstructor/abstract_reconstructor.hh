#pragma once

// deal.II
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// standard library
#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Per-component gradient type: one Tensor<1,dim> per solution component.
     */
    template <int dim, typename NumberType, size_t n_components>
    using GradientType = std::array<dealii::Tensor<1, dim, NumberType>, n_components>;
    /**
     * @brief Concept that any gradient-reconstruction strategy must satisfy.
     */
    template <typename T>
    concept HasReconstructor =
        // ---- compute_gradient (dim=1) ----
        requires(const dealii::Point<1> &center, const std::array<double, 1> &u_center,
                 const std::array<dealii::Point<1>, 2> &x_n, const std::array<std::array<double, 1>, 2> &u_n) {
          {
            T::template compute_gradient<1, 1>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<1, double, 1>>;
        } &&
        // ---- compute_gradient (dim=2) ----
        requires(const dealii::Point<2> &center, const std::array<double, 2> &u_center,
                 const std::array<dealii::Point<2>, 4> &x_n, const std::array<std::array<double, 2>, 4> &u_n) {
          {
            T::template compute_gradient<2, 2>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<2, double, 2>>;
        } &&
        // ---- compute_gradient_derivative (dim=1) ----
        requires(const dealii::Point<1> &center, const std::array<std::pair<double, bool>, 1> &u_center,
                 const std::array<dealii::Point<1>, 2> &x_n,
                 const std::array<std::array<std::pair<double, bool>, 1>, 2> &u_n) {
          {
            T::template compute_gradient_derivative<1, 1>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<1, double, 1>>;
        } &&
        // ---- compute_gradient_derivative (dim=2) ----
        requires(const dealii::Point<2> &center, const std::array<std::pair<double, bool>, 2> &u_center,
                 const std::array<dealii::Point<2>, 4> &x_n,
                 const std::array<std::array<std::pair<double, bool>, 2>, 4> &u_n) {
          {
            T::template compute_gradient_derivative<2, 2>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<2, double, 2>>;
        };

  } // namespace def
} // namespace DiFfRG
