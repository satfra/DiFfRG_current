#pragma once

// deal.II
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// autodiff
#include <autodiff/forward/real/real.hpp>

// standard library
#include <array>
#include <concepts>
#include <cstddef>

namespace DiFfRG
{
  namespace def
  {
    template <int dim> inline constexpr std::size_t n_faces = 2 * dim;

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
        requires {
          {
            T::dim
          } -> std::convertible_to<int>;
        } &&
        // ---- compute_gradient (n_components=1) ----
        requires(const dealii::Point<T::dim> &center, const std::array<double, 1> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<double, 1>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient<1>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 1>>;
        } &&
        // ---- compute_gradient (n_components=2) ----
        requires(const dealii::Point<T::dim> &center, const std::array<double, 2> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<double, 2>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient<2>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 2>>;
        } &&
        // ---- compute_gradient_at_point (n_components=1) ----
        requires(const dealii::Point<T::dim> &center, const dealii::Point<T::dim> &x,
                 const std::array<double, 1> &u_center, const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<double, 1>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_at_point<1>(center, x, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 1>>;
        } &&
        // ---- compute_gradient_at_point (n_components=2) ----
        requires(const dealii::Point<T::dim> &center, const dealii::Point<T::dim> &x,
                 const std::array<double, 2> &u_center, const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<double, 2>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_at_point<2>(center, x, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 2>>;
        } &&
        // ---- compute_gradient_derivative (n_components=1) ----
        requires(const dealii::Point<T::dim> &center, const std::array<autodiff::Real<1, double>, 1> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<autodiff::Real<1, double>, 1>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_derivative<1>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 1>>;
        } &&
        // ---- compute_gradient_derivative (n_components=2) ----
        requires(const dealii::Point<T::dim> &center, const std::array<autodiff::Real<1, double>, 2> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<autodiff::Real<1, double>, 2>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_derivative<2>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 2>>;
        };

  } // namespace def
} // namespace DiFfRG
