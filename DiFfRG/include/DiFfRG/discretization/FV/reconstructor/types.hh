#pragma once

#include <array>
#include <cstddef>
#include <deal.II/base/tensor.h>

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

    template <int dim, typename NumberType, size_t n_components> struct ReconstructionDerivativeData {
      std::array<NumberType, n_components> u{};
      GradientType<dim, NumberType, n_components> grad{};
    };

  } // namespace def
} // namespace DiFfRG
