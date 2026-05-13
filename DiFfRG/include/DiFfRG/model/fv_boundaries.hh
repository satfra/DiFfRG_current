#pragma once

// standard library
#include <array>

// external libraries
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

namespace DiFfRG
{
  namespace def
  {
    using namespace dealii;
    template <int dim, typename NumberType, std::size_t n_components>
    using BoundaryStencilValues = std::array<std::array<NumberType, n_components>, 2 * dim + 3>;

    template <int dim> using BoundaryStencilPoints = std::array<Point<dim>, 2 * dim + 3>;

    namespace BoundaryStencilIndex
    {
      constexpr size_t lower_outer = 0;
      constexpr size_t lower_inner = 1;
      constexpr size_t physical_cell = 2;
      constexpr size_t upper_inner = 3;
      constexpr size_t upper_outer = 4;
    } // namespace BoundaryStencilIndex

    /**
     * @brief Default FV boundary strategy used by the Kurganov-Tadmor assembler.
     *
     * In the stencil-only KT implementation this means affine extrapolation from the
     * nearest physical cells on either boundary.
     */
    template <typename Model> class FVDefaultBoundaries
    {
    public:
      template <int dim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
                                  BoundaryStencilPoints<dim> &x_stencil, const Point<dim> &x_face) const
      {
        static_assert(dim == 1, "FV KT boundary stencils currently support only one-dimensional FV domains.");

        using namespace BoundaryStencilIndex;
        const bool lower_boundary = x_face[0] <= x_stencil[physical_cell][0];
        const double delta =
            lower_boundary ? (x_stencil[upper_inner][0] - x_stencil[physical_cell][0])
                           : (x_stencil[physical_cell][0] - x_stencil[lower_inner][0]);

        if (lower_boundary) {
          x_stencil[lower_inner][0] = x_stencil[physical_cell][0] - delta;
          x_stencil[lower_outer][0] = x_stencil[physical_cell][0] - 2.0 * delta;
          for (size_t c = 0; c < n_components; ++c) {
            u_stencil[lower_inner][c] = NumberType(2.0) * u_stencil[physical_cell][c] - u_stencil[upper_inner][c];
            u_stencil[lower_outer][c] =
                NumberType(3.0) * u_stencil[physical_cell][c] - NumberType(2.0) * u_stencil[upper_inner][c];
          }
          return true;
        }

        x_stencil[upper_inner][0] = x_stencil[physical_cell][0] + delta;
        x_stencil[upper_outer][0] = x_stencil[physical_cell][0] + 2.0 * delta;
        for (size_t c = 0; c < n_components; ++c) {
          u_stencil[upper_inner][c] = NumberType(2.0) * u_stencil[physical_cell][c] - u_stencil[lower_inner][c];
          u_stencil[upper_outer][c] =
              NumberType(3.0) * u_stencil[physical_cell][c] - NumberType(2.0) * u_stencil[lower_inner][c];
        }
        return true;
      }
    };

    /**
     * @brief FV boundary strategy using odd reflection at the origin and linear extrapolation at the outer boundary.
     *
     * Within the current face-based FV implementation this uses:
     * - odd reflection of the conserved quantity at the lower boundary (sigma = 0),
     * - linear extrapolation at the upper boundary (sigma = sigma_max),
     * - smooth-extension ghost gradients.
     *
     * The strategy currently targets one-dimensional FV domains, i.e. a single field-space direction.
     */
    template <typename Model> class OriginOddLinearExtrapolationBoundaries : public FVDefaultBoundaries<Model>
    {
      static bool is_lower_boundary_1d(const Point<1> &x_face, const BoundaryStencilPoints<1> &x_stencil)
      {
        return x_face[0] <= x_stencil[BoundaryStencilIndex::physical_cell][0];
      }

    public:
      template <int dim, typename NumberType, size_t n_components>
      bool apply_boundary_stencil(BoundaryStencilValues<dim, NumberType, n_components> &u_stencil,
                                  BoundaryStencilPoints<dim> &x_stencil, const Point<dim> &x_face) const
      {
        static_assert(dim == 1,
                      "OriginOddLinearExtrapolationBoundaries currently supports only one-dimensional FV domains.");

        const bool lower_boundary = is_lower_boundary_1d(x_face, x_stencil);
        using namespace BoundaryStencilIndex;
        const double delta = lower_boundary ? (x_stencil[upper_inner][0] - x_stencil[physical_cell][0])
                                            : (x_stencil[physical_cell][0] - x_stencil[lower_inner][0]);

        if (lower_boundary) {
          x_stencil[lower_inner][0] = x_stencil[physical_cell][0] - delta;
          x_stencil[lower_outer][0] = x_stencil[physical_cell][0] - 2.0 * delta;
          for (size_t c = 0; c < n_components; ++c) {
            u_stencil[lower_inner][c] = -u_stencil[upper_inner][c];
            u_stencil[lower_outer][c] = -u_stencil[upper_outer][c];
            u_stencil[physical_cell][c] = NumberType(0.0);
          }
          return true;
        }

        x_stencil[upper_inner][0] = x_stencil[physical_cell][0] + delta;
        x_stencil[upper_outer][0] = x_stencil[physical_cell][0] + 2.0 * delta;
        for (size_t c = 0; c < n_components; ++c) {
          u_stencil[upper_inner][c] = NumberType(2.0) * u_stencil[physical_cell][c] - u_stencil[lower_inner][c];
          u_stencil[upper_outer][c] =
              NumberType(3.0) * u_stencil[physical_cell][c] - NumberType(2.0) * u_stencil[lower_inner][c];
        }
        return true;
      }
    };
  } // namespace def
} // namespace DiFfRG
