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

    /**
     * @brief Default FV boundary strategy used by the Kurganov-Tadmor assembler.
     *
     * The ghost states are left untouched and the ghost gradient is copied from the
     * interior cell, reproducing the behavior that used to live in AbstractModel.
     */
    template <typename Model> class FVDefaultBoundaries
    {
    public:
      template <int dim, typename NumberType, size_t n_components, size_t n_faces>
      void
      apply_boundary_conditions([[maybe_unused]] std::array<std::array<NumberType, n_components>, n_faces> &u_neighbors,
                                [[maybe_unused]] std::array<Point<dim>, n_faces> &x_neighbors,
                                [[maybe_unused]] const std::array<types::boundary_id, n_faces> &boundary_ids,
                                [[maybe_unused]] const std::array<Point<dim>, n_faces> &face_centers,
                                [[maybe_unused]] const std::array<NumberType, n_components> &u_cell,
                                [[maybe_unused]] const Point<dim> &x_cell) const
      {
      }

      template <int dim, typename NumberType, size_t n_components, size_t n_faces>
      void
      boundary_ghost_gradient(std::array<Tensor<1, dim, NumberType>, n_components> &ghost_gradient,
                              [[maybe_unused]] const types::boundary_id boundary_id,
                              [[maybe_unused]] const Tensor<1, dim> &normal, [[maybe_unused]] const Point<dim> &x_face,
                              [[maybe_unused]] const std::array<NumberType, n_components> &u_ghost,
                              [[maybe_unused]] const Point<dim> &x_ghost,
                              [[maybe_unused]] const std::array<NumberType, n_components> &u_cell,
                              [[maybe_unused]] const Point<dim> &x_cell,
                              const std::array<Tensor<1, dim, NumberType>, n_components> &cell_gradient,
                              [[maybe_unused]] const std::array<types::boundary_id, n_faces> &boundary_ids,
                              [[maybe_unused]] const std::array<std::array<Tensor<1, dim, NumberType>, n_components>,
                                                                n_faces> &neighboring_gradients) const
      {
        ghost_gradient = cell_gradient;
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
    public:
      template <int dim, typename NumberType, size_t n_components, size_t n_faces>
      void apply_boundary_conditions(std::array<std::array<NumberType, n_components>, n_faces> &u_neighbors,
                                     std::array<Point<dim>, n_faces> &x_neighbors,
                                     const std::array<types::boundary_id, n_faces> &boundary_ids,
                                     const std::array<Point<dim>, n_faces> &face_centers,
                                     const std::array<NumberType, n_components> &u_cell, const Point<dim> &x_cell) const
      {
        static_assert(dim == 1,
                      "OriginOddLinearExtrapolationBoundaries currently supports only one-dimensional FV domains.");
        static_assert(n_faces == GeometryInfo<dim>::faces_per_cell,
                      "OriginOddLinearExtrapolationBoundaries expects one entry per cell face.");

        for (size_t f = 0; f < n_faces; ++f) {
          if (boundary_ids[f] == numbers::invalid_boundary_id) continue;

          const auto opposite_face = GeometryInfo<dim>::opposite_face[f];
          const double x_face = face_centers[f][0];

          if (x_face <= x_cell[0]) {
            if (boundary_ids[opposite_face] == numbers::invalid_boundary_id) {
              x_neighbors[f][0] = -x_neighbors[opposite_face][0];
              for (size_t c = 0; c < n_components; ++c)
                u_neighbors[f][c] = -u_neighbors[opposite_face][c];
            }
            continue;
          }

          x_neighbors[f][0] = 2.0 * x_face - x_cell[0];
          if (boundary_ids[opposite_face] == numbers::invalid_boundary_id) {
            for (size_t c = 0; c < n_components; ++c)
              u_neighbors[f][c] = NumberType(2.0) * u_cell[c] - u_neighbors[opposite_face][c];
          }
        }
      }

      template <int dim, typename NumberType, size_t n_components, size_t n_faces>
      void boundary_ghost_gradient(
          std::array<Tensor<1, dim, NumberType>, n_components> &ghost_gradient,
          [[maybe_unused]] const types::boundary_id boundary_id, [[maybe_unused]] const Tensor<1, dim> &normal,
          [[maybe_unused]] const Point<dim> &x_face,
          [[maybe_unused]] const std::array<NumberType, n_components> &u_ghost,
          [[maybe_unused]] const Point<dim> &x_ghost,
          [[maybe_unused]] const std::array<NumberType, n_components> &u_cell,
          [[maybe_unused]] const Point<dim> &x_cell,
          [[maybe_unused]] const std::array<Tensor<1, dim, NumberType>, n_components> &cell_gradient,
          const std::array<types::boundary_id, n_faces> &boundary_ids,
          const std::array<std::array<Tensor<1, dim, NumberType>, n_components>, n_faces> &neighboring_gradients) const
      {
        static_assert(dim == 1,
                      "OriginOddLinearExtrapolationBoundaries currently supports only one-dimensional FV domains.");
        static_assert(n_faces == GeometryInfo<dim>::faces_per_cell,
                      "OriginOddLinearExtrapolationBoundaries expects one entry per cell face.");

        for (size_t f = 0; f < n_faces; ++f) {
          if (boundary_ids[f] != numbers::invalid_boundary_id) continue;

          ghost_gradient = neighboring_gradients[f];
          return;
        }
      }
    };
  } // namespace def
} // namespace DiFfRG
