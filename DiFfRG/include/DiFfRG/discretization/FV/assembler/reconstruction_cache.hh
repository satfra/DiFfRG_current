#pragma once

#include "DiFfRG/common/math.hh"

#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/discretization/FV/reconstructor/abstract_reconstructor.hh>

#include <array>
#include <autodiff/forward/real/real.hpp>
#include <cstddef>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        template <int dim, typename NumberType, size_t n_components>
        using GradientType = def::GradientType<dim, NumberType, n_components>;

        template <int dim> struct BoundaryStencilIndex {
          static_assert(dim == 1, "Paper-style boundary stencil indices currently support only dim=1.");
        };

        template <int dim>
          requires(dim == 1)
        struct BoundaryStencilIndex<dim> {
          static constexpr size_t lower_outer = 0;
          static constexpr size_t lower_inner = 1;
          static constexpr size_t physical_cell = 2;
          static constexpr size_t upper_inner = 3;
          static constexpr size_t upper_outer = 4;
        };

        template <int dim, typename NumberType, size_t n_components> struct ReconstructionDerivativeData {
          std::array<NumberType, n_components> u{};
          GradientType<dim, NumberType, n_components> grad{};
        };

        template <int dim, typename NumberType, size_t n_components> struct FaceReconstructionState {
          std::array<NumberType, n_components> u_minus{};
          std::array<NumberType, n_components> u_plus{};
          GradientType<dim, NumberType, n_components> center_grad_minus{};
          GradientType<dim, NumberType, n_components> center_grad_plus{};
          GradientType<dim, NumberType, n_components> face_grad_minus{};
          GradientType<dim, NumberType, n_components> face_grad_plus{};
        };

        template <int dim, typename NumberType, size_t n_components>
        FaceReconstructionState<dim, NumberType, n_components>
        reverse_face_reconstruction(const FaceReconstructionState<dim, NumberType, n_components> &state)
        {
          FaceReconstructionState<dim, NumberType, n_components> reversed{};
          reversed.u_minus = state.u_plus;
          reversed.u_plus = state.u_minus;
          reversed.center_grad_minus = state.center_grad_plus;
          reversed.center_grad_plus = state.center_grad_minus;
          reversed.face_grad_minus = state.face_grad_plus;
          reversed.face_grad_plus = state.face_grad_minus;
          return reversed;
        }

        template <int dim, typename NumberType, size_t n_components> struct CellData {
          dealii::Point<dim> x;
          std::array<NumberType, n_components> u;
          std::array<dealii::types::global_dof_index, n_components> dof_indices;
        };

        template <int dim, size_t n_components> struct CellTopologyData {
          dealii::Point<dim> x;
          std::array<dealii::types::global_dof_index, n_components> dof_indices;
        };

        template <int dim, typename NumberType, size_t n_components> struct NeighborData {
          static constexpr size_t n_faces = 2 * dim;
          std::array<dealii::Point<dim>, n_faces> x;
          std::array<std::array<NumberType, n_components>, n_faces> u;
          std::array<std::array<dealii::types::global_dof_index, n_components>, n_faces> dof_indices;
        };

        template <int dim, size_t n_components> struct NeighborTopologyData {
          static constexpr size_t n_faces = 2 * dim;
          std::array<dealii::Point<dim>, n_faces> x;
          std::array<std::array<dealii::types::global_dof_index, n_components>, n_faces> dof_indices;
        };

        template <int dim, typename NumberType, size_t n_components> struct CellStencilData {
          static constexpr size_t n_faces = 2 * dim;
          CellData<dim, NumberType, n_components> cell;
          NeighborData<dim, NumberType, n_components> neighbors;
          std::array<dealii::types::boundary_id, n_faces> boundary_ids;
          std::array<dealii::Point<dim>, n_faces> face_centers;
        };

        template <int dim, typename NumberType, size_t n_components> struct SolutionReconstructionCache {
          static constexpr size_t n_faces = 2 * dim;
          std::vector<CellStencilData<dim, NumberType, n_components>> cell_stencils;
          std::vector<std::array<FaceReconstructionState<dim, NumberType, n_components>, n_faces>>
              face_reconstructions;
          std::vector<std::array<bool, n_faces>> face_reconstruction_valid;
        };

        template <int dim, size_t n_components> struct CellStencilTopologyData {
          static constexpr size_t n_faces = 2 * dim;
          CellTopologyData<dim, n_components> cell;
          NeighborTopologyData<dim, n_components> neighbors;
          std::array<dealii::types::boundary_id, n_faces> boundary_ids;
          std::array<dealii::Point<dim>, n_faces> face_centers;
        };

        template <int dim, typename NumberType, size_t n_components>
        std::array<NumberType, n_components> reconstruct_u(const std::array<NumberType, n_components> &u_center,
                                                           const dealii::Point<dim> &center,
                                                           const dealii::Point<dim> &x,
                                                           const GradientType<dim, NumberType, n_components> &u_grad)
        {
          AssertDimension(u_center.size(), n_components);
          std::array<NumberType, n_components> result;
          for (size_t c = 0; c < n_components; ++c) {
            result[c] = u_center[c];
            result[c] += dealii::scalar_product(u_grad[c], x - center);
          }
          return result;
        }

        template <int dim, typename NumberType, size_t n_components, typename VectorType>
        void fill_cell_data_from_topology(const CellTopologyData<dim, n_components> &topology,
                                          const VectorType &solution_global,
                                          CellData<dim, NumberType, n_components> &data)
        {
          data.x = topology.x;
          data.dof_indices = topology.dof_indices;
          for (unsigned int i = 0; i < n_components; ++i)
            data.u[i] = solution_global(topology.dof_indices[i]);
        }

        template <def::HasReconstructor Reconstructor, int dim, typename NumberType, size_t n_components>
        FaceReconstructionState<dim, NumberType, n_components> compute_interior_face_reconstruction_state(
            const CellStencilData<dim, NumberType, n_components> &minus_stencil,
            const CellStencilData<dim, NumberType, n_components> &plus_stencil, const dealii::Point<dim> &x_q)
        {
          FaceReconstructionState<dim, NumberType, n_components> state{};
          state.center_grad_minus = Reconstructor::template compute_gradient<n_components>(
              minus_stencil.cell.x, minus_stencil.cell.u, minus_stencil.neighbors.x, minus_stencil.neighbors.u);
          state.center_grad_plus = Reconstructor::template compute_gradient<n_components>(
              plus_stencil.cell.x, plus_stencil.cell.u, plus_stencil.neighbors.x, plus_stencil.neighbors.u);
          state.face_grad_minus = Reconstructor::template compute_gradient_at_point<n_components>(
              minus_stencil.cell.x, x_q, minus_stencil.cell.u, minus_stencil.neighbors.x, minus_stencil.neighbors.u);
          state.face_grad_plus = Reconstructor::template compute_gradient_at_point<n_components>(
              plus_stencil.cell.x, x_q, plus_stencil.cell.u, plus_stencil.neighbors.x, plus_stencil.neighbors.u);
          state.u_minus = reconstruct_u(minus_stencil.cell.u, minus_stencil.cell.x, x_q, state.center_grad_minus);
          state.u_plus = reconstruct_u(plus_stencil.cell.u, plus_stencil.cell.x, x_q, state.center_grad_plus);
          return state;
        }

        template <def::HasReconstructor Reconstructor, int dim, typename NumberType, size_t n_components>
        std::array<NumberType, n_components> reconstruct_u_derivative(
            const std::array<autodiff::Real<1, NumberType>, n_components> &u_center,
            const dealii::Point<dim> &center, const dealii::Point<dim> &x,
            const std::array<dealii::Point<dim>, 2 * dim> &x_n,
            const std::array<std::array<autodiff::Real<1, NumberType>, n_components>, 2 * dim> &u_n)
        {
          const auto u_grad_deriv =
              Reconstructor::template compute_gradient_derivative<n_components>(center, u_center, x_n, u_n);
          std::array<NumberType, n_components> result;
          for (size_t c = 0; c < n_components; ++c) {
            result[c] = derivative(u_center[c]);
            result[c] += dealii::scalar_product(u_grad_deriv[c], x - center);
          }
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        bool is_lower_boundary_stencil(const std::array<dealii::Point<dim>, 5> &x_stencil,
                                       const dealii::Point<dim> &x_face)
        {
          static_assert(dim == 1, "Paper-style boundary stencil orientation currently supports only dim=1.");

          return x_face[0] <= x_stencil[BoundaryStencilIndex<dim>::physical_cell][0];
        }

        template <int dim, typename NumberType, size_t n_components> struct BoundaryStencilData {
          static_assert(dim == 1, "Paper-style boundary stencil data currently supports only dim=1.");
        };

        template <int dim, typename NumberType, size_t n_components>
          requires(dim == 1)
        struct BoundaryStencilData<dim, NumberType, n_components> {
          std::array<dealii::Point<dim>, 5> x{};
          std::array<std::array<NumberType, n_components>, 5> u{};
          std::array<std::array<dealii::types::global_dof_index, n_components>, 5> dof_indices{};
          bool lower_boundary = true;
          size_t ghost_center = BoundaryStencilIndex<dim>::lower_inner;
          size_t ghost_left = BoundaryStencilIndex<dim>::lower_outer;
          size_t ghost_right = BoundaryStencilIndex<dim>::physical_cell;
        };

        template <int dim, size_t n_components> struct BoundaryStencilTopologyData {
          static_assert(dim == 1, "Paper-style boundary stencil topology currently supports only dim=1.");
        };

        template <int dim, size_t n_components>
          requires(dim == 1)
        struct BoundaryStencilTopologyData<dim, n_components> {
          std::array<dealii::Point<dim>, 5> x{};
          std::array<std::array<dealii::types::global_dof_index, n_components>, 5> dof_indices{};
          bool lower_boundary = true;
          size_t ghost_center = BoundaryStencilIndex<dim>::lower_inner;
          size_t ghost_left = BoundaryStencilIndex<dim>::lower_outer;
          size_t ghost_right = BoundaryStencilIndex<dim>::physical_cell;
        };

        template <typename BoundaryNumberType, int dim, size_t n_components, typename VectorType>
        BoundaryStencilData<dim, BoundaryNumberType, n_components> fill_boundary_stencil_from_topology(
            const BoundaryStencilTopologyData<dim, n_components> &topology, const VectorType &solution_global)
        {
          static_assert(dim == 1, "Paper-style boundary stencil data currently supports only dim=1.");

          BoundaryStencilData<dim, BoundaryNumberType, n_components> boundary_stencil{};
          boundary_stencil.x = topology.x;
          boundary_stencil.dof_indices = topology.dof_indices;
          boundary_stencil.lower_boundary = topology.lower_boundary;
          boundary_stencil.ghost_center = topology.ghost_center;
          boundary_stencil.ghost_left = topology.ghost_left;
          boundary_stencil.ghost_right = topology.ghost_right;
          for (size_t stencil_index = 0; stencil_index < boundary_stencil.u.size(); ++stencil_index) {
            for (size_t c = 0; c < n_components; ++c) {
              const auto dof = topology.dof_indices[stencil_index][c];
              boundary_stencil.u[stencil_index][c] =
                  dof == dealii::numbers::invalid_dof_index ? BoundaryNumberType{} :
                                                               BoundaryNumberType(solution_global(dof));
            }
          }
          return boundary_stencil;
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_boundary_side_stencil_1d(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                      const size_t center_index, const size_t left_index, const size_t right_index)
        {
          static_assert(dim == 1, "Paper-style boundary side stencils currently support only dim=1.");

          CellStencilData<dim, NumberType, n_components> result{};
          result.boundary_ids.fill(dealii::numbers::invalid_boundary_id);
          result.face_centers = {};

          result.cell.x = boundary_stencil.x[center_index];
          result.cell.u = boundary_stencil.u[center_index];
          result.cell.dof_indices = boundary_stencil.dof_indices[center_index];
          result.neighbors.x[0] = boundary_stencil.x[left_index];
          result.neighbors.x[1] = boundary_stencil.x[right_index];
          result.neighbors.u[0] = boundary_stencil.u[left_index];
          result.neighbors.u[1] = boundary_stencil.u[right_index];
          result.neighbors.dof_indices[0] = boundary_stencil.dof_indices[left_index];
          result.neighbors.dof_indices[1] = boundary_stencil.dof_indices[right_index];
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_physical_boundary_side_stencil_1d(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil)
        {
          static_assert(dim == 1, "Paper-style physical boundary side stencils currently support only dim=1.");

          return make_boundary_side_stencil_1d<dim>(boundary_stencil, BoundaryStencilIndex<dim>::physical_cell,
                                                    BoundaryStencilIndex<dim>::lower_inner,
                                                    BoundaryStencilIndex<dim>::upper_inner);
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_ghost_boundary_side_stencil_1d(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil)
        {
          static_assert(dim == 1, "Paper-style ghost boundary side stencils currently support only dim=1.");

          return make_boundary_side_stencil_1d<dim>(boundary_stencil, boundary_stencil.ghost_center,
                                               boundary_stencil.ghost_left, boundary_stencil.ghost_right);
        }

        template <def::HasReconstructor Reconstructor, int dim, typename Model, typename NumberType, size_t n_components>
        FaceReconstructionState<dim, NumberType, n_components> compute_boundary_face_reconstruction_state(
            BoundaryStencilData<dim, NumberType, n_components> boundary_stencil, const dealii::Point<dim> &x_q,
            const Model &model)
        {
          static_assert(dim == 1, "Paper-style boundary face reconstruction currently supports only dim=1.");

          const bool boundary_supported = model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x, x_q);
          AssertThrow(boundary_supported, dealii::ExcMessage("KT boundary stencil was rejected by the model boundary policy."));

          const auto physical_stencil = make_boundary_side_stencil_1d<dim, NumberType, n_components>(
              boundary_stencil, BoundaryStencilIndex<dim>::physical_cell, BoundaryStencilIndex<dim>::lower_inner,
              BoundaryStencilIndex<dim>::upper_inner);
          const auto ghost_stencil = make_ghost_boundary_side_stencil_1d<dim, NumberType, n_components>(boundary_stencil);
          return compute_interior_face_reconstruction_state<Reconstructor>(physical_stencil, ghost_stencil, x_q);
        }

        template <int dim, typename NumberType, size_t n_components>
        CellData<dim, autodiff::Real<1, NumberType>, n_components>
        tag_cell_dofs(const CellData<dim, NumberType, n_components> &cell_data, dealii::types::global_dof_index dof_j)
        {
          using AD = autodiff::Real<1, NumberType>;
          CellData<dim, AD, n_components> result;
          result.x = cell_data.x;
          result.dof_indices = cell_data.dof_indices;
          for (size_t c = 0; c < n_components; ++c) {
            result.u[c] = AD(cell_data.u[c]);
            if (cell_data.dof_indices[c] == dof_j) seed(result.u[c]);
          }
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        NeighborData<dim, autodiff::Real<1, NumberType>, n_components>
        make_tagged_neighbors(const NeighborData<dim, NumberType, n_components> &neighbor_data,
                              dealii::types::global_dof_index dof_j)
        {
          using AD = autodiff::Real<1, NumberType>;
          NeighborData<dim, AD, n_components> result;
          result.x = neighbor_data.x;
          result.dof_indices = neighbor_data.dof_indices;
          for (size_t face = 0; face < NeighborData<dim, NumberType, n_components>::n_faces; ++face) {
            for (size_t c = 0; c < n_components; ++c) {
              result.u[face][c] = AD(neighbor_data.u[face][c]);
              if (neighbor_data.dof_indices[face][c] == dof_j) seed(result.u[face][c]);
            }
          }
          return result;
        }

        template <typename NumberType, size_t n_components>
        std::array<std::array<autodiff::Real<1, NumberType>, n_components>, 5> make_tagged_physical_boundary_stencil(
            const std::array<std::array<NumberType, n_components>, 5> &u_stencil,
            const std::array<std::array<dealii::types::global_dof_index, n_components>, 5> &dof_stencil,
            dealii::types::global_dof_index dof_j)
        {
          using AD = autodiff::Real<1, NumberType>;
          std::array<std::array<AD, n_components>, 5> result{};
          for (size_t stencil_index = 0; stencil_index < 5; ++stencil_index) {
            for (size_t c = 0; c < n_components; ++c) {
              result[stencil_index][c] = AD(u_stencil[stencil_index][c]);
              if (dof_stencil[stencil_index][c] == dof_j) seed(result[stencil_index][c]);
            }
          }
          return result;
        }

      } // namespace internal
    }   // namespace KurganovTadmor
  }     // namespace FV
} // namespace DiFfRG
