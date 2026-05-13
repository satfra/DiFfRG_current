#pragma once

// external libraries

// DiFfRG
#include "DiFfRG/common/math.hh"
#include <array>
#include <autodiff/forward/real/real.hpp>
#include <cstddef>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria_iterator_base.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/assemble_flags.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <iostream>
#include <optional>
#include <spdlog/spdlog.h>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>
#include <DiFfRG/discretization/FV/reconstructor/tvd_reconstructor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>

#include <DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh>
#include <DiFfRG/discretization/FV/assembler/flux_ties.hh>
#include <DiFfRG/discretization/FV/wave_speed/abstract_wave_speed.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed_zero_deriv.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <tuple>
#include <utility>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      using namespace dealii;

      namespace internal
      {
        template <int dim, typename NumberType, size_t n_components>
        using GradientType = def::GradientType<dim, NumberType, n_components>;

        namespace BoundaryStencilIndex
        {
          constexpr size_t lower_outer = 0;
          constexpr size_t lower_inner = 1;
          constexpr size_t physical_cell = 2;
          constexpr size_t upper_inner = 3;
          constexpr size_t upper_outer = 4;
        } // namespace BoundaryStencilIndex

        template <int dim, typename NumberType, size_t n_components> struct ReconstructionDerivativeData {
          std::array<NumberType, n_components> u{};
          GradientType<dim, NumberType, n_components> grad{};
        };

        template <int dim, typename NumberType, size_t n_components> struct CellData {
          dealii::Point<dim> x;
          std::array<NumberType, n_components> u;
          std::array<dealii::types::global_dof_index, n_components> dof_indices;
        };

        template <int dim, typename NumberType, size_t n_components> struct NeighborData {
          static constexpr size_t n_faces = 2 * dim;
          std::array<dealii::Point<dim>, n_faces> x;
          std::array<std::array<NumberType, n_components>, n_faces> u;
          std::array<std::array<dealii::types::global_dof_index, n_components>, n_faces> dof_indices;
        };

        template <int dim, typename NumberType, size_t n_components> struct CellStencilData {
          static constexpr size_t n_faces = 2 * dim;
          CellData<dim, NumberType, n_components> cell;
          NeighborData<dim, NumberType, n_components> neighbors;
          std::array<types::boundary_id, n_faces> boundary_ids;
          std::array<dealii::Point<dim>, n_faces> face_centers;
        };

        /**
         * @brief Class to hold data for each assembly thread, i.e. FEValues for cells, interfaces, as well as
         * pre-allocated data structures for the solutions
         */
        template <int dim, typename NumberType, size_t n_components> struct ScratchData {
          using QuadratureValue = std::array<NumberType, n_components>;

          ScratchData(const Quadrature<dim> &quadrature)
              : cell_dof_indices(n_components),
                ncell_dof_indices(n_components),
                solution_values(quadrature.size()), solution_dot_values(quadrature.size())
          {
          }

          ScratchData(const ScratchData<dim, NumberType, n_components> &scratch_data)
              : cell_dof_indices(n_components),
                ncell_dof_indices(n_components),
                solution_values(scratch_data.solution_values.size()),
                solution_dot_values(scratch_data.solution_dot_values.size())
          {
          }

          std::vector<types::global_dof_index> cell_dof_indices;
          std::vector<types::global_dof_index> ncell_dof_indices;
          std::vector<QuadratureValue> solution_values;
          std::vector<QuadratureValue> solution_dot_values;
          std::array<std::vector<ReconstructionDerivativeData<dim, NumberType, n_components>>, 2>
              reconstructed_derivatives;
          CellStencilData<dim, NumberType, n_components> cell_stencil;
          CellStencilData<dim, NumberType, n_components> ncell_stencil;
          CellStencilData<dim, NumberType, n_components> temporary_stencil;
        };

        template <int dim, typename NumberType, size_t n_components>
        std::array<NumberType, n_components> reconstruct_u(const std::array<NumberType, n_components> &u_center,
                                                           const Point<dim> &center, const Point<dim> &x,
                                                           const GradientType<dim, NumberType, n_components> &u_grad)
        {
          AssertDimension(u_center.size(), n_components);
          std::array<NumberType, n_components> result;
          for (size_t c = 0; c < n_components; ++c) {
            result[c] = u_center[c];
            result[c] += scalar_product(u_grad[c], x - center);
          }
          return result;
        }

        template <typename Reconstructor, int dim, typename NumberType, size_t n_components>
        std::array<NumberType, n_components> reconstruct_u_derivative(
            const std::array<autodiff::Real<1, NumberType>, n_components> &u_center, const Point<dim> &center,
            const Point<dim> &x, const std::array<Point<dim>, 2 * dim> &x_n,
            const std::array<std::array<autodiff::Real<1, NumberType>, n_components>, 2 * dim> &u_n)
        {
          const auto u_grad_deriv =
              Reconstructor::template compute_gradient_derivative<n_components>(center, u_center, x_n, u_n);
          std::array<NumberType, n_components> result;
          for (size_t c = 0; c < n_components; ++c) {
            result[c] = derivative(u_center[c]);
            result[c] += scalar_product(u_grad_deriv[c], x - center);
          }
          return result;
        }

        template <typename NumberType, size_t n_components>
        bool is_lower_boundary_stencil(const std::array<dealii::Point<1>, 5> &x_stencil, const dealii::Point<1> &x_face)
        {
          return x_face[0] <= x_stencil[BoundaryStencilIndex::physical_cell][0];
        }

        template <typename NumberType, size_t n_components> struct BoundaryStencilData1D {
          std::array<dealii::Point<1>, 5> x{};
          std::array<std::array<NumberType, n_components>, 5> u{};
          std::array<std::array<dealii::types::global_dof_index, n_components>, 5> dof_indices{};
          bool lower_boundary = true;
          size_t ghost_center = BoundaryStencilIndex::lower_inner;
          size_t ghost_left = BoundaryStencilIndex::lower_outer;
          size_t ghost_right = BoundaryStencilIndex::physical_cell;
        };

        template <typename NumberType, size_t n_components>
        CellStencilData<1, NumberType, n_components>
        make_boundary_side_stencil_1d(const BoundaryStencilData1D<NumberType, n_components> &boundary_stencil,
                                      const size_t center_index, const size_t left_index, const size_t right_index)
        {
          CellStencilData<1, NumberType, n_components> result{};
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

        template <typename NumberType, size_t n_components>
        CellStencilData<1, NumberType, n_components>
        make_physical_boundary_side_stencil_1d(const BoundaryStencilData1D<NumberType, n_components> &boundary_stencil)
        {
          using namespace BoundaryStencilIndex;
          return make_boundary_side_stencil_1d(boundary_stencil, physical_cell, lower_inner, upper_inner);
        }

        template <typename NumberType, size_t n_components>
        CellStencilData<1, NumberType, n_components>
        make_ghost_boundary_side_stencil_1d(const BoundaryStencilData1D<NumberType, n_components> &boundary_stencil)
        {
          return make_boundary_side_stencil_1d(boundary_stencil, boundary_stencil.ghost_center,
                                               boundary_stencil.ghost_left, boundary_stencil.ghost_right);
        }

        // TODO fewer memory allocations
        template <typename NumberType> struct CopyData_R {
          struct CopyDataFace_R {
            Vector<NumberType> cell_residual;
            std::vector<types::global_dof_index> joint_dof_indices;

            void reinit(const unsigned int n_face_dofs)
            {
              cell_residual.reinit(n_face_dofs);
              joint_dof_indices.resize(n_face_dofs);
            }
          };

          Vector<NumberType> cell_residual;
          Vector<NumberType> cell_mass;
          std::vector<types::global_dof_index> local_dof_indices;
          std::vector<CopyDataFace_R> face_data;
          unsigned int active_face_count = 0;

          template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell)
          {
            cell_residual.reinit(dofs_per_cell);
            cell_mass.reinit(dofs_per_cell);
            local_dof_indices.resize(dofs_per_cell);
            if (face_data.size() != cell->n_faces()) face_data.resize(cell->n_faces());
            active_face_count = 0;
            cell->get_dof_indices(local_dof_indices);
          }

          CopyDataFace_R &next_face_data()
          {
            AssertIndexRange(active_face_count, face_data.size());
            return face_data[active_face_count++];
          }
        };

        // TODO fewer memory allocations
        template <typename NumberType, int dim> struct CopyData_J {
          using Iterator = typename DoFHandler<dim>::active_cell_iterator;

          struct CopyDataFace_J {
            // since face dofs do not only depend on themself but also on their neighbors (because of the
            // reconstruction) we need to consider a general rectengular local jacobi matrix
            FullMatrix<NumberType> cell_jacobian;
            FullMatrix<NumberType> extractor_cell_jacobian;
            std::vector<types::global_dof_index> to_dofs;
            std::vector<types::global_dof_index> from_dofs;
            std::vector<types::global_dof_index> neighbor_dof_indices;

            static void append_neighbor_dofs(std::vector<types::global_dof_index> &from_dofs,
                                             std::vector<types::global_dof_index> &neighbor_dof_indices,
                                             const Iterator &root_cell)
            {
              if (neighbor_dof_indices.size() != root_cell->get_fe().n_dofs_per_cell())
                neighbor_dof_indices.resize(root_cell->get_fe().n_dofs_per_cell());

              const auto append_recursive = [&](const auto &self, const Iterator &cell, const unsigned int depth) -> void {
                for (const auto face_index : cell->face_indices()) {
                  if (cell->at_boundary(face_index)) continue;

                  const auto neighbor = cell->neighbor(face_index);
                  neighbor->get_dof_indices(neighbor_dof_indices);
                  from_dofs.insert(from_dofs.end(), neighbor_dof_indices.begin(), neighbor_dof_indices.end());

                  if (depth > 1) self(self, neighbor, depth - 1);
                }
              };

              append_recursive(append_recursive, root_cell, 2);
            }

            template <std::size_t n_local_dofs>
            static void append_cell_dofs(std::vector<types::global_dof_index> &dofs,
                                         const std::array<types::global_dof_index, n_local_dofs> &cell_dofs)
            {
              dofs.insert(dofs.end(), cell_dofs.begin(), cell_dofs.end());
            }

            template <std::size_t n_local_dofs>
            void reinit(const std::array<types::global_dof_index, n_local_dofs> &cell_dofs, const Iterator &cell,
                        const std::optional<std::array<types::global_dof_index, n_local_dofs>> &ncell_dofs = std::nullopt,
                        const std::optional<Iterator> &ncell = std::nullopt)
            {
              to_dofs.resize(ncell_dofs.has_value() ? 2 * n_local_dofs : n_local_dofs);
              std::copy(cell_dofs.begin(), cell_dofs.end(), to_dofs.begin());
              if (ncell_dofs.has_value())
                std::copy(ncell_dofs->begin(), ncell_dofs->end(), to_dofs.begin() + n_local_dofs);
              from_dofs.clear();
              const auto required_capacity =
                  to_dofs.size() +
                  GeometryInfo<dim>::faces_per_cell * cell->get_fe().n_dofs_per_cell() * (ncell.has_value() ? 2 : 1);
              if (from_dofs.capacity() < required_capacity) from_dofs.reserve(required_capacity);
              append_cell_dofs(from_dofs, cell_dofs);
              if (ncell_dofs.has_value()) append_cell_dofs(from_dofs, *ncell_dofs);
              append_neighbor_dofs(from_dofs, neighbor_dof_indices, cell);
              if (ncell.has_value()) append_neighbor_dofs(from_dofs, neighbor_dof_indices, ncell.value());

              std::sort(from_dofs.begin(), from_dofs.end());
              from_dofs.erase(std::unique(from_dofs.begin(), from_dofs.end()), from_dofs.end());
              cell_jacobian.reinit(size(to_dofs), size(from_dofs));
            }
          };

          FullMatrix<NumberType> cell_jacobian;
          FullMatrix<NumberType> extractor_cell_jacobian;
          FullMatrix<NumberType> cell_mass_jacobian;
          std::vector<types::global_dof_index> local_dof_indices;
          std::vector<CopyDataFace_J> face_data;
          unsigned int active_face_count = 0;

          template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell, uint n_extractors)
          {
            cell_jacobian.reinit(dofs_per_cell, dofs_per_cell);
            if (n_extractors > 0) extractor_cell_jacobian.reinit(dofs_per_cell, n_extractors);
            cell_mass_jacobian.reinit(dofs_per_cell, dofs_per_cell);
            local_dof_indices.resize(dofs_per_cell);
            if (face_data.size() != cell->n_faces()) face_data.resize(cell->n_faces());
            active_face_count = 0;
            cell->get_dof_indices(local_dof_indices);
          }

          CopyDataFace_J &next_face_data()
          {
            AssertIndexRange(active_face_count, face_data.size());
            return face_data[active_face_count++];
          }
        };

        template <typename NumberType> struct CopyData_I {
          struct CopyFaceData_I {
            std::array<uint, 2> cell_indices;
            std::array<double, 2> values;
          };
          std::vector<CopyFaceData_I> face_data;
          double value;
          uint cell_index;
        };

        template <typename T> int sgn(T val) { return (T{} < val) - (val < T{}); }

        /**
         * @brief Result struct for compute_kt_flux_and_speeds.
         */
        template <int dim, typename NumberType, size_t n_components> struct KTFluxData {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F_plus;
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F_minus;
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> a_half;
        };

        template <typename WaveSpeedStrategy, typename Model, typename NumberType, int dim, size_t n_components>
        KTFluxData<dim, NumberType, n_components>
        compute_kt_flux_and_speeds(const std::array<NumberType, n_components> &u_plus,
                                   const std::array<NumberType, n_components> &u_minus, const dealii::Point<dim> &x_q,
                                   const Model &model)
        {
          using ADNumberType = autodiff::Real<1, NumberType>;

          KTFluxData<dim, NumberType, n_components> result{};

          std::array<ADNumberType, n_components> u_plus_AD{}, u_minus_AD{};
          for (size_t i = 0; i < n_components; ++i) {
            u_plus_AD[i] = ADNumberType(u_plus[i]);
            u_minus_AD[i] = ADNumberType(u_minus[i]);
          }

          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> F_AD_plus{}, F_AD_minus{};

          std::array<JacobianMatrix<NumberType, n_components>, dim> J_plus{}, J_minus{};

          for (size_t j = 0; j < n_components; ++j) {
            seed(u_plus_AD[j]);
            seed(u_minus_AD[j]);

            model.KurganovTadmor_advection_flux(F_AD_plus, x_q, advection_flux_tie(u_plus_AD));
            model.KurganovTadmor_advection_flux(F_AD_minus, x_q, advection_flux_tie(u_minus_AD));

            for (size_t d = 0; d < dim; ++d) {
              for (size_t i = 0; i < n_components; ++i) {
                J_plus[d][i][j] = autodiff::derivative(F_AD_plus[i][d]);
                J_minus[d][i][j] = autodiff::derivative(F_AD_minus[i][d]);

                if (j == 0) {
                  result.F_plus[i][d] = F_AD_plus[i][d].val();
                  result.F_minus[i][d] = F_AD_minus[i][d].val();
                }
              }
            }

            unseed(u_plus_AD[j]);
            unseed(u_minus_AD[j]);
          }

          const auto a = WaveSpeedStrategy::template compute_speeds<NumberType, dim, n_components>(J_plus, J_minus);

          for (size_t d = 0; d < dim; ++d)
            for (size_t component = 0; component < n_components; ++component)
              result.a_half[component][d] = a[d];

          return result;
        }
        template <int dim, typename NumberType, size_t n_components>
        std::array<dealii::Tensor<1, dim, NumberType>, n_components>
        compute_numerical_flux(const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &F_plus,
                               const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &F_minus,
                               const std::array<dealii::Tensor<1, dim, NumberType>, n_components> &a_half,
                               const std::array<NumberType, n_components> &u_plus,
                               const std::array<NumberType, n_components> &u_minus)
        {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> H{};
          for (size_t c = 0; c < n_components; ++c) {
            H[c] = (F_plus[c] + F_minus[c]) * 0.5 - a_half[c] * (u_plus[c] - u_minus[c]) * 0.5;
          }
          return H;
        }

      } // namespace internal

      namespace internal
      {
        template <typename WaveSpeedStrategy, typename Model, typename NumberType, int dim, size_t n_components>
        std::array<SimpleMatrix<dealii::Tensor<1, dim, NumberType>, n_components>, 2>
        compute_kt_numflux_jacobian(const std::array<NumberType, n_components> &u_plus,
                                    const std::array<NumberType, n_components> &u_minus, const dealii::Point<dim> &x_q,
                                    const Model &model)
        {
          // 1. Compute flux Jacobians J and full Hessian H via second-order forward-mode AD
          const auto [F_plus, J_plus, H_plus] =
              compute_flux_jacobian_and_hessian<Model, NumberType, dim, n_components>(u_plus, x_q, model);
          const auto [F_minus, J_minus, H_minus] =
              compute_flux_jacobian_and_hessian<Model, NumberType, dim, n_components>(u_minus, x_q, model);

          // 2. Compute a_half (max local wave speed per dimension) via the strategy
          const auto a = WaveSpeedStrategy::template compute_speeds<NumberType, dim, n_components>(J_plus, J_minus);

          // 3. Compute da_half / du_plus and da_half / du_minus via the strategy
          const auto [da_plus, da_minus] =
              WaveSpeedStrategy::template compute_speed_derivatives<NumberType, dim, n_components>(J_plus, J_minus,
                                                                                                   H_plus, H_minus);

          // 4. Assemble j_numflux
          std::array<SimpleMatrix<dealii::Tensor<1, dim, NumberType>, n_components>, 2> j_numflux{};

          for (size_t d = 0; d < dim; ++d) {
            for (size_t i = 0; i < n_components; ++i) {
              const NumberType du_i = u_plus[i] - u_minus[i];
              for (size_t c = 0; c < n_components; ++c) {
                const NumberType delta_ic = (i == c) ? NumberType(1) : NumberType(0);

                // dH_i^d / du_minus_c
                j_numflux[0](i, c)[d] = NumberType(0.5) * J_minus[d][i][c] + NumberType(0.5) * a[d] * delta_ic -
                                        NumberType(0.5) * du_i * da_minus[d][c];

                // dH_i^d / du_plus_c
                j_numflux[1](i, c)[d] = NumberType(0.5) * J_plus[d][i][c] - NumberType(0.5) * a[d] * delta_ic -
                                        NumberType(0.5) * du_i * da_plus[d][c];
              }
            }
          }

          return j_numflux;
        }

      } // namespace internal

      namespace internal
      {
        template <typename Model, typename NumberType, int dim, size_t n_components>
        std::array<dealii::Tensor<1, dim, NumberType>, n_components>
        compute_diffusion_flux(const std::array<NumberType, n_components> &u_plus,
                               const std::array<NumberType, n_components> &u_minus,
                               const GradientType<dim, NumberType, n_components> &grad_u_plus,
                               const GradientType<dim, NumberType, n_components> &grad_u_minus,
                               const dealii::Point<dim> &x_q, const Model &model)
        {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D_minus{}, D_plus{};
          model.flux(D_minus, x_q, flux_tie(u_minus, grad_u_minus));
          model.flux(D_plus, x_q, flux_tie(u_plus, grad_u_plus));

          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D{};
          for (size_t c = 0; c < n_components; ++c)
            D[c] = 0.5 * (D_minus[c] + D_plus[c]);
          return D;
        }

        template <int dim, typename NumberType, size_t n_components> struct DiffusionFluxJacobianData {
          std::array<SimpleMatrix<dealii::Tensor<1, dim, NumberType>, n_components>, 2> u{};
          std::array<SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<1, dim, NumberType>>, n_components>, 2> grad{};
        };

        template <typename Model, typename NumberType, int dim, size_t n_components>
        DiffusionFluxJacobianData<dim, NumberType, n_components>
        compute_diffusion_flux_jacobian(const std::array<NumberType, n_components> &u_plus,
                                        const std::array<NumberType, n_components> &u_minus,
                                        const GradientType<dim, NumberType, n_components> &grad_u_plus,
                                        const GradientType<dim, NumberType, n_components> &grad_u_minus,
                                        const dealii::Point<dim> &x_q, const Model &model)
        {
          using ADNumberType = autodiff::Real<1, NumberType>;

          DiffusionFluxJacobianData<dim, NumberType, n_components> result{};

          std::array<ADNumberType, n_components> u_plus_AD{}, u_minus_AD{};
          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> grad_u_plus_AD{}, grad_u_minus_AD{};
          for (size_t c = 0; c < n_components; ++c) {
            u_plus_AD[c] = ADNumberType(u_plus[c]);
            u_minus_AD[c] = ADNumberType(u_minus[c]);
            for (size_t d = 0; d < dim; ++d) {
              grad_u_plus_AD[c][d] = ADNumberType(grad_u_plus[c][d]);
              grad_u_minus_AD[c][d] = ADNumberType(grad_u_minus[c][d]);
            }
          }

          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> D_minus_AD{}, D_plus_AD{};

          for (size_t c = 0; c < n_components; ++c) {
            seed(u_minus_AD[c]);
            model.flux(D_minus_AD, x_q, flux_tie(u_minus_AD, grad_u_minus_AD));
            for (size_t i = 0; i < n_components; ++i)
              for (size_t d = 0; d < dim; ++d)
                result.u[0](i, c)[d] = NumberType(0.5) * derivative(D_minus_AD[i][d]);
            unseed(u_minus_AD[c]);

            seed(u_plus_AD[c]);
            model.flux(D_plus_AD, x_q, flux_tie(u_plus_AD, grad_u_plus_AD));
            for (size_t i = 0; i < n_components; ++i)
              for (size_t d = 0; d < dim; ++d)
                result.u[1](i, c)[d] = NumberType(0.5) * derivative(D_plus_AD[i][d]);
            unseed(u_plus_AD[c]);

            for (size_t d_in = 0; d_in < dim; ++d_in) {
              seed(grad_u_minus_AD[c][d_in]);
              model.flux(D_minus_AD, x_q, flux_tie(u_minus_AD, grad_u_minus_AD));
              for (size_t i = 0; i < n_components; ++i)
                for (size_t d_out = 0; d_out < dim; ++d_out)
                  result.grad[0](i, c)[d_out][d_in] = NumberType(0.5) * derivative(D_minus_AD[i][d_out]);
              unseed(grad_u_minus_AD[c][d_in]);

              seed(grad_u_plus_AD[c][d_in]);
              model.flux(D_plus_AD, x_q, flux_tie(u_plus_AD, grad_u_plus_AD));
              for (size_t i = 0; i < n_components; ++i)
                for (size_t d_out = 0; d_out < dim; ++d_out)
                  result.grad[1](i, c)[d_out][d_in] = NumberType(0.5) * derivative(D_plus_AD[i][d_out]);
              unseed(grad_u_plus_AD[c][d_in]);
            }
          }

          return result;
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
        std::array<std::array<autodiff::Real<1, NumberType>, n_components>, 5>
        make_tagged_physical_boundary_stencil(
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

      template <typename Discretization_, typename Model_,
                def::HasReconstructor Reconstructor_ =
                    def::TVDReconstructor<Discretization_::dim, def::MinModLimiter, double>,
                def::HasWaveSpeed WaveSpeedStrategy_ = MaxEigenvalueWaveSpeed>
        requires MeshIsRectangular<typename Discretization_::Mesh>
      class Assembler : public AbstractAssembler<typename Discretization_::VectorType,
                                                 typename Discretization_::SparseMatrixType, Discretization_::dim>
      {
      protected:
        template <typename... T> auto fv_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions">>(std::tie(t...));
        }

        template <typename... T> static constexpr auto v_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, StringSet<"variables", "extractors">>(std::tie(t...));
        }

        template <typename... T> static constexpr auto e_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>,
                             StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors", "variables">>(
              std::tie(t...));
        }

      public:
        using Discretization = Discretization_;
        using Model = Model_;
        using Reconstructor = Reconstructor_;
        using WaveSpeedStrategy = WaveSpeedStrategy_;
        using NumberType = typename Discretization::NumberType;
        using VectorType = typename Discretization::VectorType;

        using Components = typename Discretization::Components;
        static constexpr uint dim = Discretization::dim;
        static_assert(Reconstructor::dim == dim, "Reconstructor dimension must match the discretization dimension.");
        static constexpr uint n_components = Components::count_fe_functions(0);
        static constexpr uint n_faces = GeometryInfo<dim>::faces_per_cell;
        // using CacheData = internal::Cache_Data<NumberType, dim, n_components>;
        using GradientType = internal::GradientType<dim, NumberType, n_components>;
        using Iterator = typename DoFHandler<Discretization::dim>::active_cell_iterator;
        using Point = dealii::Point<dim>;
        using Scratch = internal::ScratchData<dim, NumberType, n_components>;
        struct CellGeometryCacheEntry {
          std::array<types::global_dof_index, n_components> dof_indices{};
          std::vector<Point> quadrature_points;
          std::vector<NumberType> jxw;
        };

        Assembler(Discretization &discretization, Model &model, const JSONValue &json)
            : discretization(discretization), model(model), dof_handler(discretization.get_dof_handler()),
              mapping(discretization.get_mapping()), triangulation(discretization.get_triangulation()), json(json),
              fe(discretization.get_fe()), threads(json.get_uint("/discretization/threads")),
              batch_size(json.get_uint("/discretization/batch_size")),
              EoM_abs_tol(json.get_double("/discretization/EoM_abs_tol")),
              EoM_max_iter(json.get_uint("/discretization/EoM_max_iter")),
              quadrature(1 + json.get_uint("/discretization/overintegration")),
              quadrature_face(1 + json.get_uint("/discretization/overintegration"))
        {
          if (this->threads == 0) this->threads = dealii::MultithreadInfo::n_threads() / 2;
          spdlog::get("log")->info("FV: Using {} threads for assembly.", threads);

          AssertThrow(fe.dofs_per_cell == n_components,
                      ExcMessage("FV Kurganov-Tadmor assembler expects one dof per component."));
          for (uint i = 0; i < n_components; ++i)
            local_component_of_dof[i] = fe.system_to_component_index(i).first;

          reinit();
        }

        virtual void reinit_vector(VectorType &vec) const override
        {
          const auto block_structure = discretization.get_block_structure();
          vec.reinit(block_structure[0]);
        }

        virtual IndexSet get_differential_indices() const override
        {
          ComponentMask component_mask(model.template differential_components<dim>());
          return DoFTools::extract_dofs(dof_handler, component_mask);
        }

        virtual void attach_data_output(DataOutput<dim, VectorType> &data_out, const VectorType &solution,
                                        const VectorType & /* variables*/, const VectorType &dt_solution = VectorType(),
                                        const VectorType &residual = VectorType())
        {
          const auto fe_function_names = Components::FEFunction_Descriptor::get_names_vector();
          std::vector<std::string> fe_function_names_residual;
          for (const auto &name : fe_function_names)
            fe_function_names_residual.push_back(name + "_residual");
          std::vector<std::string> fe_function_names_dot;
          for (const auto &name : fe_function_names)
            fe_function_names_dot.push_back(name + "_dot");

          auto &fe_out = data_out.fe_output();
          fe_out.attach(dof_handler, solution, fe_function_names);
          if (dt_solution.size() > 0) fe_out.attach(dof_handler, dt_solution, fe_function_names_dot);
          if (residual.size() > 0) fe_out.attach(dof_handler, residual, fe_function_names_residual);

          // readouts(data_out, solution, variables);
        }

        virtual void reinit() override
        {
          Timer timer;

          const auto metadata = DiFfRG::internal::build_affine_constraint_metadata<Components, dim>(discretization);
          const AffineConstraintContext<Components, dim> context(metadata);

          auto &constraints = discretization.get_constraints();
          constraints.clear();
          DoFTools::make_hanging_node_constraints(dof_handler, constraints);
          model.affine_constraints(constraints, context);
          constraints.close();

          // Mass sparsity pattern
          {
            DynamicSparsityPattern dsp(dof_handler.n_dofs());
            DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_dofs = */ true);
            sparsity_pattern_mass.copy_from(dsp);
            mass_matrix.reinit(sparsity_pattern_mass);
            MatrixCreator::create_mass_matrix(dof_handler, quadrature, mass_matrix,
                                              static_cast<Function<dim, NumberType> *>(nullptr), constraints);
          }
          // // Jacobian sparsity pattern
          // {
          //   DynamicSparsityPattern dsp(dof_handler.n_dofs());
          //   DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
          //                                        /*keep_constrained_dofs = */ true);
          //   sparsity_pattern_jacobian.copy_from(dsp);
          // }

          constexpr uint stencil = 2;
          build_sparsity(sparsity_pattern_jacobian, dof_handler, dof_handler, stencil, true);
          rebuild_cell_geometry_cache();

          timings_reinit.push_back(timer.wall_time());
        }

        virtual void set_time(double t) override { model.set_time(t); }

        virtual const SparsityPattern &get_sparsity_pattern_jacobian() const override
        {
          return sparsity_pattern_jacobian;
        }
        virtual const SparseMatrix<NumberType> &get_mass_matrix() const override { return mass_matrix; }

        virtual void residual_variables(VectorType &residual, const VectorType &variables, const VectorType &) override
        {
          model.dt_variables(residual, fv_tie(variables));
        };

        virtual void jacobian_variables([[maybe_unused]] FullMatrix<NumberType> &jacobian,
                                        [[maybe_unused]] const VectorType &variables, const VectorType &) override
        {
          static_assert(true, "jacobian_variables is not implemented for this model");
          // model.template jacobian_variables<0>(jacobian, fv_tie(variables));
        };

        void readouts(DataOutput<dim, VectorType> &data_out, const VectorType &, const VectorType &variables) const
        {
          auto helper = [&](auto EoMfun, auto outputter) {
            (void)EoMfun;
            outputter(data_out, dealii::Point<0>(), fv_tie(variables));
          };
          model.template readouts_multiple<0>(helper, data_out);
        }

        virtual void mass(VectorType &mass, const VectorType &solution_global, const VectorType &solution_global_dot,
                          NumberType weight) override
        {
          using CopyData = internal::CopyData_R<NumberType>;
          const auto &constraints = discretization.get_constraints();

          Scratch scratch_data(quadrature);
          CopyData copy_data;

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_geometry(cell);
            constexpr uint n_dofs = n_components;

            copy_data.reinit(cell, n_dofs);

            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch_data);

            std::array<NumberType, n_components> mass_values{};
            for (size_t q_index = 0; q_index < cell_geometry.quadrature_points.size(); ++q_index) {
              const auto &x_q = cell_geometry.quadrature_points[q_index];
              model.mass(mass_values, x_q, scratch_data.solution_values[q_index], scratch_data.solution_dot_values[q_index]);

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = local_component_of_dof[i];
                copy_data.cell_mass(i) +=
                    weight * cell_geometry.jxw[q_index] * mass_values[component_i]; // +phi_i(x_q) * mass(x_q, u_q)
              }
            }
          };

          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_mass, c.local_dof_indices, mass);
          };

          MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells;

          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, nullptr, nullptr, threads, batch_size);
        }

        using CellData = internal::CellData<dim, NumberType, n_components>;
        using NeighborData = internal::NeighborData<dim, NumberType, n_components>;
        using CellStencilData = internal::CellStencilData<dim, NumberType, n_components>;

        static void fill_cell_data(const Iterator &cell, const VectorType &solution_global,
                                   std::vector<types::global_dof_index> &scratch_dof_indices, CellData &data)
        {
          data.x = cell->center();
          cell->get_dof_indices(scratch_dof_indices);
          for (unsigned int i = 0; i < n_components; ++i) {
            data.dof_indices[i] = scratch_dof_indices[i];
            data.u[i] = solution_global(scratch_dof_indices[i]);
          }
        }

        static void fill_cell_stencil(const Iterator &cell, const VectorType &solution_global, const Model &model,
                                      std::vector<types::global_dof_index> &scratch_dof_indices,
                                      CellStencilData &stencil)
        {
          stencil.neighbors = {};
          stencil.boundary_ids.fill(numbers::invalid_boundary_id);
          stencil.face_centers = {};
          for (auto &neighbor_dof_indices : stencil.neighbors.dof_indices)
            neighbor_dof_indices.fill(numbers::invalid_dof_index);

          fill_cell_data(cell, solution_global, scratch_dof_indices, stencil.cell);

          for (const auto face_index : cell->face_indices()) {
            const auto face = cell->face(face_index);
            stencil.face_centers[face_index] = face->center();
            if (cell->at_boundary(face_index)) {
              stencil.boundary_ids[face_index] = face->boundary_id();
              if constexpr (dim == 1) {
                auto boundary_stencil =
                    build_boundary_stencil_1d<NumberType>(cell, face_index, solution_global, scratch_dof_indices);
                const bool boundary_supported = model.apply_boundary_stencil(
                    boundary_stencil.u, boundary_stencil.x, dealii::Point<1>(face->center()[0]));
                AssertThrow(boundary_supported,
                            ExcMessage("KT boundary stencil was rejected while populating a boundary-adjacent cell stencil."));

                stencil.neighbors.x[face_index] = boundary_stencil.x[boundary_stencil.ghost_center];
                stencil.neighbors.u[face_index] = boundary_stencil.u[boundary_stencil.ghost_center];
                stencil.neighbors.dof_indices[face_index] = boundary_stencil.dof_indices[boundary_stencil.ghost_center];
              }
              continue;
            }

            const auto neighbor = cell->neighbor(face_index);
            CellData neighbor_data;
            fill_cell_data(neighbor, solution_global, scratch_dof_indices, neighbor_data);
            stencil.neighbors.x[face_index] = neighbor_data.x;
            stencil.neighbors.u[face_index] = neighbor_data.u;
            stencil.neighbors.dof_indices[face_index] = neighbor_data.dof_indices;
          }

        }

        template <typename BoundaryNumberType>
        static internal::BoundaryStencilData1D<BoundaryNumberType, n_components>
        build_boundary_stencil_1d(const Iterator &cell, const unsigned int boundary_face_no,
                                  const VectorType &solution_global,
                                  std::vector<types::global_dof_index> &scratch_dof_indices)
        {
          static_assert(dim == 1, "Paper-style boundary stencils currently support only dim=1.");

          using namespace internal::BoundaryStencilIndex;
          const auto interior_face = GeometryInfo<1>::opposite_face[boundary_face_no];
          AssertThrow(!cell->at_boundary(interior_face),
                      ExcMessage("KT boundary stencil requires at least two interior cells behind the boundary face."));

          internal::BoundaryStencilData1D<BoundaryNumberType, n_components> boundary_stencil{};
          boundary_stencil.lower_boundary = boundary_face_no == 0;
          boundary_stencil.ghost_center = boundary_stencil.lower_boundary ? lower_inner : upper_inner;
          boundary_stencil.ghost_left = boundary_stencil.lower_boundary ? lower_outer : physical_cell;
          boundary_stencil.ghost_right = boundary_stencil.lower_boundary ? physical_cell : upper_outer;
          for (auto &dofs : boundary_stencil.dof_indices)
            dofs.fill(numbers::invalid_dof_index);

          CellData cell_data;
          fill_cell_data(cell, solution_global, scratch_dof_indices, cell_data);
          boundary_stencil.x[physical_cell] = cell_data.x;
          boundary_stencil.u[physical_cell] = cell_data.u;
          boundary_stencil.dof_indices[physical_cell] = cell_data.dof_indices;

          auto neighbor = cell->neighbor(interior_face);
          CellData first_interior;
          fill_cell_data(neighbor, solution_global, scratch_dof_indices, first_interior);

          AssertThrow(!neighbor->at_boundary(interior_face),
                      ExcMessage("KT boundary stencil requires a second interior cell behind the boundary face."));
          auto next_neighbor = neighbor->neighbor(interior_face);
          CellData second_interior;
          fill_cell_data(next_neighbor, solution_global, scratch_dof_indices, second_interior);

          if (boundary_stencil.lower_boundary) {
            boundary_stencil.x[upper_inner] = first_interior.x;
            boundary_stencil.x[upper_outer] = second_interior.x;
            boundary_stencil.u[upper_inner] = first_interior.u;
            boundary_stencil.u[upper_outer] = second_interior.u;
            boundary_stencil.dof_indices[upper_inner] = first_interior.dof_indices;
            boundary_stencil.dof_indices[upper_outer] = second_interior.dof_indices;
          } else {
            boundary_stencil.x[lower_inner] = first_interior.x;
            boundary_stencil.x[lower_outer] = second_interior.x;
            boundary_stencil.u[lower_inner] = first_interior.u;
            boundary_stencil.u[lower_outer] = second_interior.u;
            boundary_stencil.dof_indices[lower_inner] = first_interior.dof_indices;
            boundary_stencil.dof_indices[lower_outer] = second_interior.dof_indices;
          }

          return boundary_stencil;
        }

        static void fill_constant_quadrature_values(const Iterator &cell, const VectorType &solution_global,
                                                    const VectorType &solution_global_dot, Scratch &scratch_data)
        {
          fill_cell_data(cell, solution_global, scratch_data.cell_dof_indices, scratch_data.cell_stencil.cell);
          for (auto &values : scratch_data.solution_values)
            for (uint c = 0; c < n_components; ++c)
              values[c] = scratch_data.cell_stencil.cell.u[c];

          for (auto &values_dot : scratch_data.solution_dot_values)
            for (uint c = 0; c < n_components; ++c)
              values_dot[c] = solution_global_dot(scratch_data.cell_stencil.cell.dof_indices[c]);
        }

        static std::array<internal::GradientType<dim, NumberType, n_components>, n_faces>
        compute_neighbor_gradients(const Iterator &cell, const VectorType &solution_global, const Model &model,
                                   std::vector<types::global_dof_index> &scratch_dof_indices,
                                   CellStencilData &temporary_stencil)
        {
          std::array<internal::GradientType<dim, NumberType, n_components>, n_faces> gradients{};

          for (const auto face_index : cell->face_indices()) {
            if (cell->at_boundary(face_index)) continue;

            const auto neighbor = cell->neighbor(face_index);
            fill_cell_stencil(neighbor, solution_global, model, scratch_dof_indices, temporary_stencil);

            gradients[face_index] = Reconstructor::template compute_gradient<n_components>(
                temporary_stencil.cell.x, temporary_stencil.cell.u, temporary_stencil.neighbors.x,
                temporary_stencil.neighbors.u);
          }

          return gradients;
        }

        static Tensor<1, dim> face_normal_from_cell(const Iterator &cell, const unsigned int face_no)
        {
          const auto face_offset = cell->face(face_no)->center() - cell->center();
          const auto norm = face_offset.norm();
          AssertThrow(norm > 0., ExcMessage("Degenerate FV face normal."));
          return face_offset / norm;
        }

        static double face_jxw(const Iterator &cell, const unsigned int face_no)
        {
          if constexpr (dim == 1)
            return 1.;
          else
            return cell->face(face_no)->measure();
        }

        virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                              const VectorType &solution_global_dot, NumberType weight_mass,
                              const VectorType & /* variables */ = VectorType()) override
        {
          using CopyData = internal::CopyData_R<NumberType>;
          const auto &constraints = discretization.get_constraints();

          Scratch scratch_data(quadrature);
          CopyData copy_data;

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_geometry(cell);
            constexpr uint n_dofs = n_components;

            copy_data.reinit(cell, n_dofs);

            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch_data);

            std::array<NumberType, n_components> mass{};
            std::array<NumberType, n_components> source{};
            for (size_t q_index = 0; q_index < cell_geometry.quadrature_points.size(); ++q_index) {
              const auto &x_q = cell_geometry.quadrature_points[q_index];
              model.mass(mass, x_q, scratch_data.solution_values[q_index], scratch_data.solution_dot_values[q_index]);
              model.source(source, x_q, fv_tie(scratch_data.solution_values[q_index]));

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = local_component_of_dof[i];
                copy_data.cell_mass(i) +=
                    weight_mass * cell_geometry.jxw[q_index] * mass[component_i]; // +phi_i(x_q) * mass(x_q, u_q)
                copy_data.cell_residual(i) +=
                    cell_geometry.jxw[q_index] * weight * source[component_i]; // -phi_i(x_q) * source(x_q, u_q)
              }
            }
          };

          const auto face_worker = [&](const Iterator &cell, const unsigned int &f, const unsigned int &sf,
                                       const Iterator &ncell, const unsigned int &nf, const unsigned int &nsf,
                                       Scratch &scratch_data, CopyData &copy_data) {
            const int q_face_index = 0; // only one quadrature point per face for FV (constant FE)
            (void)q_face_index;
            const auto x_q = cell->face(f)->center();
            const auto n_face = face_normal_from_cell(cell, f);
            const auto JxW = face_jxw(cell, f);
            const uint n_face_dofs = 2 * n_components;

            auto &copy_data_face = copy_data.next_face_data();
            copy_data_face.reinit(n_face_dofs);

            fill_cell_stencil(cell, solution_global, model, scratch_data.cell_dof_indices,
                              scratch_data.cell_stencil);
            fill_cell_stencil(ncell, solution_global, model, scratch_data.ncell_dof_indices,
                              scratch_data.ncell_stencil);
            const auto &cell_stencil = scratch_data.cell_stencil;
            const auto &ncell_stencil = scratch_data.ncell_stencil;
            const auto &cell_neighbors = cell_stencil.neighbors;
            const auto &ncell_neighbors = ncell_stencil.neighbors;
            const auto &cell_data = cell_stencil.cell;
            const auto &ncell_data = ncell_stencil.cell;

            const GradientType u_grad_cell = Reconstructor::template compute_gradient<n_components>(
                cell_data.x, cell_data.u, cell_neighbors.x, cell_neighbors.u);
            const GradientType u_grad_ncell = Reconstructor::template compute_gradient<n_components>(
                ncell_data.x, ncell_data.u, ncell_neighbors.x, ncell_neighbors.u);
            const GradientType u_grad_minus = Reconstructor::template compute_gradient_at_point<n_components>(
                cell_data.x, x_q, cell_data.u, cell_neighbors.x, cell_neighbors.u);
            const GradientType u_grad_plus = Reconstructor::template compute_gradient_at_point<n_components>(
                ncell_data.x, x_q, ncell_data.u, ncell_neighbors.x, ncell_neighbors.u);

            const std::array<NumberType, n_components> u_minus =
                internal::reconstruct_u(cell_data.u, cell_data.x, x_q, u_grad_cell);
            const std::array<NumberType, n_components> u_plus =
                internal::reconstruct_u(ncell_data.u, ncell_data.x, x_q, u_grad_ncell);

            const auto [F_plus, F_minus, a_half] =
                internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(u_plus, u_minus, x_q, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
            const auto D = internal::compute_diffusion_flux(u_plus, u_minus, u_grad_plus, u_grad_minus, x_q, model);

            for (uint component_i = 0; component_i < n_components; ++component_i) {
              copy_data_face.joint_dof_indices[component_i] = cell_data.dof_indices[component_i];
              copy_data_face.joint_dof_indices[n_components + component_i] = ncell_data.dof_indices[component_i];
              const auto flux_contribution =
                  weight * JxW * (scalar_product(H[component_i], n_face) - scalar_product(D[component_i], n_face));
              copy_data_face.cell_residual(component_i) += flux_contribution;
              copy_data_face.cell_residual(n_components + component_i) -= flux_contribution;
            }
          };

          const auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no, Scratch &scratch_data,
                                           CopyData &copy_data) {
            const uint n_face_dofs = n_components;

            auto &copy_data_face = copy_data.next_face_data();
            copy_data_face.reinit(n_face_dofs);

            const int q_face_index = 0; // only one quadrature point per face for FV
            (void)q_face_index;
            const auto x_q = cell->face(face_no)->center();
            const auto JxW = face_jxw(cell, face_no);
            const auto n_bnd = face_normal_from_cell(cell, face_no);

            fill_cell_stencil(cell, solution_global, model, scratch_data.cell_dof_indices,
                              scratch_data.cell_stencil);
            const auto &cell_stencil = scratch_data.cell_stencil;
            const auto &cell_data = cell_stencil.cell;
            GradientType u_grad_plus{};
            GradientType u_grad_minus{};
            std::array<NumberType, n_components> u_plus{};
            std::array<NumberType, n_components> u_minus{};

            static_assert(dim == 1, "KT boundary-face assembly currently requires one-dimensional boundary stencils.");

            auto boundary_stencil =
                build_boundary_stencil_1d<NumberType>(cell, face_no, solution_global, scratch_data.ncell_dof_indices);
            const bool boundary_supported =
                model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x, dealii::Point<1>(x_q[0]));
            AssertThrow(boundary_supported, ExcMessage("KT boundary stencil was rejected by the model boundary policy."));

            const auto physical_stencil = internal::make_boundary_side_stencil_1d<NumberType, n_components>(
                boundary_stencil, internal::BoundaryStencilIndex::physical_cell, internal::BoundaryStencilIndex::lower_inner,
                internal::BoundaryStencilIndex::upper_inner);
            const auto ghost_stencil =
                internal::make_ghost_boundary_side_stencil_1d<NumberType, n_components>(boundary_stencil);

            const auto u_grad_cell = Reconstructor::template compute_gradient<n_components>(
                physical_stencil.cell.x, physical_stencil.cell.u, physical_stencil.neighbors.x, physical_stencil.neighbors.u);
            const auto u_grad_ghost = Reconstructor::template compute_gradient<n_components>(
                ghost_stencil.cell.x, ghost_stencil.cell.u, ghost_stencil.neighbors.x, ghost_stencil.neighbors.u);
            u_grad_minus = Reconstructor::template compute_gradient_at_point<n_components>(
                physical_stencil.cell.x, x_q, physical_stencil.cell.u, physical_stencil.neighbors.x,
                physical_stencil.neighbors.u);
            u_grad_plus = Reconstructor::template compute_gradient_at_point<n_components>(
                ghost_stencil.cell.x, x_q, ghost_stencil.cell.u, ghost_stencil.neighbors.x, ghost_stencil.neighbors.u);
            u_minus = internal::reconstruct_u(physical_stencil.cell.u, physical_stencil.cell.x, x_q, u_grad_cell);
            u_plus = internal::reconstruct_u(ghost_stencil.cell.u, ghost_stencil.cell.x, x_q, u_grad_ghost);

            const auto [F_plus, F_minus, a_half] =
                internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(u_plus, u_minus, x_q, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);

            const auto D_bnd = internal::compute_diffusion_flux(u_plus, u_minus, u_grad_plus, u_grad_minus, x_q, model);

            for (uint component_i = 0; component_i < n_components; ++component_i) {
              copy_data_face.joint_dof_indices[component_i] = cell_data.dof_indices[component_i];
              copy_data_face.cell_residual(component_i) +=
                  weight * JxW * (scalar_product(H[component_i], n_bnd) - scalar_product(D_bnd[component_i], n_bnd));
            }
          };

          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, residual);
            constraints.distribute_local_to_global(c.cell_mass, c.local_dof_indices, residual);
            for (unsigned int face_index = 0; face_index < c.active_face_count; ++face_index) {
              const auto &face_data = c.face_data[face_index];
              constraints.distribute_local_to_global(face_data.cell_residual, face_data.joint_dof_indices, residual);
            }
          };

          MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                            MeshWorker::assemble_own_interior_faces_once;

          Timer timer;

          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, boundary_worker, face_worker, threads, batch_size);
          timings_residual.push_back(timer.wall_time());
        }

        virtual void jacobian_mass(SparseMatrix<NumberType> &jacobian, const VectorType &solution_global,
                                   const VectorType &solution_global_dot, NumberType alpha = 1.,
                                   NumberType beta = 1.) override
        {
          using Iterator = typename DoFHandler<dim>::active_cell_iterator;
          using CopyData = internal::CopyData_J<NumberType, dim>;
          const auto &constraints = discretization.get_constraints();

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_geometry(cell);
            constexpr uint n_dofs = n_components;

            copy_data.reinit(cell, n_dofs, Components::count_extractors());

            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch_data);

            SimpleMatrix<NumberType, n_components> j_mass;
            SimpleMatrix<NumberType, n_components> j_mass_dot;
            for (size_t q_index = 0; q_index < cell_geometry.quadrature_points.size(); ++q_index) {
              const auto &x_q = cell_geometry.quadrature_points[q_index];
              model.template jacobian_mass<0>(j_mass, x_q, scratch_data.solution_values[q_index],
                                              scratch_data.solution_dot_values[q_index]);
              model.template jacobian_mass<1>(j_mass_dot, x_q, scratch_data.solution_values[q_index],
                                              scratch_data.solution_dot_values[q_index]);

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = local_component_of_dof[i];
                for (uint j = 0; j < n_dofs; ++j) {
                  const auto component_j = local_component_of_dof[j];
                  copy_data.cell_jacobian(i, j) += cell_geometry.jxw[q_index] *
                                                   (alpha * j_mass_dot(component_i, component_j) +
                                                    beta * j_mass(component_i, component_j));
                }
              }
            }
          };
          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_jacobian, c.local_dof_indices, jacobian);
          };

          Scratch scratch_data(quadrature);
          CopyData copy_data;
          MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells;

          Timer timer;
          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, nullptr, nullptr, threads, batch_size);
          timings_jacobian.push_back(timer.wall_time());
        }

        virtual void jacobian(SparseMatrix<NumberType> &jacobian, const VectorType &solution_global, NumberType weight,
                              const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                              const VectorType & /* variables */ = VectorType()) override
        {
          using Iterator = typename DoFHandler<dim>::active_cell_iterator;
          using CopyData = internal::CopyData_J<NumberType, dim>;
          const auto &constraints = discretization.get_constraints();

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_geometry(cell);
            constexpr uint n_dofs = n_components;

            copy_data.reinit(cell, n_dofs, Components::count_extractors());

            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch_data);

            SimpleMatrix<NumberType, n_components> j_mass;
            SimpleMatrix<NumberType, n_components> j_mass_dot;
            SimpleMatrix<NumberType, n_components> j_source;
            for (size_t q_index = 0; q_index < cell_geometry.quadrature_points.size(); ++q_index) {
              const auto &x_q = cell_geometry.quadrature_points[q_index];
              model.template jacobian_mass<0>(j_mass, x_q, scratch_data.solution_values[q_index],
                                              scratch_data.solution_dot_values[q_index]);
              model.template jacobian_mass<1>(j_mass_dot, x_q, scratch_data.solution_values[q_index],
                                              scratch_data.solution_dot_values[q_index]);
              model.template jacobian_source<0, 0>(j_source, x_q, fv_tie(scratch_data.solution_values[q_index]));

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = local_component_of_dof[i];
                for (uint j = 0; j < n_dofs; ++j) {
                  const auto component_j = local_component_of_dof[j];
                  copy_data.cell_jacobian(i, j) +=
                      weight * cell_geometry.jxw[q_index] * j_source(component_i, component_j); // -phi_i * jsource
                  copy_data.cell_mass_jacobian(i, j) +=
                      cell_geometry.jxw[q_index] *
                      (alpha * j_mass_dot(component_i, component_j) + beta * j_mass(component_i, component_j));
                }
              }
            }
          };
          const auto face_worker = [&](const Iterator &cell, const unsigned int &f, const unsigned int &sf,
                                       const Iterator &ncell, const unsigned int &nf, const unsigned int &nsf,
                                       Scratch &scratch_data, CopyData &copy_data) {
            const int q_face_index = 0;
            (void)q_face_index;
            const auto x_q = cell->face(f)->center();
            const auto JxW = face_jxw(cell, f);
            const auto n_face = face_normal_from_cell(cell, f);

            fill_cell_stencil(cell, solution_global, model, scratch_data.cell_dof_indices,
                              scratch_data.cell_stencil);
            fill_cell_stencil(ncell, solution_global, model, scratch_data.ncell_dof_indices,
                              scratch_data.ncell_stencil);
            const auto &cell_stencil = scratch_data.cell_stencil;
            const auto &ncell_stencil = scratch_data.ncell_stencil;
            const auto &cell_neighbors = cell_stencil.neighbors;
            const auto &ncell_neighbors = ncell_stencil.neighbors;
            const auto &cell_data = cell_stencil.cell;
            const auto &ncell_data = ncell_stencil.cell;

            auto &copy_data_face = copy_data.next_face_data();
            copy_data_face.reinit(cell_data.dof_indices, cell,
                                  std::optional<decltype(ncell_data.dof_indices)>{ncell_data.dof_indices}, ncell);

            // Compute gradients and reconstructed interface values u_minus / u_plus
            const GradientType u_grad_cell = Reconstructor::template compute_gradient<n_components>(
                cell_data.x, cell_data.u, cell_neighbors.x, cell_neighbors.u);
            const GradientType u_grad_ncell = Reconstructor::template compute_gradient<n_components>(
                ncell_data.x, ncell_data.u, ncell_neighbors.x, ncell_neighbors.u);
            const GradientType u_grad_minus = Reconstructor::template compute_gradient_at_point<n_components>(
                cell_data.x, x_q, cell_data.u, cell_neighbors.x, cell_neighbors.u);
            const GradientType u_grad_plus = Reconstructor::template compute_gradient_at_point<n_components>(
                ncell_data.x, x_q, ncell_data.u, ncell_neighbors.x, ncell_neighbors.u);

            const std::array<NumberType, n_components> u_minus =
                internal::reconstruct_u(cell_data.u, cell_data.x, x_q, u_grad_cell);
            const std::array<NumberType, n_components> u_plus =
                internal::reconstruct_u(ncell_data.u, ncell_data.x, x_q, u_grad_ncell);

            // Precompute reconstructed state and face-gradient derivatives for each dependency dof.
            const uint n_from = size(copy_data_face.from_dofs);
            for (auto &derivatives : scratch_data.reconstructed_derivatives) {
              if (derivatives.capacity() < n_from) derivatives.reserve(n_from);
              derivatives.resize(n_from);
            }
            auto &reconstructed_deriv = scratch_data.reconstructed_derivatives;

            for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
              const auto dof_j = copy_data_face.from_dofs[j];

              // face_no=0: d(u⁻)/d(u_j) — reconstruction from cell side
              {
                auto u_center_tagged = internal::tag_cell_dofs(cell_data, dof_j);
                auto u_n_tagged = internal::make_tagged_neighbors(cell_neighbors, dof_j);
                reconstructed_deriv[0][j].u =
                    internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
                        u_center_tagged.u, cell_data.x, x_q, cell_neighbors.x, u_n_tagged.u);
                reconstructed_deriv[0][j].grad =
                    Reconstructor::template compute_gradient_at_point_derivative<n_components>(
                        cell_data.x, x_q, u_center_tagged.u, cell_neighbors.x, u_n_tagged.u);
              }

              // face_no=1: d(u⁺)/d(u_j) — reconstruction from neighbor side
              {
                auto u_center_tagged = internal::tag_cell_dofs(ncell_data, dof_j);
                auto u_n_tagged = internal::make_tagged_neighbors(ncell_neighbors, dof_j);
                reconstructed_deriv[1][j].u =
                    internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
                        u_center_tagged.u, ncell_data.x, x_q, ncell_neighbors.x, u_n_tagged.u);
                reconstructed_deriv[1][j].grad =
                    Reconstructor::template compute_gradient_at_point_derivative<n_components>(
                        ncell_data.x, x_q, u_center_tagged.u, ncell_neighbors.x, u_n_tagged.u);
              }
            }

            const auto j_numflux =
                internal::compute_kt_numflux_jacobian<WaveSpeedStrategy, Model, NumberType, dim, n_components>(
                    u_plus, u_minus, x_q, model);
            const auto j_diffusion = internal::compute_diffusion_flux_jacobian<Model, NumberType, dim, n_components>(
                u_plus, u_minus, u_grad_plus, u_grad_minus, x_q, model);

            for (uint i = 0; i < size(copy_data_face.to_dofs); ++i) {
              const bool cell_side_i = i < n_components;
              const auto component_i = cell_side_i ? i : i - n_components;
              const auto jump_i = cell_side_i ? NumberType(1.) : NumberType(-1.);
              for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
                NumberType diffusion_contribution{};
                for (size_t face_no = 0; face_no < 2; ++face_no) {
                  NumberType advection_contribution{};
                  for (size_t c = 0; c < n_components; ++c) {
                    advection_contribution +=
                        scalar_product(j_numflux[face_no](component_i, c), n_face) *
                        reconstructed_deriv[face_no][j].u[c];

                    diffusion_contribution +=
                        scalar_product(j_diffusion.u[face_no](component_i, c), n_face) *
                        reconstructed_deriv[face_no][j].u[c];
                    for (size_t d_in = 0; d_in < dim; ++d_in)
                      for (size_t d_out = 0; d_out < dim; ++d_out)
                        diffusion_contribution += j_diffusion.grad[face_no](component_i, c)[d_out][d_in] *
                                                  n_face[d_out] *
                                                  reconstructed_deriv[face_no][j].grad[c][d_in];
                  }
                  copy_data_face.cell_jacobian(i, j) +=
                      weight * JxW * jump_i * advection_contribution;
                }

                // The residual uses [[phi_i]] * ((H - D) · n), so after summing the
                // contributions from both face traces into diffusion_contribution we
                // subtract the full diffusive chain-rule term once here.
                copy_data_face.cell_jacobian(i, j) -=
                    weight * JxW * jump_i * diffusion_contribution;
              }
            }
          };

          const auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no, Scratch &scratch_data,
                                           CopyData &copy_data) {
            const int q_face_index = 0;
            (void)q_face_index;
            const auto x_q = cell->face(face_no)->center();
            const auto JxW = face_jxw(cell, face_no);
            const auto n_face = face_normal_from_cell(cell, face_no);

            fill_cell_stencil(cell, solution_global, model, scratch_data.cell_dof_indices,
                              scratch_data.cell_stencil);
            const auto &cell_stencil = scratch_data.cell_stencil;
            const auto &cell_data = cell_stencil.cell;
            auto &copy_data_face = copy_data.next_face_data();
            copy_data_face.reinit(cell_data.dof_indices, cell);
            GradientType u_grad_ghost{};
            GradientType u_grad_minus{};
            std::array<NumberType, n_components> u_plus{};
            std::array<NumberType, n_components> u_minus{};

            static_assert(dim == 1, "KT boundary-face Jacobians currently require one-dimensional boundary stencils.");
            const auto boundary_stencil =
                build_boundary_stencil_1d<NumberType>(cell, face_no, solution_global, scratch_data.ncell_dof_indices);

            auto conditioned_boundary_stencil = boundary_stencil;
            const bool boundary_supported =
                model.apply_boundary_stencil(conditioned_boundary_stencil.u, conditioned_boundary_stencil.x,
                                             dealii::Point<1>(x_q[0]));
            AssertThrow(boundary_supported, ExcMessage("KT boundary stencil was rejected by the model boundary policy."));

            const auto physical_stencil = internal::make_boundary_side_stencil_1d<NumberType, n_components>(
                conditioned_boundary_stencil, internal::BoundaryStencilIndex::physical_cell,
                internal::BoundaryStencilIndex::lower_inner, internal::BoundaryStencilIndex::upper_inner);
            const auto ghost_stencil =
                internal::make_ghost_boundary_side_stencil_1d<NumberType, n_components>(conditioned_boundary_stencil);

            const auto u_grad_cell = Reconstructor::template compute_gradient<n_components>(
                physical_stencil.cell.x, physical_stencil.cell.u, physical_stencil.neighbors.x, physical_stencil.neighbors.u);
            const auto u_grad_plus = Reconstructor::template compute_gradient<n_components>(
                ghost_stencil.cell.x, ghost_stencil.cell.u, ghost_stencil.neighbors.x, ghost_stencil.neighbors.u);
            u_grad_minus = Reconstructor::template compute_gradient_at_point<n_components>(
                physical_stencil.cell.x, x_q, physical_stencil.cell.u, physical_stencil.neighbors.x,
                physical_stencil.neighbors.u);
            u_grad_ghost = Reconstructor::template compute_gradient_at_point<n_components>(
                ghost_stencil.cell.x, x_q, ghost_stencil.cell.u, ghost_stencil.neighbors.x, ghost_stencil.neighbors.u);
            u_minus = internal::reconstruct_u(physical_stencil.cell.u, physical_stencil.cell.x, x_q, u_grad_cell);
            u_plus = internal::reconstruct_u(ghost_stencil.cell.u, ghost_stencil.cell.x, x_q, u_grad_plus);

            // Precompute reconstructed state and face-gradient derivatives for each dependency dof.
            const uint n_from = size(copy_data_face.from_dofs);
            for (auto &derivatives : scratch_data.reconstructed_derivatives) {
              if (derivatives.capacity() < n_from) derivatives.reserve(n_from);
              derivatives.resize(n_from);
            }
            auto &reconstructed_deriv = scratch_data.reconstructed_derivatives;

            for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
              const auto dof_j = copy_data_face.from_dofs[j];
              auto u_stencil_tagged = internal::make_tagged_physical_boundary_stencil<NumberType, n_components>(
                  boundary_stencil.u, boundary_stencil.dof_indices, dof_j);
              internal::BoundaryStencilData1D<autodiff::Real<1, NumberType>, n_components> boundary_stencil_ad{};
              boundary_stencil_ad.x = boundary_stencil.x;
              boundary_stencil_ad.u = u_stencil_tagged;
              boundary_stencil_ad.dof_indices = boundary_stencil.dof_indices;
              boundary_stencil_ad.lower_boundary = boundary_stencil.lower_boundary;
              boundary_stencil_ad.ghost_center = boundary_stencil.ghost_center;
              boundary_stencil_ad.ghost_left = boundary_stencil.ghost_left;
              boundary_stencil_ad.ghost_right = boundary_stencil.ghost_right;
              const bool boundary_supported_ad =
                  model.apply_boundary_stencil(boundary_stencil_ad.u, boundary_stencil_ad.x, dealii::Point<1>(x_q[0]));
              AssertThrow(boundary_supported_ad,
                          ExcMessage("KT boundary stencil was rejected during AD reconstruction tracing."));
              const auto physical_stencil_ad =
                  internal::make_physical_boundary_side_stencil_1d<autodiff::Real<1, NumberType>, n_components>(
                      boundary_stencil_ad);
              const auto ghost_stencil_ad =
                  internal::make_ghost_boundary_side_stencil_1d<autodiff::Real<1, NumberType>, n_components>(
                      boundary_stencil_ad);
              using ReconstructorAD = def::TVDReconstructor<1, typename Reconstructor::LimiterType,
                                                            autodiff::Real<1, NumberType>>;
              const auto u_grad_cell_ad = ReconstructorAD::template compute_gradient<n_components>(
                  physical_stencil_ad.cell.x, physical_stencil_ad.cell.u, physical_stencil_ad.neighbors.x,
                  physical_stencil_ad.neighbors.u);
              const auto u_grad_ghost_ad = ReconstructorAD::template compute_gradient<n_components>(
                  ghost_stencil_ad.cell.x, ghost_stencil_ad.cell.u, ghost_stencil_ad.neighbors.x,
                  ghost_stencil_ad.neighbors.u);
              const auto u_grad_minus_ad = ReconstructorAD::template compute_gradient_at_point<n_components>(
                  physical_stencil_ad.cell.x, x_q, physical_stencil_ad.cell.u, physical_stencil_ad.neighbors.x,
                  physical_stencil_ad.neighbors.u);
              const auto u_grad_plus_ad = ReconstructorAD::template compute_gradient_at_point<n_components>(
                  ghost_stencil_ad.cell.x, x_q, ghost_stencil_ad.cell.u, ghost_stencil_ad.neighbors.x,
                  ghost_stencil_ad.neighbors.u);
              const auto u_minus_ad =
                  internal::reconstruct_u(physical_stencil_ad.cell.u, physical_stencil_ad.cell.x, x_q, u_grad_cell_ad);
              const auto u_plus_ad =
                  internal::reconstruct_u(ghost_stencil_ad.cell.u, ghost_stencil_ad.cell.x, x_q, u_grad_ghost_ad);
              for (size_t c = 0; c < n_components; ++c) {
                reconstructed_deriv[0][j].u[c] = derivative(u_minus_ad[c]);
                reconstructed_deriv[1][j].u[c] = derivative(u_plus_ad[c]);
                for (size_t d = 0; d < dim; ++d) {
                  reconstructed_deriv[0][j].grad[c][d] = derivative(u_grad_minus_ad[c][d]);
                  reconstructed_deriv[1][j].grad[c][d] = derivative(u_grad_plus_ad[c][d]);
                }
              }
            }

            // Compute numerical flux Jacobian
            const auto j_numflux =
                internal::compute_kt_numflux_jacobian<WaveSpeedStrategy, Model, NumberType, dim, n_components>(
                    u_plus, u_minus, x_q, model);
            const auto j_diffusion = internal::compute_diffusion_flux_jacobian<Model, NumberType, dim, n_components>(
                u_plus, u_minus, u_grad_ghost, u_grad_minus, x_q, model);

            // Chain-rule assembly (same pattern as interior face_worker)
            for (uint i = 0; i < size(copy_data_face.to_dofs); ++i) {
              const auto component_i = i;
              for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
                NumberType diffusion_contribution{};
                for (size_t face_no = 0; face_no < 2; ++face_no) {
                  NumberType advection_contribution{};
                  for (size_t c = 0; c < n_components; ++c) {
                    advection_contribution +=
                        scalar_product(j_numflux[face_no](component_i, c), n_face) *
                        reconstructed_deriv[face_no][j].u[c];

                    diffusion_contribution +=
                        scalar_product(j_diffusion.u[face_no](component_i, c), n_face) *
                        reconstructed_deriv[face_no][j].u[c];
                    for (size_t d_in = 0; d_in < dim; ++d_in)
                      for (size_t d_out = 0; d_out < dim; ++d_out)
                        diffusion_contribution += j_diffusion.grad[face_no](component_i, c)[d_out][d_in] *
                                                  n_face[d_out] *
                                                  reconstructed_deriv[face_no][j].grad[c][d_in];
                  }
                  copy_data_face.cell_jacobian(i, j) +=
                      weight * JxW * advection_contribution;
                }

                // Boundary faces use the same residual sign convention: [[phi_i]] *
                // ((H - D) · n). diffusion_contribution already contains both the
                // interior-side and ghost-side chain-rule pieces, so we subtract it
                // once after the face_no sum.
                copy_data_face.cell_jacobian(i, j) -=
                    weight * JxW * diffusion_contribution;
              }
            }
          };

          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_jacobian, c.local_dof_indices, jacobian);
            constraints.distribute_local_to_global(c.cell_mass_jacobian, c.local_dof_indices, jacobian);
            for (unsigned int face_index = 0; face_index < c.active_face_count; ++face_index) {
              const auto &face_data = c.face_data[face_index];
              constraints.distribute_local_to_global(face_data.cell_jacobian, face_data.to_dofs, face_data.from_dofs,
                                                     jacobian);
            }
          };

          Scratch scratch_data(quadrature);
          CopyData copy_data;
          MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                            MeshWorker::assemble_own_interior_faces_once;

          Timer timer;
          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, boundary_worker, face_worker, threads, batch_size);
          timings_jacobian.push_back(timer.wall_time());
        }

        virtual void refinement_indicator([[maybe_unused]] Vector<double> &indicator,
                                          [[maybe_unused]] const VectorType &solution_global)
        {
        }

        void build_sparsity(SparsityPattern &sparsity_pattern, const DoFHandler<dim> &to_dofh,
                            const DoFHandler<dim> &from_dofh, const int stencil = 2,
                            [[maybe_unused]] bool add_extractor_dofs = false) const
        {
          const auto &triangulation = discretization.get_triangulation();

          DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

          const auto to_dofs_per_cell = to_dofh.get_fe().dofs_per_cell;
          const auto from_dofs_per_cell = from_dofh.get_fe().dofs_per_cell;

          for (const auto &t_cell : triangulation.active_cell_iterators()) {
            std::vector<types::global_dof_index> to_dofs(to_dofs_per_cell);
            std::vector<types::global_dof_index> from_dofs;
            from_dofs.reserve(from_dofs_per_cell +
                              stencil * from_dofs_per_cell); // reserve enough space for the cell itself + neighbors
            const auto to_cell = typename DoFHandler<dim>::active_cell_iterator(
                &to_dofh.get_triangulation(), t_cell->level(), t_cell->index(), &to_dofh);
            const auto from_cell = typename DoFHandler<dim>::active_cell_iterator(
                &from_dofh.get_triangulation(), t_cell->level(), t_cell->index(), &from_dofh);
            to_cell->get_dof_indices(to_dofs);
            from_cell->get_dof_indices(from_dofs);

            std::function<void(decltype(from_cell) &, const int)> add_all_neighbor_dofs =
                [&](const auto &from_cell, const int stencil_level = 1) {
                  for (const auto face_no : from_cell->face_indices()) {
                    const auto face = from_cell->face(face_no);
                    if (!face->at_boundary()) {
                      auto neighbor_cell = from_cell->neighbor(face_no);

                      if (neighbor_cell->has_children()) {
                        throw std::runtime_error("AMR is not yet supported in the Kurganov-Tadmor assembler.");
                      }

                      std::vector<types::global_dof_index> tmp(from_dofs_per_cell);
                      neighbor_cell->get_dof_indices(tmp);

                      from_dofs.insert(std::end(from_dofs), std::begin(tmp), std::end(tmp)); // Let's wait for C++23 :(

                      if (stencil_level < stencil) add_all_neighbor_dofs(neighbor_cell, stencil_level + 1);
                    }
                  }
                };

            add_all_neighbor_dofs(from_cell, 1);

            for (const auto i : to_dofs)
              for (const auto j : from_dofs)
                dsp.add(i, j);
          }

          // if (add_extractor_dofs)
          //   throw std::runtime_error("Extractor dofs are not yet supported in the Kurganov-Tadmor assembler.");
          sparsity_pattern.copy_from(dsp);
        }

        void rebuild_cell_geometry_cache()
        {
          FEValues<dim> fe_values(mapping, fe, quadrature, update_quadrature_points | update_JxW_values);
          cell_geometry_cache.clear();
          cell_geometry_cache.resize(triangulation.n_active_cells());

          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            auto &cache_entry = cell_geometry_cache[cell->active_cell_index()];
            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            for (uint i = 0; i < n_components; ++i)
              cache_entry.dof_indices[i] = dof_indices[i];
            cache_entry.quadrature_points = fe_values.get_quadrature_points();
            cache_entry.jxw.assign(fe_values.get_JxW_values().begin(), fe_values.get_JxW_values().end());
          }
        }

        const CellGeometryCacheEntry &get_cell_geometry(const Iterator &cell) const
        {
          const auto cell_index = cell->active_cell_index();
          AssertIndexRange(cell_index, cell_geometry_cache.size());
          return cell_geometry_cache[cell_index];
        }

        void log(const std::string logger)
        {
          std::stringstream ss;
          ss << "FV Assembler: " << std::endl;
          ss << "        Reinit: " << average_time_reinit() * 1000 << "ms (" << num_reinits() << ")" << std::endl;
          ss << "        Residual: " << average_time_residual_assembly() * 1000 << "ms (" << num_residuals() << ")"
             << std::endl;
          ss << "        Jacobian: " << average_time_jacobian_assembly() * 1000 << "ms (" << num_jacobians() << ")"
             << std::endl;
          spdlog::get(logger)->info(ss.str());
        }

        double average_time_reinit() const
        {
          double t = 0.;
          double n = timings_reinit.size();
          for (const auto &t_ : timings_reinit)
            t += t_ / n;
          return t;
        }
        uint num_reinits() const { return timings_reinit.size(); }

        double average_time_residual_assembly() const
        {
          double t = 0.;
          double n = timings_residual.size();
          for (const auto &t_ : timings_residual)
            t += t_ / n;
          return t;
        }
        uint num_residuals() const { return timings_residual.size(); }

        double average_time_jacobian_assembly() const
        {
          double t = 0.;
          double n = timings_jacobian.size();
          for (const auto &t_ : timings_jacobian)
            t += t_ / n;
          return t;
        }
        uint num_jacobians() const { return timings_jacobian.size(); }

      protected:
        Discretization &discretization;
        Model &model;
        const DoFHandler<dim> &dof_handler;
        const Mapping<dim> &mapping;
        const Triangulation<dim> &triangulation;
        const FiniteElement<dim> &fe;

        const JSONValue &json;

        uint threads;
        const uint batch_size;

        mutable typename Triangulation<dim>::cell_iterator EoM_cell;
        typename Triangulation<dim>::cell_iterator old_EoM_cell;
        const double EoM_abs_tol;
        const uint EoM_max_iter;

        const QGauss<dim> quadrature;
        const QGauss<dim - 1> quadrature_face;

        SparsityPattern sparsity_pattern_mass;
        SparsityPattern sparsity_pattern_jacobian;
        SparseMatrix<NumberType> mass_matrix;

        std::vector<double> timings_reinit;
        std::vector<double> timings_residual;
        std::vector<double> timings_jacobian;
        std::array<unsigned int, n_components> local_component_of_dof{};
        std::vector<CellGeometryCacheEntry> cell_geometry_cache;
      };
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
