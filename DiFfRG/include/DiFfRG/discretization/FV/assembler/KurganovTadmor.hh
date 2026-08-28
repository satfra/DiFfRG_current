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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_iterator_base.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/assemble_flags.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <spdlog/spdlog.h>
#include <sstream>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/first_order_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/tvd_reconstructor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>
#include <DiFfRG/discretization/common/assembly_schedule.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/discretization/FV/assembler/assembly_context.hh>
#include <DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh>
#include <DiFfRG/discretization/FV/assembler/flux_ties.hh>
#include <DiFfRG/discretization/FV/assembler/reconstruction_cache.hh>
#include <DiFfRG/discretization/FV/wave_speed/abstract_wave_speed.hh>
#include <DiFfRG/discretization/FV/wave_speed/max_eigenvalue_wave_speed.hh>
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
        /**
         * @brief Class to hold data for each assembly thread, i.e. FEValues for cells, interfaces, as well as
         * pre-allocated data structures for the solutions
         */
        template <int dim, typename NumberType, size_t n_components> struct ScratchData {
          using QuadratureValue = std::array<NumberType, n_components>;

          ScratchData(const dealii::Quadrature<dim> &quadrature)
              : cell_dof_indices(n_components), ncell_dof_indices(n_components), solution_values(quadrature.size()),
                solution_dot_values(quadrature.size())
          {
          }

          ScratchData(const ScratchData<dim, NumberType, n_components> &scratch_data)
              : cell_dof_indices(n_components), ncell_dof_indices(n_components),
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
          std::array<std::vector<ReconstructionDerivativeData<dim, NumberType, n_components>>, 2> diffusion_derivatives;
          // d(cell-centre gradient)/d(u_j) for the nonlocal part of the source jacobian. Separate from
          // reconstructed_derivatives, which the face workers resize for their own dependency sets.
          std::vector<GradientType<dim, NumberType, n_components>> source_gradient_derivatives;
          CellStencilData<dim, NumberType, n_components> cell_stencil;
          CellStencilData<dim, NumberType, n_components> ncell_stencil;
          CellStencilData<dim, NumberType, n_components> temporary_stencil;
        };

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

            void reinit(const std::vector<types::global_dof_index> &cached_to_dofs,
                        const std::vector<types::global_dof_index> &cached_from_dofs)
            {
              to_dofs = cached_to_dofs;
              from_dofs = cached_from_dofs;
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
            // +1: the cell worker claims one block for the nonlocal source jacobian before the faces do.
            if (face_data.size() != cell->n_faces() + 1) face_data.resize(cell->n_faces() + 1);
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
            std::array<uint, 2> cell_indices{};
            std::array<double, 2> values{};
          };
          std::vector<CopyFaceData_I> face_data;
          double value = 0.;
          uint cell_index = 0;
        };

        template <typename T> int sgn(T val) { return (T{} < val) - (val < T{}); }

        /**
         * @brief Copy of @p J with every row and column outside @p block zeroed.
         *
         * The spectral radius of the result is that of the block's own diagonal sub-matrix, padded
         * with zero eigenvalues, so a wave-speed strategy can be handed this and needs to know
         * nothing about blocks. @see AbstractModel::wave_speed_blocks.
         */
        template <typename NumberType, int dim, size_t n_components>
        std::array<JacobianMatrix<NumberType, n_components>, dim>
        restrict_jacobian_to_block(const std::array<JacobianMatrix<NumberType, n_components>, dim> &J,
                                   const std::array<int, n_components> &blocks, const int block)
        {
          std::array<JacobianMatrix<NumberType, n_components>, dim> restricted{};
          for (size_t d = 0; d < dim; ++d)
            for (size_t i = 0; i < n_components; ++i) {
              if (blocks[i] != block) continue;
              for (size_t j = 0; j < n_components; ++j)
                if (blocks[j] == block) restricted[d][i][j] = J[d][i][j];
            }
          return restricted;
        }

        /**
         * @brief Copy of @p H with the two jacobian indices restricted to @p block.
         *
         * The differentiation index is left alone: the block's speed depends on the whole solution
         * through the entries of its own sub-matrix. @see restrict_jacobian_to_block.
         */
        template <typename NumberType, int dim, size_t n_components>
        HessianTensor<NumberType, dim, n_components>
        restrict_hessian_to_block(const HessianTensor<NumberType, dim, n_components> &H,
                                  const std::array<int, n_components> &blocks, const int block)
        {
          HessianTensor<NumberType, dim, n_components> restricted{};
          for (size_t d = 0; d < dim; ++d)
            for (size_t i = 0; i < n_components; ++i) {
              if (blocks[i] != block) continue;
              for (size_t j = 0; j < n_components; ++j) {
                if (blocks[j] != block) continue;
                for (size_t c = 0; c < n_components; ++c)
                  restricted[d][i][j][c] = H[d][i][j][c];
              }
            }
          return restricted;
        }

        /**
         * @brief Run @p compute once per distinct block and hand each component its block's result.
         *
         * Components carrying `no_wave_speed` are skipped and keep the value-initialised entry.
         * `compute(block)` is called at most once per distinct id, so a model that declares one
         * block pays exactly what it paid before blocks existed.
         */
        template <size_t n_components, typename Result, typename ComputeFUN>
        std::array<Result, n_components> per_block(const std::array<int, n_components> &blocks,
                                                   const ComputeFUN &compute)
        {
          std::array<Result, n_components> per_component{};
          for (size_t i = 0; i < n_components; ++i) {
            if (blocks[i] < 0) continue;
            size_t source = i;
            for (size_t j = 0; j < i; ++j)
              if (blocks[j] == blocks[i]) {
                source = j;
                break;
              }
            per_component[i] = source == i ? compute(blocks[i]) : per_component[source];
          }
          return per_component;
        }

        /**
         * @brief Result struct for compute_kt_flux_and_speeds.
         */
        template <int dim, typename NumberType, size_t n_components> struct KTFluxData {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F_plus;
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F_minus;
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> a_half;
        };

        template <typename WaveSpeedStrategy, typename Model, typename NumberType, int dim, size_t n_components,
                  typename ExtractorArray, typename VariableVector>
        KTFluxData<dim, NumberType, n_components> compute_kt_flux_and_speeds(
            const std::array<NumberType, n_components> &u_plus, const std::array<NumberType, n_components> &u_minus,
            const GradientType<dim, NumberType, n_components> &grad_u_plus,
            const GradientType<dim, NumberType, n_components> &grad_u_minus, const dealii::Point<dim> &x_q,
            const double cell_width_plus, const double cell_width_minus, const ExtractorArray &extractors,
            const VariableVector &variables, const Model &model)
        {
          using ADNumberType = autodiff::Real<1, NumberType>;

          KTFluxData<dim, NumberType, n_components> result{};

          std::array<ADNumberType, n_components> u_plus_AD{}, u_minus_AD{};
          GradientType<dim, ADNumberType, n_components> grad_u_plus_AD{}, grad_u_minus_AD{};
          for (size_t i = 0; i < n_components; ++i) {
            u_plus_AD[i] = ADNumberType(u_plus[i]);
            u_minus_AD[i] = ADNumberType(u_minus[i]);
            for (size_t d = 0; d < dim; ++d) {
              grad_u_plus_AD[i][d] = ADNumberType(grad_u_plus[i][d]);
              grad_u_minus_AD[i][d] = ADNumberType(grad_u_minus[i][d]);
            }
          }

          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> F_AD_plus{}, F_AD_minus{};

          std::array<JacobianMatrix<NumberType, n_components>, dim> J_plus{}, J_minus{};

          for (size_t j = 0; j < n_components; ++j) {
            seed(u_plus_AD[j]);
            seed(u_minus_AD[j]);

            F_AD_plus = {};
            F_AD_minus = {};
            model.flux(F_AD_plus, x_q, flux_tie(u_plus_AD, grad_u_plus_AD, extractors, variables, cell_width_plus));
            model.flux(F_AD_minus, x_q, flux_tie(u_minus_AD, grad_u_minus_AD, extractors, variables, cell_width_minus));

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

          // One speed per block, from the flux jacobian restricted to that block, and a component
          // marked no_wave_speed gets none at all -- its numerical flux is then identically zero, so
          // the reconstruction, and its slope limiter, never reaches its row. See
          // AbstractModel::wave_speed_blocks for why a shared speed is wrong for a system that
          // mixes a conservation law with constraints.
          std::array<int, n_components> blocks;
          model.wave_speed_blocks(blocks);

          const auto a = per_block<n_components, std::array<NumberType, dim>>(blocks, [&](const int block) {
            return WaveSpeedStrategy::template compute_speeds<NumberType, dim, n_components>(
                restrict_jacobian_to_block<NumberType, dim, n_components>(J_plus, blocks, block),
                restrict_jacobian_to_block<NumberType, dim, n_components>(J_minus, blocks, block));
          });

          for (size_t d = 0; d < dim; ++d)
            for (size_t component = 0; component < n_components; ++component)
              result.a_half[component][d] = a[component][d];

          // A model that excludes a component and then writes a flux for it keeps the physical
          // average (F^+ + F^-)/2 but loses the dissipation that stabilises it -- a central flux on
          // a hyperbolic equation, which oscillates rather than failing outright. Catch the
          // contradiction here instead of leaving it in the solution.
          for (size_t i = 0; i < n_components; ++i)
            if (blocks[i] < 0)
              Assert(result.F_plus[i].norm() == NumberType(0) && result.F_minus[i].norm() == NumberType(0),
                     dealii::ExcMessage("A component marked no_wave_speed by wave_speed_blocks() wrote a flux."));

          return result;
        }

        template <typename WaveSpeedStrategy, typename Model, typename NumberType, int dim, size_t n_components,
                  typename ExtractorArray, typename VariableVector>
        KTFluxData<dim, NumberType, n_components> compute_kt_flux_and_speeds(
            const std::array<NumberType, n_components> &u_plus, const std::array<NumberType, n_components> &u_minus,
            const dealii::Point<dim> &x_q, const double cell_width_plus, const double cell_width_minus,
            const ExtractorArray &extractors, const VariableVector &variables, const Model &model)
        {
          const GradientType<dim, NumberType, n_components> zero_grad{};
          return compute_kt_flux_and_speeds<WaveSpeedStrategy>(u_plus, u_minus, zero_grad, zero_grad, x_q,
                                                               cell_width_plus, cell_width_minus, extractors, variables,
                                                               model);
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
        template <int dim, typename NumberType, size_t n_components> struct KTNumFluxJacobianData {
          std::array<SimpleMatrix<dealii::Tensor<1, dim, NumberType>, n_components>, 2> u{};
          std::array<SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<1, dim, NumberType>>, n_components>, 2> grad{};
        };

        template <typename WaveSpeedStrategy, typename Model, typename NumberType, int dim, size_t n_components,
                  typename ExtractorArray, typename VariableVector>
        KTNumFluxJacobianData<dim, NumberType, n_components> compute_kt_numflux_jacobian(
            const std::array<NumberType, n_components> &u_plus, const std::array<NumberType, n_components> &u_minus,
            const GradientType<dim, NumberType, n_components> &grad_u_plus,
            const GradientType<dim, NumberType, n_components> &grad_u_minus, const dealii::Point<dim> &x_q,
            const double cell_width_plus, const double cell_width_minus, const ExtractorArray &extractors,
            const VariableVector &variables, const Model &model)
        {
          // 1. Compute the physical flux, its value/gradient Jacobians, and the second derivatives needed for da.
          const auto plus = compute_flux_derivatives_ad<Model, NumberType, dim, n_components>(
              u_plus, grad_u_plus, x_q, cell_width_plus, extractors, variables, model);
          const auto minus = compute_flux_derivatives_ad<Model, NumberType, dim, n_components>(
              u_minus, grad_u_minus, x_q, cell_width_minus, extractors, variables, model);

          // 2. One wave speed per block, and its derivative, from the flux jacobian restricted to
          // that block. This has to mirror compute_kt_flux_and_speeds exactly -- a residual and a
          // jacobian that disagree about the dissipation are an inconsistent linearisation, which is
          // worse than either choice on its own. See AbstractModel::wave_speed_blocks.
          std::array<int, n_components> blocks;
          model.wave_speed_blocks(blocks);

          const auto a = per_block<n_components, std::array<NumberType, dim>>(blocks, [&](const int block) {
            return WaveSpeedStrategy::template compute_speeds<NumberType, dim, n_components>(
                restrict_jacobian_to_block<NumberType, dim, n_components>(plus.J, blocks, block),
                restrict_jacobian_to_block<NumberType, dim, n_components>(minus.J, blocks, block));
          });

          // 3. Differentiate the selected physical wave speed with the AD flux Hessian.
          using SpeedDerivatives = std::pair<std::array<std::array<NumberType, n_components>, dim>,
                                             std::array<std::array<NumberType, n_components>, dim>>;
          const auto da = per_block<n_components, SpeedDerivatives>(blocks, [&](const int block) {
            return WaveSpeedStrategy::template compute_selected_speed_derivatives<NumberType, dim, n_components>(
                restrict_jacobian_to_block<NumberType, dim, n_components>(plus.J, blocks, block),
                restrict_jacobian_to_block<NumberType, dim, n_components>(minus.J, blocks, block),
                restrict_hessian_to_block<NumberType, dim, n_components>(plus.H, blocks, block),
                restrict_hessian_to_block<NumberType, dim, n_components>(minus.H, blocks, block));
          });

          // 4. Assemble j_numflux
          KTNumFluxJacobianData<dim, NumberType, n_components> j_numflux{};

          for (size_t d = 0; d < dim; ++d) {
            for (size_t i = 0; i < n_components; ++i) {
              const NumberType du_i = u_plus[i] - u_minus[i];
              const auto &[da_plus, da_minus] = da[i];
              for (size_t c = 0; c < n_components; ++c) {
                const NumberType delta_ic = (i == c) ? NumberType(1) : NumberType(0);

                // dH_i^d / du_minus_c
                j_numflux.u[0](i, c)[d] = NumberType(0.5) * minus.J[d][i][c] + NumberType(0.5) * a[i][d] * delta_ic -
                                          NumberType(0.5) * du_i * da_minus[d][c];

                // dH_i^d / du_plus_c
                j_numflux.u[1](i, c)[d] = NumberType(0.5) * plus.J[d][i][c] - NumberType(0.5) * a[i][d] * delta_ic -
                                          NumberType(0.5) * du_i * da_plus[d][c];
              }
            }
          }

          for (size_t d_in = 0; d_in < dim; ++d_in) {
            const auto da_grad = per_block<n_components, SpeedDerivatives>(blocks, [&](const int block) {
              return WaveSpeedStrategy::template compute_selected_speed_derivatives<NumberType, dim, n_components>(
                  restrict_jacobian_to_block<NumberType, dim, n_components>(plus.J, blocks, block),
                  restrict_jacobian_to_block<NumberType, dim, n_components>(minus.J, blocks, block),
                  restrict_hessian_to_block<NumberType, dim, n_components>(plus.mixed_H[d_in], blocks, block),
                  restrict_hessian_to_block<NumberType, dim, n_components>(minus.mixed_H[d_in], blocks, block));
            });
            for (size_t d_out = 0; d_out < dim; ++d_out)
              for (size_t i = 0; i < n_components; ++i) {
                const NumberType du_i = u_plus[i] - u_minus[i];
                const auto &[da_grad_plus, da_grad_minus] = da_grad[i];
                for (size_t c = 0; c < n_components; ++c) {
                  j_numflux.grad[0](i, c)[d_out][d_in] = NumberType(0.5) * minus.grad_J[i][c][d_out][d_in] -
                                                         NumberType(0.5) * du_i * da_grad_minus[d_out][c];
                  j_numflux.grad[1](i, c)[d_out][d_in] = NumberType(0.5) * plus.grad_J[i][c][d_out][d_in] -
                                                         NumberType(0.5) * du_i * da_grad_plus[d_out][c];
                }
              }
          }

          return j_numflux;
        }

      } // namespace internal

      namespace internal
      {
        template <typename Model, typename NumberType, int dim, size_t n_components, typename ExtractorArray,
                  typename VariableVector>
        std::array<dealii::Tensor<1, dim, NumberType>, n_components> compute_diffusion_flux(
            const std::array<NumberType, n_components> &u_minus, const std::array<NumberType, n_components> &u_plus,
            const GradientType<dim, NumberType, n_components> &grad_u_minus,
            const GradientType<dim, NumberType, n_components> &grad_u_plus,
            const ThirdDerivativeType<dim, NumberType, n_components> &third_derivatives_minus,
            const ThirdDerivativeType<dim, NumberType, n_components> &third_derivatives_plus,
            const dealii::Point<dim> &x_q, const double cell_width_minus, const double cell_width_plus,
            const ExtractorArray &extractors, const VariableVector &variables, const Model &model)
        {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D_minus{};
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D_plus{};
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D{};
          model.diffusion_flux(D_minus, x_q,
                               diffusion_flux_tie(u_minus, grad_u_minus, third_derivatives_minus, extractors, variables,
                                                  cell_width_minus));
          model.diffusion_flux(
              D_plus, x_q,
              diffusion_flux_tie(u_plus, grad_u_plus, third_derivatives_plus, extractors, variables, cell_width_plus));
          for (size_t c = 0; c < n_components; ++c)
            D[c] = NumberType(0.5) * (D_minus[c] + D_plus[c]);
          return D;
        }

        template <int dim, typename NumberType, size_t n_components> struct DiffusionFluxJacobianData {
          std::array<SimpleMatrix<dealii::Tensor<1, dim, NumberType>, n_components>, 2> u{};
          std::array<SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<1, dim, NumberType>>, n_components>, 2> grad{};
          std::array<SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<3, dim, NumberType>>, n_components>, 2>
              third_derivatives{};
        };

        template <typename Model, typename NumberType, int dim, size_t n_components, typename ExtractorArray,
                  typename VariableVector>
        DiffusionFluxJacobianData<dim, NumberType, n_components> compute_diffusion_flux_jacobian(
            const std::array<NumberType, n_components> &u_minus, const std::array<NumberType, n_components> &u_plus,
            const GradientType<dim, NumberType, n_components> &grad_u_minus,
            const GradientType<dim, NumberType, n_components> &grad_u_plus,
            const ThirdDerivativeType<dim, NumberType, n_components> &third_derivatives_minus,
            const ThirdDerivativeType<dim, NumberType, n_components> &third_derivatives_plus,
            const dealii::Point<dim> &x_q, const double cell_width_minus, const double cell_width_plus,
            const ExtractorArray &extractors, const VariableVector &variables, const Model &model)
        {
          using ADNumberType = autodiff::Real<1, NumberType>;

          DiffusionFluxJacobianData<dim, NumberType, n_components> result{};

          std::array<ADNumberType, n_components> u_minus_AD{};
          std::array<ADNumberType, n_components> u_plus_AD{};
          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> grad_u_minus_AD{};
          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> grad_u_plus_AD{};
          ThirdDerivativeType<dim, ADNumberType, n_components> third_derivatives_minus_AD{};
          ThirdDerivativeType<dim, ADNumberType, n_components> third_derivatives_plus_AD{};
          for (size_t c = 0; c < n_components; ++c) {
            u_minus_AD[c] = ADNumberType(u_minus[c]);
            u_plus_AD[c] = ADNumberType(u_plus[c]);
            for (size_t d = 0; d < dim; ++d) {
              grad_u_minus_AD[c][d] = ADNumberType(grad_u_minus[c][d]);
              grad_u_plus_AD[c][d] = ADNumberType(grad_u_plus[c][d]);
            }
            for (size_t d0 = 0; d0 < dim; ++d0)
              for (size_t d1 = 0; d1 < dim; ++d1)
                for (size_t d2 = 0; d2 < dim; ++d2) {
                  third_derivatives_minus_AD[c][d0][d1][d2] = ADNumberType(third_derivatives_minus[c][d0][d1][d2]);
                  third_derivatives_plus_AD[c][d0][d1][d2] = ADNumberType(third_derivatives_plus[c][d0][d1][d2]);
                }
          }

          std::array<dealii::Tensor<1, dim, ADNumberType>, n_components> D_AD{};

          for (size_t c = 0; c < n_components; ++c) {
            seed(u_minus_AD[c]);
            D_AD = {};
            model.diffusion_flux(D_AD, x_q,
                                 diffusion_flux_tie(u_minus_AD, grad_u_minus_AD, third_derivatives_minus_AD, extractors,
                                                    variables, cell_width_minus));
            for (size_t i = 0; i < n_components; ++i)
              for (size_t d = 0; d < dim; ++d)
                result.u[0](i, c)[d] = NumberType(0.5) * derivative(D_AD[i][d]);
            unseed(u_minus_AD[c]);

            seed(u_plus_AD[c]);
            D_AD = {};
            model.diffusion_flux(D_AD, x_q,
                                 diffusion_flux_tie(u_plus_AD, grad_u_plus_AD, third_derivatives_plus_AD, extractors,
                                                    variables, cell_width_plus));
            for (size_t i = 0; i < n_components; ++i)
              for (size_t d = 0; d < dim; ++d)
                result.u[1](i, c)[d] = NumberType(0.5) * derivative(D_AD[i][d]);
            unseed(u_plus_AD[c]);

            for (size_t d_in = 0; d_in < dim; ++d_in) {
              seed(grad_u_minus_AD[c][d_in]);
              D_AD = {};
              model.diffusion_flux(D_AD, x_q,
                                   diffusion_flux_tie(u_minus_AD, grad_u_minus_AD, third_derivatives_minus_AD,
                                                      extractors, variables, cell_width_minus));
              for (size_t i = 0; i < n_components; ++i)
                for (size_t d_out = 0; d_out < dim; ++d_out)
                  result.grad[0](i, c)[d_out][d_in] = NumberType(0.5) * derivative(D_AD[i][d_out]);
              unseed(grad_u_minus_AD[c][d_in]);

              seed(grad_u_plus_AD[c][d_in]);
              D_AD = {};
              model.diffusion_flux(D_AD, x_q,
                                   diffusion_flux_tie(u_plus_AD, grad_u_plus_AD, third_derivatives_plus_AD, extractors,
                                                      variables, cell_width_plus));
              for (size_t i = 0; i < n_components; ++i)
                for (size_t d_out = 0; d_out < dim; ++d_out)
                  result.grad[1](i, c)[d_out][d_in] = NumberType(0.5) * derivative(D_AD[i][d_out]);
              unseed(grad_u_plus_AD[c][d_in]);
            }

            for (size_t d0 = 0; d0 < dim; ++d0)
              for (size_t d1 = 0; d1 < dim; ++d1)
                for (size_t d2 = 0; d2 < dim; ++d2) {
                  seed(third_derivatives_minus_AD[c][d0][d1][d2]);
                  D_AD = {};
                  model.diffusion_flux(D_AD, x_q,
                                       diffusion_flux_tie(u_minus_AD, grad_u_minus_AD, third_derivatives_minus_AD,
                                                          extractors, variables, cell_width_minus));
                  for (size_t i = 0; i < n_components; ++i)
                    for (size_t d_out = 0; d_out < dim; ++d_out)
                      result.third_derivatives[0](i, c)[d_out][d0][d1][d2] =
                          NumberType(0.5) * derivative(D_AD[i][d_out]);
                  unseed(third_derivatives_minus_AD[c][d0][d1][d2]);

                  seed(third_derivatives_plus_AD[c][d0][d1][d2]);
                  D_AD = {};
                  model.diffusion_flux(D_AD, x_q,
                                       diffusion_flux_tie(u_plus_AD, grad_u_plus_AD, third_derivatives_plus_AD,
                                                          extractors, variables, cell_width_plus));
                  for (size_t i = 0; i < n_components; ++i)
                    for (size_t d_out = 0; d_out < dim; ++d_out)
                      result.third_derivatives[1](i, c)[d_out][d0][d1][d2] =
                          NumberType(0.5) * derivative(D_AD[i][d_out]);
                  unseed(third_derivatives_plus_AD[c][d0][d1][d2]);
                }
          }

          return result;
        }

      } // namespace internal

      // Model_ keeps its second place and merely gains a default, rather than moving to the end.
      // Moving it would let an application that names a Reconstructor drop it, but it would also
      // force every *model-generic* discretization -- one Discretization built from a bare
      // ComponentDescriptor and shared across several models, which is how most of the regression
      // tests are written -- to spell out all four preceding arguments to reach the model.
      template <typename Discretization_,
                typename Model_ = typename DiFfRG::internal::assembler_model_of<Discretization_>::type,
                def::HasReconstructor Reconstructor_ =
                    def::TVDReconstructor<Discretization_::dim, def::MinModLimiter, double>,
                def::HasWaveSpeed WaveSpeedStrategy_ = MaxEigenvalueWaveSpeed,
                def::HasReconstructor JacobianReconstructor_ = Reconstructor_>
        requires MeshIsRectangular<typename Discretization_::Mesh>
      class Assembler : public AbstractAssembler<typename Discretization_::VectorType,
                                                 typename Discretization_::SparseMatrixType, Discretization_::dim>
      {
      protected:
        // Placeholder for the "extractors" slot of e_tie() when the extractors are being computed and so cannot
        // be passed to themselves. Same trick as DiFfRG::FEMAssembler.
        constexpr static int nothing = 0;

        /**
         * @brief The named tuple handed to model.source(), the FV counterpart of CG's fe_tie().
         *
         * "fe_derivatives" is the Reconstructor's gradient at the quadrature point, not an FE derivative --
         * at the default single quadrature point (the cell centre) that is the scheme's own limited slope.
         *
         * There is deliberately no "fe_hessians" slot: the only curvature the 2*dim stencil can produce is
         * the unlimited quadratic fit of reconstruct_readout_solution(), which is not fit for a per-cell
         * residual path. Omitting the slot makes a model that reads it fail to compile rather than silently
         * receive zeros.
         */
        template <typename... T> auto fv_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>,
                             StringSet<"fe_functions", "fe_derivatives", "extractors", "variables", "cell_width">>(
              std::tie(t...));
        }

        template <typename... T> static constexpr auto v_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, StringSet<"variables", "extractors">>(std::tie(t...));
        }

        template <typename... T> static constexpr auto e_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>,
                             StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors", "variables",
                                       "potential", "potential_gradient", "potential_hessian">>(std::tie(t...));
        }

      public:
        using Discretization = Discretization_;
        using Model = Model_;
        using Reconstructor = Reconstructor_;
        using WaveSpeedStrategy = WaveSpeedStrategy_;
        using JacobianReconstructor = JacobianReconstructor_;
        using NumberType = typename Discretization::NumberType;
        using VectorType = typename Discretization::VectorType;
        using SparseMatrixType = typename Discretization::SparseMatrixType;

        using Components = typename Discretization::Components;
        static constexpr uint dim = Discretization::dim;
        static_assert(Reconstructor::dim == dim, "Reconstructor dimension must match the discretization dimension.");
        static_assert(JacobianReconstructor::dim == dim,
                      "JacobianReconstructor dimension must match the discretization dimension.");
        static constexpr uint n_components = Components::count_fe_functions(0);
        static constexpr uint n_faces = GeometryInfo<dim>::faces_per_cell;
        // using CacheData = internal::Cache_Data<NumberType, dim, n_components>;
        using GradientType = internal::GradientType<dim, NumberType, n_components>;
        using ThirdDerivativeType = internal::ThirdDerivativeType<dim, NumberType, n_components>;
        using Iterator = typename DoFHandler<Discretization::dim>::active_cell_iterator;
        using Point = dealii::Point<dim>;
        using Scratch = internal::ScratchData<dim, NumberType, n_components>;
        struct FaceJacobianDependencyCacheEntry {
          std::vector<types::global_dof_index> to_dofs;
          std::vector<types::global_dof_index> from_dofs;
        };
        struct FaceReconstructionDescriptor {
          unsigned int cell_index = 0;
          unsigned int face_index = 0;
          bool boundary = false;
          std::optional<unsigned int> neighbor_index;
          std::optional<unsigned int> neighbor_face_index;
          Point face_center;
        };
        struct CellTopologyCacheEntry {
          internal::CellStencilTopologyData<dim, n_components> stencil;
          std::array<internal::BoundaryReconstructionStencilTopologyData<dim, n_components>, n_faces>
              boundary_stencils{};
          std::array<FaceJacobianDependencyCacheEntry, n_faces> face_jacobian_dependencies{};
          FaceJacobianDependencyCacheEntry source_jacobian_dependencies{};
          std::vector<Point> quadrature_points;
          std::vector<NumberType> jxw;
          /// Local grid spacing, handed to the model through the flux and source ties.
          /// @see DiFfRG::internal::cell_width, DiFfRG::FV::KurganovTadmor::internal::flux_tie.
          double cell_width = 0.;
        };
        [[deprecated("Pass output.log_port() or an intentional LogPort{}")]] Assembler(
            Discretization &discretization, Model &model,
            DiFfRG::internal::LegacyDefaultLogPortArgument<Discretization, ConfigTree> config)
            : Assembler(discretization, model, config.value(),
                        DiFfRG::internal::legacy_default_log_port<Discretization>())
        {
        }

        Assembler(Discretization &discretization, Model &model, const ConfigTree &config, LogPort log_port)
            : discretization(discretization), model(model), log_port(std::move(log_port)),
              dof_handler(discretization.get_dof_handler()), mapping(discretization.get_mapping()),
              triangulation(discretization.get_triangulation()), fe(discretization.get_fe()),
              schedule_overrides(AssemblyScheduleOverrides::from_config(config)),
              EoM_cell(*(dof_handler.active_cell_iterators().end())),
              old_EoM_cell(*(dof_handler.active_cell_iterators().end())),
              EoM_config(DiFfRG::internal::resolve_eom_config(dof_handler, Config::EoMConfig(config))),
              quadrature(1 + config.get_uint("/discretization/overintegration", 0)),
              quadrature_face(1 + config.get_uint("/discretization/overintegration", 0)),
              diagnose_flux_conditioning(config.get_bool("/discretization/diagnose_flux_conditioning", false))
        {
          AssertThrow(fe.dofs_per_cell == n_components,
                      ExcMessage("FV Kurganov-Tadmor assembler expects one dof per component."));
          for (uint i = 0; i < n_components; ++i)
            local_component_of_dof[i] = fe.system_to_component_index(i).first;

          reinit();
        }

        virtual void reinit_vector(VectorType &vec) const override
        {
          reinit_la_vector(vec, discretization.get_locally_owned_dofs(), discretization.get_communicator());
        }

        virtual void reinit_matrix(SparseMatrixType &matrix) const override
        {
          reinit_la_matrix(matrix, get_sparsity_pattern_jacobian(), discretization.get_locally_owned_dofs(),
                           discretization.get_communicator());
        }

        virtual MPI_Comm get_communicator() const override { return discretization.get_communicator(); }

        virtual void reinit_solution_view(SolutionView<VectorType> &view) const override
        {
          view.reinit(discretization.get_locally_owned_dofs(), discretization.get_locally_relevant_dofs(),
                      discretization.get_communicator());
        }

        virtual IndexSet get_differential_indices() const override
        {
          ComponentMask component_mask(model.template differential_components<dim>());
          // See FEMAssembler::get_differential_indices for why this is restricted to owned rows.
          return restrict_to_owned<VectorType>(DoFTools::extract_dofs(dof_handler, component_mask),
                                               discretization.get_locally_owned_dofs());
        }

        virtual void attach_data_output(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                                        const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                        const VectorType &residual = VectorType()) override
        {
          const auto fe_function_names = Components::FEFunction_Descriptor::get_names_vector();
          std::vector<std::string> fe_function_names_residual;
          for (const auto &name : fe_function_names)
            fe_function_names_residual.push_back(name + "_residual");
          std::vector<std::string> fe_function_names_dot;
          for (const auto &name : fe_function_names)
            fe_function_names_dot.push_back(name + "_dot");

          auto fe_out = data_out.fields();
          fe_out.attach(dof_handler, solution, fe_function_names);
          if (dt_solution.size() > 0) fe_out.attach(dof_handler, dt_solution, fe_function_names_dot);
          if (residual.size() > 0) fe_out.attach(dof_handler, residual, fe_function_names_residual);

          readouts(data_out, solution, variables);
        }

        virtual void reinit() override
        {
          Timer timer;

          const auto metadata = DiFfRG::internal::build_affine_constraint_metadata<Components, dim>(discretization);
          const AffineConstraintContext<Components, dim> context(metadata);

          auto &constraints = discretization.get_constraints();
          constraints.clear();
          DoFTools::make_hanging_node_constraints(dof_handler, constraints);
          DiFfRG::internal::apply_model_affine_constraints(model, constraints, context);
          constraints.close();

          // Mass sparsity pattern
          {
            DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
            DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_dofs = */ true);
            finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern_mass, discretization.get_locally_owned_dofs(),
                                                   discretization.get_locally_relevant_dofs(),
                                                   discretization.get_communicator());
            reinit_la_matrix(mass_matrix, sparsity_pattern_mass, discretization.get_locally_owned_dofs(),
                             discretization.get_communicator());
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

          // Hoisted out of probe_diffusion_flux_conditioning(): GridTools::diameter is COLLECTIVE
          // on a partitioned triangulation, and that function is entered conditionally
          // (diagnose_flux_conditioning, and only on the first residual assembly). A collective
          // behind a condition is a hang waiting for the condition to stop agreeing across ranks.
          // Here every rank arrives unconditionally, and the value cannot go stale because KT
          // refuses to run under mesh adaptivity.
          domain_diameter = GridTools::diameter(triangulation);

          update_assembly_schedules();

          rebuild_cell_topology_cache();
          rebuild_face_reconstruction_descriptors();
          residual_reconstruction_cache.topology_initialized = false;
          jacobian_reconstruction_cache.topology_initialized = false;
          build_cached_jacobian_sparsity(sparsity_pattern_jacobian);

          timings_reinit.push_back(timer.wall_time());
        }

        virtual void set_time(double t) override { model.set_time(t); }

        virtual const get_type::SparsityPattern<SparseMatrixType> &get_sparsity_pattern_jacobian() const override
        {
          return sparsity_pattern_jacobian;
        }
        virtual const SparseMatrixType &get_mass_matrix() const override { return mass_matrix; }

        virtual void residual_variables(VectorType &residual, const VectorType &variables,
                                        const VectorType &spatial_solution) override
        {
          // Mirrors DiFfRG::FEMAssembler::residual_variables: run the extractors at the EoM first, then hand
          // dt_variables the v_tie(variables, extractors) tuple. Without the extractors a model whose
          // Variables flow depends on its FE solution cannot be assembled.
          std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
          if constexpr (Components::count_extractors() > 0)
            extract(__extracted_data, spatial_solution, variables, true, false, false);
          const auto &extracted_data = __extracted_data;
          model.dt_variables(residual, v_tie(variables, extracted_data));
        };

        virtual void jacobian_variables([[maybe_unused]] FullMatrix<NumberType> &jacobian,
                                        [[maybe_unused]] const VectorType &variables, const VectorType &) override
        {
          static_assert(true, "jacobian_variables is not implemented for this model");
          // model.template jacobian_variables<0>(jacobian, fv_tie(variables));
        };

        struct ReadoutSolution {
          std::array<NumberType, n_components> values{};
          GradientType gradients{};
          std::array<Tensor<2, dim, NumberType>, n_components> hessians{};
        };

        void readouts(OutputFrame<dim, VectorType> &data_out, const VectorType &solution_global,
                      const VectorType &variables) const
        {
          auto raw_potential = reconstruct_raw_potential(
              solution_global, dof_handler, mapping,
              [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, EoM_config,
              &potential_cache);
          auto helper = [&](auto &&...args) {
            if constexpr (sizeof...(args) == 3) {
              auto &&[id, EoMfun, outputter] = std::forward_as_tuple(std::forward<decltype(args)>(args)...);
              data_out.register_readout(id);
              auto EoM_cell = this->EoM_cell;
              auto EoM_result = get_EoM_point_with_potential(
                  EoM_cell, solution_global, dof_handler, mapping, EoMfun,
                  [](const auto &point, const auto &) { return point; }, EoM_config, EoM_minimum_guess,
                  &potential_cache);
              if (EoM_result.potential) EoM_minimum_guess = EoM_result.potential->minimum;
              const auto EoM = EoM_result.point;
              this->EoM_cell = EoM_cell;

              auto solution = reconstruct_readout_solution(EoM_cell, solution_global, EoM);
              const auto potential = evaluate_raw_potential(raw_potential, mapping, EoM);

              // The readout is always at this readout's EoM. The extractors may not be: a model that
              // defines extractor_point reads them elsewhere, and dt_variables must see the same
              // values here as it does during assembly.
              std::array<NumberType, Components::count_extractors()> extracted_data{{}};
              if constexpr (Components::count_extractors() > 0) {
                const auto [x, cell] = resolve_extractor_point(EoM, EoM_cell, solution_global);
                const auto extractor_solution =
                    reconstruct_readout_solution(cell, solution_global, x, /*with_hessians=*/true);
                const auto extractor_potential = evaluate_raw_potential(raw_potential, mapping, x);
                model.extract(extracted_data, x,
                              e_tie(extractor_solution.values, extractor_solution.gradients,
                                    extractor_solution.hessians, nothing, variables, extractor_potential.value,
                                    extractor_potential.gradient, extractor_potential.hessian));
              }
              outputter(data_out, EoM,
                        e_tie(solution.values, solution.gradients, solution.hessians, extracted_data, variables,
                              potential.value, potential.gradient, potential.hessian));
              data_out.attach_eom_potential(std::move(EoM_result));
            } else {
              DiFfRG::internal::validate_readout_helper_arity<decltype(args)...>();
            }
          };
          model.readouts_multiple(helper, data_out);
          data_out.attach_raw_potential(std::move(raw_potential));
        }

        /**
         * @param with_hessians Also fill ReadoutSolution::hessians. Off by default and intended ONLY for the
         * single EoM-point evaluation in extract(): it is a plain unlimited difference, not part of the
         * scheme, and must not be wired into the flux path.
         */
        /**
         * @brief Where the model wants its extractors evaluated, and the cell holding that point.
         *
         * The EoM itself when the model does not define `extractor_point` -- and then this costs
         * nothing, because building the SolutionSample is inside the `if constexpr`.
         *
         * Values are read straight off the dofs -- with one dof per cell they are the cell averages
         * already -- and gradients are recovered by central differences afterwards. Deliberately not
         * the Reconstructor's limited slopes: reconstructing per cell means a stencil fill over the
         * whole mesh on every residual evaluation, which is serial work that dominates the assembly
         * (it cost a factor of four here). The limited slope stays where it belongs, in the flux
         * path. The FE path in make_solution_sample() is no use for either, since DG0 shape
         * functions have zero gradient.
         */
        std::pair<Point, Iterator> resolve_extractor_point(const Point &EoM_point, const Iterator &EoM_cell_,
                                                           [[maybe_unused]] const VectorType &solution_global) const
        {
          if constexpr (HasExtractorPoint<Model, dim, NumberType>) {
            // One dof per cell, so the value at the cell centre is that dof -- no reconstruction
            // needed, and none wanted: this runs on every residual evaluation, and a per-cell
            // stencil fill over the whole mesh is serial work that would dominate the assembly.
            std::vector<types::global_dof_index> cell_dofs(n_components);
            auto sample = make_solution_sample<dim, NumberType>(dof_handler, mapping, n_components,
                                                                [&](const Iterator &cell, const Point &,
                                                                    std::vector<NumberType> &values,
                                                                    std::vector<Tensor<1, dim, NumberType>> &) {
                                                                  cell->get_dof_indices(cell_dofs);
                                                                  for (uint c = 0; c < n_components; ++c)
                                                                    values[c] = solution_global[cell_dofs[c]];
                                                                });
            sample.compute_central_difference_gradients();
            const auto point = model.template extractor_point<dim, NumberType>(EoM_point, sample);
            if (point == EoM_point) return {EoM_point, EoM_cell_};
            return {point, GridTools::find_active_cell_around_point(dof_handler, point)};
          } else
            return {EoM_point, EoM_cell_};
        }

        ReadoutSolution reconstruct_readout_solution(const Iterator &cell, const VectorType &solution_global,
                                                     const Point &x, bool with_hessians = false) const
        {
          CellStencilData stencil;
          fill_cell_stencil(cell, solution_global, stencil);

          const auto gradients = Reconstructor::template compute_gradient<n_components>(
              stencil.cell.x, stencil.cell.u, stencil.neighbors.x, stencil.neighbors.u);

          ReadoutSolution solution;
          solution.values = internal::reconstruct_u(stencil.cell.u, stencil.cell.x, x, gradients);
          solution.gradients = Reconstructor::template compute_gradient_at_point<n_components>(
              stencil.cell.x, x, stencil.cell.u, stencil.neighbors.x, stencil.neighbors.u);

          if (with_hessians) {
            // Second derivative from the quadratic through the same three cell averages the gradient uses:
            // with s measured from the cell centre and u(s) = u_C + a s + b s^2 through (dx_1, u_1), (0, u_C),
            // (dx_2, u_2), one gets u'' = 2b = 2 (du_2 - du_1) / (dx_2 - dx_1), where du_i are exactly the
            // one-sided slopes of compute_gradient. Being a quadratic fit, u'' is constant over the cell, so
            // it does not matter that it is not evaluated at `x` itself.
            //
            // Deliberately UNLIMITED: the limiter exists to keep the reconstructed *slope* monotone for the
            // scheme, and applying it to a curvature would bias it toward zero exactly where the potential is
            // most curved. The price is that this is the noisiest quantity available near a steep front --
            // acceptable because nothing in the flux consumes it.
            //
            // Only the diagonal d^2/dx_d^2 entries are filled: the 2*dim stencil has no corner neighbours, so
            // mixed derivatives are not available (harmless for the 1D field-space this assembler targets).
            for (uint c = 0; c < n_components; ++c)
              for (int d = 0, i_n_1 = 0, i_n_2 = 1; d < dim; ++d, i_n_1 += 2, i_n_2 += 2) {
                const auto dx_1 = stencil.neighbors.x[i_n_1][d] - stencil.cell.x[d];
                const auto dx_2 = stencil.neighbors.x[i_n_2][d] - stencil.cell.x[d];
                const auto du_1 = (stencil.neighbors.u[i_n_1][c] - stencil.cell.u[c]) / dx_1;
                const auto du_2 = (stencil.neighbors.u[i_n_2][c] - stencil.cell.u[c]) / dx_2;
                solution.hessians[c][d][d] = NumberType(2.) * (du_2 - du_1) / (dx_2 - dx_1);
              }
          }
          return solution;
        }

        /**
         * @brief Evaluate the model's extractors at the EoM point.
         *
         * This is the FV counterpart of DiFfRG::FEMAssembler::extract, and exists for the same reason: it is the
         * only bridge by which a model's FE (field-space) solution reaches its Variables. Models that couple the
         * two -- e.g. an effective potential whose flux depends on momentum-dependent dressings which in turn flow
         * with the potential's derivatives at the EoM -- cannot be assembled without it.
         *
         * The reconstruction is the same one readouts() uses: the EoM point is found from the cell-averaged
         * solution, then values and gradients are reconstructed there by the Reconstructor. Hessians are
         * additionally reconstructed here (and ONLY here -- see reconstruct_readout_solution): one extra
         * three-point difference at a single point per step is free, whereas doing it per cell in the flux
         * path would be neither cheap nor meaningful.
         *
         * @param data           Output: the extractor values.
         * @param search_EoM     Re-locate the EoM point instead of reusing the cached one.
         * @param set_EoM        Store the located point/cell as the new cache.
         * @param postprocess    Apply the model's EoM_postprocess to the located point.
         */
        /**
         * @brief The raw potential for the extractors, or an inert placeholder if the model does not read it.
         *
         * Reconstructing it is a direct solve over the whole mesh, and extract() runs on every residual and
         * every jacobian -- so a model that never touches the potential slots should say so and skip it.
         */
        auto extractor_raw_potential(const VectorType &solution_global) const
        {
          if constexpr (Model::extract_uses_potential)
            return reconstruct_raw_potential(
                solution_global, dof_handler, mapping,
                [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, EoM_config,
                &potential_cache);
          else
            return UnusedPotential{};
        }

        void extract(std::array<NumberType, Components::count_extractors()> &data, const VectorType &solution_global,
                     const VectorType &variables, bool search_EoM, bool set_EoM, bool postprocess) const
        {
          auto EoM = this->EoM;
          auto EoM_cell = this->EoM_cell;
          if (search_EoM || EoM_cell == *(dof_handler.active_cell_iterators().end())) {
            auto EoM_result = get_EoM_point_with_potential(
                EoM_cell, solution_global, dof_handler, mapping,
                [&](const auto &p, const auto &values) { return model.EoM(p, values); },
                [&](const auto &p, const auto &values) { return postprocess ? model.EoM_postprocess(p, values) : p; },
                EoM_config, EoM_minimum_guess, &potential_cache);
            EoM = EoM_result.point;
            if (EoM_result.potential) EoM_minimum_guess = EoM_result.potential->minimum;
          }
          if (set_EoM) {
            this->EoM = EoM;
            this->EoM_cell = EoM_cell;
          }

          const auto [x, cell] = resolve_extractor_point(EoM, EoM_cell, solution_global);
          auto solution = reconstruct_readout_solution(cell, solution_global, x, /*with_hessians=*/true);
          const auto raw_potential = extractor_raw_potential(solution_global);
          const auto potential = evaluate_raw_potential(raw_potential, mapping, x);
          model.extract(data, x,
                        e_tie(solution.values, solution.gradients, solution.hessians, nothing, variables,
                              potential.value, potential.gradient, potential.hessian));
        }

        virtual void mass(VectorType &mass, const VectorType &solution_global, const VectorType &solution_global_dot,
                          NumberType weight) override
        {
          using CopyData = internal::CopyData_R<NumberType>;
          const auto &constraints = discretization.get_constraints();

          Scratch scratch_data(quadrature);
          CopyData copy_data;

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_topology(cell);
            constexpr uint n_dofs = n_components;

            copy_data.reinit(cell, n_dofs);

            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch_data);

            std::array<NumberType, n_components> mass_values{};
            for (size_t q_index = 0; q_index < cell_geometry.quadrature_points.size(); ++q_index) {
              const auto &x_q = cell_geometry.quadrature_points[q_index];
              model.mass(mass_values, x_q, scratch_data.solution_values[q_index],
                         scratch_data.solution_dot_values[q_index]);

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

          // map() is collective and each rank visits only its own cells; see NoMapsHere.
          const NoMapsHere no_maps_during_assembly;
          const auto schedule = schedule_for(assembly_cost::local_fe);
          MeshWorker::mesh_loop(locally_owned_cells(dof_handler), cell_worker, copier, scratch_data, copy_data, flags,
                                nullptr, nullptr, schedule.queue_length, schedule.chunk_size);
          // Resolve contributions this rank made to rows it does not own. A partition-boundary
          // face is assembled by exactly one of its two neighbours (mesh_loop hands it to the
          // smaller subdomain id), and that rank writes BOTH sides -- so the other side's rows
          // arrive here. A no-op for the serial types.
          mass.compress(dealii::VectorOperation::add);
        }

        using CellData = internal::CellData<dim, NumberType, n_components>;
        using NeighborData = internal::NeighborData<dim, NumberType, n_components>;
        using CellStencilData = internal::CellStencilData<dim, NumberType, n_components>;
        using CellGeometryDofs = internal::CellGeometryDofs<dim, n_components>;
        using CellStencilTopologyData = internal::CellStencilTopologyData<dim, n_components>;
        using FaceReconstructionState = internal::FaceReconstructionState<dim, NumberType, n_components>;
        using SolutionReconstructionCache = internal::SolutionReconstructionCache<dim, NumberType, n_components>;
        template <typename NT> using CellStencilDataT = internal::CellStencilData<dim, NT, n_components>;

        struct AssemblyFaceGeometryProvider {
          dealii::Tensor<1, dim> normal(const Iterator &cell, const unsigned int face_index) const
          {
            return Assembler::face_normal_from_cell(cell, face_index);
          }

          double jxw(const Iterator &cell, const unsigned int face_index) const
          {
            return Assembler::face_jxw(cell, face_index);
          }
        };

        auto make_assembly_context_view(const SolutionReconstructionCache &cache) const
        {
          using FaceRange =
              FaceAssemblyViewRange<dim, NumberType, n_components, Iterator, AssemblyFaceGeometryProvider>;
          using CellRange = CellAssemblyViewRange<dim, NumberType, n_components, Iterator>;
          return AssemblyContextView<FaceRange, CellRange>(
              FaceRange(dof_handler.begin_active(), dof_handler.end(), cache, AssemblyFaceGeometryProvider{}),
              CellRange(dof_handler.begin_active(), dof_handler.end(), cache));
        }

        auto make_assembly_context_view(SolutionReconstructionCache &&cache) const = delete;

        template <HasAssemblyContextView Context>
        void run_fv_kt_pre_assembly_hook(const AssemblyStage stage, const Context &context)
        {
          dispatch_fv_kt_pre_assembly(model, stage, context);
        }

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

        static bool is_physical_boundary_face(const Iterator &cell, const unsigned int face_index)
        {
          return cell->at_boundary(face_index);
        }

        static Iterator face_neighbor(const Iterator &cell, const unsigned int face_index)
        {
          return cell->neighbor(face_index);
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
            if (is_physical_boundary_face(cell, face_index)) {
              stencil.boundary_ids[face_index] = face->boundary_id();
              if constexpr (dim == 1) {
                auto boundary_stencil =
                    build_boundary_stencil<NumberType>(cell, face_index, solution_global, scratch_dof_indices);
                const bool boundary_supported =
                    model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x, face->center());
                AssertThrow(
                    boundary_supported,
                    ExcMessage("KT boundary stencil was rejected while populating a boundary-adjacent cell stencil."));

                stencil.neighbors.x[face_index] = boundary_stencil.x[boundary_stencil.ghost_center];
                stencil.neighbors.u[face_index] = boundary_stencil.u[boundary_stencil.ghost_center];
                stencil.neighbors.dof_indices[face_index] = boundary_stencil.dof_indices[boundary_stencil.ghost_center];
              }
              continue;
            }

            const auto neighbor = face_neighbor(cell, face_index);
            CellData neighbor_data;
            fill_cell_data(neighbor, solution_global, scratch_dof_indices, neighbor_data);
            stencil.neighbors.x[face_index] = neighbor_data.x;
            stencil.neighbors.u[face_index] = neighbor_data.u;
            stencil.neighbors.dof_indices[face_index] = neighbor_data.dof_indices;
          }
        }

        template <typename BoundaryNumberType>
          requires(dim == 1)
        static internal::BoundaryStencilData<dim, BoundaryNumberType, n_components>
        build_boundary_stencil(const Iterator &cell, const unsigned int boundary_face_no,
                               const VectorType &solution_global,
                               std::vector<types::global_dof_index> &scratch_dof_indices)
        {
          static_assert(dim == 1, "Paper-style boundary stencils currently support only dim=1.");

          using BoundaryIndex = internal::BoundaryStencilIndex<dim>;
          const auto interior_face = GeometryInfo<1>::opposite_face[boundary_face_no];
          AssertThrow(!is_physical_boundary_face(cell, interior_face),
                      ExcMessage("KT boundary stencil requires at least two interior cells behind the boundary face."));

          internal::BoundaryStencilData<dim, BoundaryNumberType, n_components> boundary_stencil{};
          boundary_stencil.lower_boundary = boundary_face_no == 0;
          boundary_stencil.cell_face = boundary_face_no;
          boundary_stencil.ghost_center =
              boundary_stencil.lower_boundary ? BoundaryIndex::lower_inner : BoundaryIndex::upper_inner;
          boundary_stencil.ghost_left =
              boundary_stencil.lower_boundary ? BoundaryIndex::lower_outer : BoundaryIndex::physical_cell;
          boundary_stencil.ghost_right =
              boundary_stencil.lower_boundary ? BoundaryIndex::physical_cell : BoundaryIndex::upper_outer;
          for (auto &dofs : boundary_stencil.dof_indices)
            dofs.fill(numbers::invalid_dof_index);

          CellData cell_data;
          fill_cell_data(cell, solution_global, scratch_dof_indices, cell_data);
          boundary_stencil.x[BoundaryIndex::physical_cell] = cell_data.x;
          boundary_stencil.u[BoundaryIndex::physical_cell] = cell_data.u;
          boundary_stencil.dof_indices[BoundaryIndex::physical_cell] = cell_data.dof_indices;

          auto neighbor = face_neighbor(cell, interior_face);
          CellData first_interior;
          fill_cell_data(neighbor, solution_global, scratch_dof_indices, first_interior);

          AssertThrow(!is_physical_boundary_face(neighbor, interior_face),
                      ExcMessage("KT boundary stencil requires a second interior cell behind the boundary face."));
          auto next_neighbor = face_neighbor(neighbor, interior_face);
          CellData second_interior;
          fill_cell_data(next_neighbor, solution_global, scratch_dof_indices, second_interior);

          if (boundary_stencil.lower_boundary) {
            boundary_stencil.x[BoundaryIndex::upper_inner] = first_interior.x;
            boundary_stencil.x[BoundaryIndex::upper_outer] = second_interior.x;
            boundary_stencil.u[BoundaryIndex::upper_inner] = first_interior.u;
            boundary_stencil.u[BoundaryIndex::upper_outer] = second_interior.u;
            boundary_stencil.dof_indices[BoundaryIndex::upper_inner] = first_interior.dof_indices;
            boundary_stencil.dof_indices[BoundaryIndex::upper_outer] = second_interior.dof_indices;
          } else {
            boundary_stencil.x[BoundaryIndex::lower_inner] = first_interior.x;
            boundary_stencil.x[BoundaryIndex::lower_outer] = second_interior.x;
            boundary_stencil.u[BoundaryIndex::lower_inner] = first_interior.u;
            boundary_stencil.u[BoundaryIndex::lower_outer] = second_interior.u;
            boundary_stencil.dof_indices[BoundaryIndex::lower_inner] = first_interior.dof_indices;
            boundary_stencil.dof_indices[BoundaryIndex::lower_outer] = second_interior.dof_indices;
          }

          return boundary_stencil;
        }

        void fill_cell_data_from_topology(const CellGeometryDofs &topology, const VectorType &solution_global,
                                          CellData &data) const
        {
          internal::fill_cell_data_from_topology<dim, NumberType, n_components>(topology, solution_global, data);
        }

        template <int boundary_dim, typename BoundaryNumberType>
        internal::BoundaryStencilData<boundary_dim, BoundaryNumberType, n_components>
        build_boundary_stencil_from_cache_impl(const Iterator &cell, const unsigned int boundary_face_no,
                                               const VectorType &solution_global) const
        {
          const auto &topology = get_cell_topology(cell).boundary_stencils[boundary_face_no].primary;
          return internal::fill_boundary_stencil_from_topology<BoundaryNumberType, boundary_dim, n_components>(
              topology, solution_global);
        }

        template <typename BoundaryNumberType>
        internal::BoundaryStencilData<dim, BoundaryNumberType, n_components>
        build_boundary_stencil_from_cache(const Iterator &cell, const unsigned int boundary_face_no,
                                          const VectorType &solution_global) const
        {
          return build_boundary_stencil_from_cache_impl<dim, BoundaryNumberType>(cell, boundary_face_no,
                                                                                 solution_global);
        }

        template <typename BoundaryNumberType>
        internal::BoundaryReconstructionStencilData<dim, BoundaryNumberType, n_components>
        build_boundary_reconstruction_stencil_from_cache(const Iterator &cell, const unsigned int boundary_face_no,
                                                         const VectorType &solution_global) const
        {
          const auto &topology = get_cell_topology(cell).boundary_stencils[boundary_face_no];
          return internal::fill_boundary_reconstruction_stencil_from_topology<BoundaryNumberType, dim, n_components>(
              topology, solution_global);
        }

        void fill_cell_stencil(const Iterator &cell, const VectorType &solution_global, CellStencilData &stencil) const
        {
          initialize_cell_stencil_topology(cell->active_cell_index(), stencil);
          refresh_cell_stencil_values(cell->active_cell_index(), solution_global, stencil);
        }

        CellStencilDataT<autodiff::Real<1, NumberType>>
        tag_cell_stencil_dofs_from_cache(const Iterator &cell, const CellStencilData &cell_stencil,
                                         const VectorType &solution_global, const types::global_dof_index dof_j) const
        {
          auto tagged_stencil = internal::tag_cell_stencil_dofs<dim, NumberType, n_components>(cell_stencil, dof_j);

          for (const auto face_index : cell->face_indices()) {
            if (!is_physical_boundary_face(cell, face_index)) continue;

            const auto boundary_stencil =
                build_boundary_stencil_from_cache<NumberType>(cell, face_index, solution_global);
            auto tagged_boundary_stencil =
                internal::tag_boundary_stencil_dofs<dim, NumberType, n_components>(boundary_stencil, dof_j);
            internal::populate_boundary_neighbor_from_model_stencil(tagged_boundary_stencil, tagged_stencil, face_index,
                                                                    cell_stencil.face_centers[face_index], model);
          }

          return tagged_stencil;
        }

        unsigned int find_neighbor_face(const Iterator &cell, const Iterator &neighbor) const
        {
          for (const auto neighbor_face_index : neighbor->face_indices()) {
            if (is_physical_boundary_face(neighbor, neighbor_face_index)) continue;
            if (face_neighbor(neighbor, neighbor_face_index) == cell) return neighbor_face_index;
          }
          AssertThrow(false, ExcMessage("Could not find reciprocal KT neighbor face."));
          return 0;
        }

        template <def::HasReconstructor ActiveReconstructor>
        void rebuild_solution_reconstruction_cache(const VectorType &solution_global,
                                                   SolutionReconstructionCache &cache) const
        {
          ensure_solution_reconstruction_cache_shape(cache);
          for (auto &valid_faces : cache.face_reconstruction_valid)
            valid_faces.fill(false);

          refresh_solution_reconstruction_cache_values(solution_global, cache);

          for (const auto &descriptor : face_reconstruction_descriptors) {
            const auto cell_index = descriptor.cell_index;
            const auto face_index = descriptor.face_index;
            const auto &x_q = descriptor.face_center;

            if (descriptor.boundary) {
              const auto &topology = cell_topology_cache[cell_index].boundary_stencils[face_index];
              auto boundary_stencil =
                  internal::fill_boundary_reconstruction_stencil_from_topology<NumberType, dim, n_components>(
                      topology, solution_global);
              cache.face_reconstructions[cell_index][face_index] =
                  internal::compute_boundary_face_reconstruction_state<ActiveReconstructor>(
                      boundary_stencil, cache.cell_stencils[cell_index], x_q, model);
              cache.face_reconstruction_valid[cell_index][face_index] = true;
              continue;
            }

            Assert(descriptor.neighbor_index.has_value(), ExcInternalError());
            Assert(descriptor.neighbor_face_index.has_value(), ExcInternalError());
            const auto neighbor_index = *descriptor.neighbor_index;
            const auto neighbor_face_index = *descriptor.neighbor_face_index;
            AssertIndexRange(neighbor_index, cache.cell_stencils.size());
            AssertIndexRange(neighbor_face_index, n_faces);

            const auto state = internal::compute_interior_face_reconstruction_state<ActiveReconstructor>(
                cache.cell_stencils[cell_index], cache.cell_stencils[neighbor_index], x_q);
            cache.face_reconstructions[cell_index][face_index] = state;
            cache.face_reconstruction_valid[cell_index][face_index] = true;
            cache.face_reconstructions[neighbor_index][neighbor_face_index] =
                internal::reverse_face_reconstruction(state);
            cache.face_reconstruction_valid[neighbor_index][neighbor_face_index] = true;
          }
        }

        void ensure_solution_reconstruction_cache_shape(SolutionReconstructionCache &cache) const
        {
          const auto n_active_cells = triangulation.n_active_cells();
          if (cache.cell_stencils.size() != n_active_cells) {
            cache.cell_stencils.resize(n_active_cells);
            cache.topology_initialized = false;
          }
          if (cache.face_reconstructions.size() != n_active_cells) cache.face_reconstructions.resize(n_active_cells);
          if (cache.face_reconstruction_valid.size() != n_active_cells)
            cache.face_reconstruction_valid.resize(n_active_cells);
          if (!cache.topology_initialized) initialize_solution_reconstruction_cache_topology(cache);
        }

        void initialize_solution_reconstruction_cache_topology(SolutionReconstructionCache &cache) const
        {
          for (unsigned int cell_index = 0; cell_index < cache.cell_stencils.size(); ++cell_index)
            initialize_cell_stencil_topology(cell_index, cache.cell_stencils[cell_index]);
          cache.topology_initialized = true;
        }

        void initialize_cell_stencil_topology(const unsigned int cell_index, CellStencilData &stencil) const
        {
          const auto &topology = cell_topology_cache[cell_index].stencil;
          stencil.boundary_ids = topology.boundary_ids;
          stencil.face_centers = topology.face_centers;
          stencil.cell.x = topology.cell.x;
          stencil.cell.dof_indices = topology.cell.dof_indices;
          stencil.neighbors.x = topology.neighbors.x;
          stencil.neighbors.dof_indices = topology.neighbors.dof_indices;
        }

        void refresh_solution_reconstruction_cache_values(const VectorType &solution_global,
                                                          SolutionReconstructionCache &cache) const
        {
          for (unsigned int cell_index = 0; cell_index < cache.cell_stencils.size(); ++cell_index)
            refresh_cell_stencil_values(cell_index, solution_global, cache.cell_stencils[cell_index]);
        }

        void refresh_cell_stencil_values(const unsigned int cell_index, const VectorType &solution_global,
                                         CellStencilData &stencil) const
        {
          const auto &topology = cell_topology_cache[cell_index].stencil;
          for (unsigned int i = 0; i < n_components; ++i)
            stencil.cell.u[i] = solution_global(topology.cell.dof_indices[i]);

          for (unsigned int face_index = 0; face_index < n_faces; ++face_index) {
            if (topology.boundary_ids[face_index] == numbers::invalid_boundary_id) {
              for (unsigned int i = 0; i < n_components; ++i) {
                const auto dof = topology.neighbors.dof_indices[face_index][i];
                stencil.neighbors.u[face_index][i] = solution_global(dof);
              }
              continue;
            }

            auto boundary_stencil = internal::fill_boundary_stencil_from_topology<NumberType, dim, n_components>(
                cell_topology_cache[cell_index].boundary_stencils[face_index].primary, solution_global);
            internal::populate_boundary_neighbor_from_model_stencil(boundary_stencil, stencil, face_index,
                                                                    topology.face_centers[face_index], model);
          }
        }

        void rebuild_face_reconstruction_descriptors()
        {
          face_reconstruction_descriptors.clear();
          face_reconstruction_descriptors.reserve(triangulation.n_active_cells() * n_faces);

          // Every cell, not just the owned ones: an owned cell's flux worker reads its neighbours'
          // descriptors, and the `neighbor_index < cell_index` tiebreak below needs both sides
          // present to pick a side consistently.
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            const auto cell_index = cell->active_cell_index();
            for (const auto face_index : cell->face_indices()) {
              FaceReconstructionDescriptor descriptor;
              descriptor.cell_index = cell_index;
              descriptor.face_index = face_index;
              descriptor.boundary = is_physical_boundary_face(cell, face_index);
              descriptor.face_center = cell_topology_cache[cell_index].stencil.face_centers[face_index];
              if (descriptor.boundary) {
                face_reconstruction_descriptors.push_back(descriptor);
                continue;
              }

              const auto neighbor = face_neighbor(cell, face_index);
              const auto neighbor_index = neighbor->active_cell_index();
              if (neighbor_index < cell_index) continue;
              descriptor.neighbor_index = neighbor_index;
              descriptor.neighbor_face_index = find_neighbor_face(cell, neighbor);
              face_reconstruction_descriptors.push_back(descriptor);
            }
          }
        }

        const FaceReconstructionState &get_cached_face_reconstruction(const SolutionReconstructionCache &cache,
                                                                      const Iterator &cell,
                                                                      const unsigned int face_index) const
        {
          const auto cell_index = cell->active_cell_index();
          AssertIndexRange(cell_index, cache.face_reconstructions.size());
          AssertIndexRange(face_index, n_faces);
          AssertThrow(cache.face_reconstruction_valid[cell_index][face_index],
                      ExcMessage("KT face reconstruction cache entry was not initialized."));
          return cache.face_reconstructions[cell_index][face_index];
        }

        auto compute_interior_face_reconstruction_from_cache(const Iterator &cell, const Iterator &ncell,
                                                             const VectorType &solution_global, const Point &x_q,
                                                             Scratch &scratch_data) const
        {
          fill_cell_stencil(cell, solution_global, scratch_data.cell_stencil);
          fill_cell_stencil(ncell, solution_global, scratch_data.ncell_stencil);
          return internal::compute_interior_face_reconstruction_state<Reconstructor>(scratch_data.cell_stencil,
                                                                                     scratch_data.ncell_stencil, x_q);
        }

        auto compute_boundary_face_reconstruction_from_cache(const Iterator &cell, const unsigned int face_no,
                                                             const VectorType &solution_global, const Point &x_q) const
        {
          auto boundary_stencil =
              build_boundary_reconstruction_stencil_from_cache<NumberType>(cell, face_no, solution_global);
          CellStencilData cell_stencil;
          fill_cell_stencil(cell, solution_global, cell_stencil);
          return internal::compute_boundary_face_reconstruction_state<Reconstructor>(boundary_stencil, cell_stencil,
                                                                                     x_q, model);
        }

        auto compute_interior_jacobian_face_reconstruction_from_cache(const Iterator &cell, const Iterator &ncell,
                                                                      const VectorType &solution_global,
                                                                      const Point &x_q, Scratch &scratch_data) const
        {
          fill_cell_stencil(cell, solution_global, scratch_data.cell_stencil);
          fill_cell_stencil(ncell, solution_global, scratch_data.ncell_stencil);
          return internal::compute_interior_face_reconstruction_state<JacobianReconstructor>(
              scratch_data.cell_stencil, scratch_data.ncell_stencil, x_q);
        }

        auto compute_boundary_jacobian_face_reconstruction_from_cache(const Iterator &cell, const unsigned int face_no,
                                                                      const VectorType &solution_global,
                                                                      const Point &x_q) const
        {
          auto boundary_stencil =
              build_boundary_reconstruction_stencil_from_cache<NumberType>(cell, face_no, solution_global);
          CellStencilData cell_stencil;
          fill_cell_stencil(cell, solution_global, cell_stencil);
          return internal::compute_boundary_face_reconstruction_state<JacobianReconstructor>(boundary_stencil,
                                                                                             cell_stencil, x_q, model);
        }

        /**
         * @brief One-shot conditioning check on the model's diffusion flux.
         *
         * The FV residual differences the diffusion flux between a cell's faces. Any part of
         * the flux that does *not* depend on the gradient therefore cancels analytically —
         * but not in floating point, where it leaves its round-off, |F|·eps, behind. When that
         * gradient-independent baseline dwarfs the gradient-dependent part, the residual loses
         * the corresponding number of digits, and an implicit solver asked for a tight
         * tolerance will grind its step size down trying to resolve pure noise.
         *
         * This is the standard failure mode for fRG loop integrals at large RG scale: a flux
         * c·k^n·g(k² + ∂u) carries an O(k^(n-1)) baseline, so the relative noise is ~eps·k²/Δ(∂u)
         * and blows up as the UV cutoff grows.
         *
         * The fix belongs in the model, not here: return the flux with its zero-gradient value
         * already subtracted, F - F|_{∂u = 0}, written in a cancellation-free form. Subtracting
         * a gradient-independent constant leaves the residual unchanged (a constant flux has
         * zero divergence and cancels on the ghost side of boundary faces too).
         */
        template <typename ExtractorArray>
        void probe_diffusion_flux_conditioning(const SolutionReconstructionCache &reconstruction_cache,
                                               const ExtractorArray &extractors, const VectorType &variables) const
        {
          // Probe the MODEL, not the current data. Using the solution's actual gradient fails
          // whenever the flow starts from flat initial data: the local variation is then ~0 and
          // every flux looks baseline-dominated. Instead perturb the gradient by a synthetic
          // scale built from the solution magnitude and the domain size, which is what the
          // gradient will grow to once the flow develops.
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D_actual{};
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D_baseline{};
          const ThirdDerivativeType zero_third{};
          const GradientType zero_grad{};

          const double domain_size = std::max(domain_diameter, 1e-300);
          std::array<double, n_components> max_baseline{};
          std::array<double, n_components> max_variation{};
          double u_scale = 0.0;
          unsigned int sampled = 0;

          // First pass: the scale of the solution itself.
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (sampled >= flux_conditioning_max_samples) break;
            for (unsigned int f = 0; f < n_faces; ++f) {
              if (cell->at_boundary(f)) continue;
              const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, f);
              for (size_t c = 0; c < n_components; ++c)
                u_scale = std::max(u_scale, std::abs(static_cast<double>(reconstruction.diffusion_u_minus[c])));
              ++sampled;
              break;
            }
          }

          // A unit fallback keeps the probe meaningful for a flow starting from u == 0.
          const NumberType probe_gradient = static_cast<NumberType>(std::max(u_scale, 1.0) / domain_size);
          GradientType probe_grad{};
          for (size_t c = 0; c < n_components; ++c)
            for (int d = 0; d < dim; ++d)
              probe_grad[c][d] = probe_gradient;

          sampled = 0;
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (sampled >= flux_conditioning_max_samples) break;
            for (unsigned int f = 0; f < n_faces; ++f) {
              if (cell->at_boundary(f)) continue;
              const auto x_q = cell->face(f)->center();
              const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, f);
              const double probe_width = get_cell_topology(cell).cell_width;
              model.diffusion_flux(D_actual, x_q,
                                   internal::diffusion_flux_tie(reconstruction.diffusion_u_minus, probe_grad,
                                                                zero_third, extractors, variables, probe_width));
              model.diffusion_flux(D_baseline, x_q,
                                   internal::diffusion_flux_tie(reconstruction.diffusion_u_minus, zero_grad, zero_third,
                                                                extractors, variables, probe_width));
              for (size_t c = 0; c < n_components; ++c) {
                max_baseline[c] = std::max(max_baseline[c], static_cast<double>(D_baseline[c].norm()));
                max_variation[c] =
                    std::max(max_variation[c], static_cast<double>((D_actual[c] - D_baseline[c]).norm()));
              }
              ++sampled;
              break; // one face per cell is plenty
            }
          }

          for (size_t c = 0; c < n_components; ++c) {
            if (!(max_variation[c] > 0.0) || !std::isfinite(max_baseline[c])) continue;

            const double relative_noise =
                max_baseline[c] / max_variation[c] * std::numeric_limits<NumberType>::epsilon();
            if (relative_noise <= flux_conditioning_warn_threshold) continue;

            const auto message = spdlog::fmt_lib::format(
                "FV/KT: the diffusion flux of component {} is dominated by a gradient-independent baseline: "
                "max |F(du=0)| = {:.3e} over the mesh, but max |F - F(du=0)| = only {:.3e}. Differencing it "
                "across faces leaves a relative round-off of {:.1e} in the residual, which caps the accuracy any "
                "implicit solver can reach (and will crush its step size if the tolerance is tighter). Subtract "
                "the zero-gradient value analytically inside diffusion_flux().",
                c, max_baseline[c], max_variation[c], relative_noise);

            // Deliberately not silenceable by an unattached LogPort: this only fires when the
            // model is measurably losing digits, and it is precisely the kind of thing that
            // must not disappear in a run that did not wire up a logger.
            if (log_port)
              log_port.warn("{}", message);
            else
              spdlog::warn("{}", message);
          }
        }

        void fill_constant_quadrature_values(const Iterator &cell, const VectorType &solution_global,
                                             const VectorType &solution_global_dot, Scratch &scratch_data) const
        {
          fill_cell_data_from_topology(get_cell_topology(cell).stencil.cell, solution_global,
                                       scratch_data.cell_stencil.cell);
          for (auto &values : scratch_data.solution_values)
            for (uint c = 0; c < n_components; ++c)
              values[c] = scratch_data.cell_stencil.cell.u[c];

          for (auto &values_dot : scratch_data.solution_dot_values)
            for (uint c = 0; c < n_components; ++c)
              values_dot[c] = solution_global_dot(scratch_data.cell_stencil.cell.dof_indices[c]);
        }

        /**
         * @brief The "fe_derivatives" slot of fv_tie(), i.e. the gradient model.source() sees at x_q.
         *
         * The same reconstruction the readout path and the face fluxes use. At the default single
         * quadrature point -- the cell centre -- compute_gradient_at_point falls through to the limiter, so
         * this is the scheme's own limited slope; with overintegration it becomes one-sided towards x_q.
         *
         * Boundary cells need no special case: the stencil's neighbour slot across a physical boundary face
         * already holds the model's ghost value (see refresh_cell_stencil_values).
         */
        template <def::HasReconstructor ActiveReconstructor>
        static GradientType source_gradient(const CellStencilData &stencil, const Point &x_q)
        {
          return ActiveReconstructor::template compute_gradient_at_point<n_components>(
              stencil.cell.x, x_q, stencil.cell.u, stencil.neighbors.x, stencil.neighbors.u);
        }

        static std::array<internal::GradientType<dim, NumberType, n_components>, n_faces>
        compute_neighbor_gradients(const Iterator &cell, const VectorType &solution_global, const Model &model,
                                   std::vector<types::global_dof_index> &scratch_dof_indices,
                                   CellStencilData &temporary_stencil)
        {
          std::array<internal::GradientType<dim, NumberType, n_components>, n_faces> gradients{};

          for (const auto face_index : cell->face_indices()) {
            if (is_physical_boundary_face(cell, face_index)) continue;

            const auto neighbor = face_neighbor(cell, face_index);
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
                              const VectorType &variables = VectorType()) override
        {
          using CopyData = internal::CopyData_R<NumberType>;
          const auto &constraints = discretization.get_constraints();

          // Find the EoM and extract whatever data is needed for the model, as the FEM assemblers do. Beyond
          // filling `extracted_data` this is what gives a model the chance to refresh whatever internal state its
          // flux depends on (interpolators, self-consistently solved anomalous dimensions, ...) before the fluxes
          // are evaluated. Models without extractors are unaffected.
          std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
          if constexpr (Components::count_extractors() > 0)
            extract(__extracted_data, solution_global, variables, true, false, true);

          Scratch scratch_data(quadrature);
          CopyData copy_data;

          Timer timer;
          rebuild_solution_reconstruction_cache<Reconstructor>(solution_global, residual_reconstruction_cache);
          const auto &reconstruction_cache = residual_reconstruction_cache;
          const auto assembly_context = make_assembly_context_view(reconstruction_cache);
          run_fv_kt_pre_assembly_hook(AssemblyStage::residual, assembly_context);

          // Runs on the first residual assembly only, outside the mesh loop, so it costs
          // nothing on the hot path. The first assembly is also the most informative sample:
          // for an fRG flow it happens at k = Lambda, where a flux baseline is largest.
          if (diagnose_flux_conditioning && !flux_conditioning_probed) {
            flux_conditioning_probed = true;
            probe_diffusion_flux_conditioning(reconstruction_cache, __extracted_data, variables);
          }

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_topology(cell);
            constexpr uint n_dofs = n_components;

            copy_data.reinit(cell, n_dofs);

            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch_data);
            const auto &stencil = reconstruction_cache.cell_stencils[cell->active_cell_index()];

            std::array<NumberType, n_components> mass{};
            std::array<NumberType, n_components> source{};
            for (size_t q_index = 0; q_index < cell_geometry.quadrature_points.size(); ++q_index) {
              const auto &x_q = cell_geometry.quadrature_points[q_index];
              const auto gradients = source_gradient<Reconstructor>(stencil, x_q);
              model.mass(mass, x_q, scratch_data.solution_values[q_index], scratch_data.solution_dot_values[q_index]);
              model.source(source, x_q,
                           fv_tie(scratch_data.solution_values[q_index], gradients, __extracted_data, variables,
                                  cell_geometry.cell_width));

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = local_component_of_dof[i];
                copy_data.cell_mass(i) +=
                    weight_mass * cell_geometry.jxw[q_index] * mass[component_i]; // +phi_i(x_q) * mass(x_q, u_q)
                copy_data.cell_residual(i) +=
                    cell_geometry.jxw[q_index] * weight * source[component_i]; // -phi_i(x_q) * source(x_q, u_q)
              }
            }
          };

          const auto face_worker = [&](const Iterator &cell, const unsigned int &f,
                                       [[maybe_unused]] const unsigned int &sf, const Iterator &ncell,
                                       [[maybe_unused]] const unsigned int &nf,
                                       [[maybe_unused]] const unsigned int &nsf, [[maybe_unused]] Scratch &scratch_data,
                                       CopyData &copy_data) {
            [[maybe_unused]] const int q_face_index = 0; // only one quadrature point per face for FV (constant FE)
            const auto x_q = cell->face(f)->center();
            const auto n_face = face_normal_from_cell(cell, f);
            const auto JxW = face_jxw(cell, f);
            const uint n_face_dofs = 2 * n_components;

            auto &copy_data_face = copy_data.next_face_data();
            copy_data_face.reinit(n_face_dofs);

            const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, f);
            const auto &cell_stencil = reconstruction_cache.cell_stencils[cell->active_cell_index()];
            const auto &ncell_stencil = reconstruction_cache.cell_stencils[ncell->active_cell_index()];
            const auto &cell_data = cell_stencil.cell;
            const auto &ncell_data = ncell_stencil.cell;

            const double width_minus = get_cell_topology(cell).cell_width;
            const double width_plus = get_cell_topology(ncell).cell_width;

            const auto [F_plus, F_minus, a_half] = internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(
                reconstruction.u_plus, reconstruction.u_minus, reconstruction.face_grad_plus,
                reconstruction.face_grad_minus, x_q, width_plus, width_minus, __extracted_data, variables, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, reconstruction.u_plus,
                                                            reconstruction.u_minus);
            const auto D = internal::compute_diffusion_flux(
                reconstruction.diffusion_u_minus, reconstruction.diffusion_u_plus, reconstruction.diffusion_grad_minus,
                reconstruction.diffusion_grad_plus, reconstruction.third_derivatives_minus,
                reconstruction.third_derivatives_plus, x_q, width_minus, width_plus, __extracted_data, variables,
                model);

            // Sign convention: the face flux is (H + D)·n, i.e. the advection numerical
            // flux H (from flux()) and the diffusion flux D (from diffusion_flux())
            // are SUMMED. Both model methods therefore return the physical flux
            // with the same sign — exactly the conservation-law convention used by CG /
            // LLFFlux (which sums all contributions into one flux F). A diffusion flux
            // f_diff must be a DECREASING function of the gradient (∂f_diff/∂(∂u) < 0)
            // for forward diffusion, e.g. f_diff = -ν·∂u for the heat/viscous term.
            for (uint component_i = 0; component_i < n_components; ++component_i) {
              copy_data_face.joint_dof_indices[component_i] = cell_data.dof_indices[component_i];
              copy_data_face.joint_dof_indices[n_components + component_i] = ncell_data.dof_indices[component_i];
              const auto flux_contribution =
                  weight * JxW * (scalar_product(H[component_i], n_face) + scalar_product(D[component_i], n_face));
              copy_data_face.cell_residual(component_i) += flux_contribution;
              copy_data_face.cell_residual(n_components + component_i) -= flux_contribution;
            }
          };

          const auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no,
                                           [[maybe_unused]] Scratch &scratch_data, CopyData &copy_data) {
            const uint n_face_dofs = n_components;

            auto &copy_data_face = copy_data.next_face_data();
            copy_data_face.reinit(n_face_dofs);

            [[maybe_unused]] const int q_face_index = 0; // only one quadrature point per face for FV
            const auto x_q = cell->face(face_no)->center();
            const auto JxW = face_jxw(cell, face_no);
            const auto n_bnd = face_normal_from_cell(cell, face_no);

            const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, face_no);
            const auto &cell_data = reconstruction_cache.cell_stencils[cell->active_cell_index()].cell;

            const double width_minus = get_cell_topology(cell).cell_width;
            const double width_plus = width_minus;

            const auto [F_plus, F_minus, a_half] = internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(
                reconstruction.u_plus, reconstruction.u_minus, reconstruction.face_grad_plus,
                reconstruction.face_grad_minus, x_q, width_plus, width_minus, __extracted_data, variables, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, reconstruction.u_plus,
                                                            reconstruction.u_minus);

            const auto D_bnd = internal::compute_diffusion_flux(
                reconstruction.diffusion_u_minus, reconstruction.diffusion_u_plus, reconstruction.diffusion_grad_minus,
                reconstruction.diffusion_grad_plus, reconstruction.third_derivatives_minus,
                reconstruction.third_derivatives_plus, x_q, width_minus, width_plus, __extracted_data, variables,
                model);

            for (uint component_i = 0; component_i < n_components; ++component_i) {
              copy_data_face.joint_dof_indices[component_i] = cell_data.dof_indices[component_i];
              copy_data_face.cell_residual(component_i) +=
                  weight * JxW * (scalar_product(H[component_i], n_bnd) + scalar_product(D_bnd[component_i], n_bnd));
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
                                            MeshWorker::assemble_own_interior_faces_once |
                                            MeshWorker::assemble_ghost_faces_once;

          // map() is collective and each rank visits only its own cells; see NoMapsHere.
          const NoMapsHere no_maps_during_assembly;
          const auto schedule = schedule_for(assembly_cost::momentum_integral);
          MeshWorker::mesh_loop(locally_owned_cells(dof_handler), cell_worker, copier, scratch_data, copy_data, flags,
                                boundary_worker, face_worker, schedule.queue_length, schedule.chunk_size);
          // Resolve contributions this rank made to rows it does not own. A partition-boundary
          // face is assembled by exactly one of its two neighbours (mesh_loop hands it to the
          // smaller subdomain id), and that rank writes BOTH sides -- so the other side's rows
          // arrive here. A no-op for the serial types.
          residual.compress(dealii::VectorOperation::add);
          timings_residual.push_back(timer.wall_time());
        }

        virtual void jacobian_mass(SparseMatrixType &jacobian, const VectorType &solution_global,
                                   const VectorType &solution_global_dot, NumberType alpha = 1.,
                                   NumberType beta = 1.) override
        {
          using Iterator = typename DoFHandler<dim>::active_cell_iterator;
          using CopyData = internal::CopyData_J<NumberType, dim>;
          const auto &constraints = discretization.get_constraints();

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_topology(cell);
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
                  copy_data.cell_jacobian(i, j) +=
                      cell_geometry.jxw[q_index] *
                      (alpha * j_mass_dot(component_i, component_j) + beta * j_mass(component_i, component_j));
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
          // map() is collective and each rank visits only its own cells; see NoMapsHere.
          const NoMapsHere no_maps_during_assembly;
          const auto schedule = schedule_for(assembly_cost::local_fe);
          MeshWorker::mesh_loop(locally_owned_cells(dof_handler), cell_worker, copier, scratch_data, copy_data, flags,
                                nullptr, nullptr, schedule.queue_length, schedule.chunk_size);
          // Resolve contributions this rank made to rows it does not own. A partition-boundary
          // face is assembled by exactly one of its two neighbours (mesh_loop hands it to the
          // smaller subdomain id), and that rank writes BOTH sides -- so the other side's rows
          // arrive here. A no-op for the serial types.
          jacobian.compress(dealii::VectorOperation::add);
          timings_jacobian.push_back(timer.wall_time());
        }

        virtual void jacobian(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType weight,
                              const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                              const VectorType &variables = VectorType()) override
        {
          using Iterator = typename DoFHandler<dim>::active_cell_iterator;
          using CopyData = internal::CopyData_J<NumberType, dim>;
          const auto &constraints = discretization.get_constraints();

          // See residual(): keep the model's extractor-driven state consistent with the point the jacobian is
          // linearised about. The extractor jacobian contribution itself is not assembled here (the FV
          // extractor_cell_jacobian blocks remain unused), so extractors are treated as frozen w.r.t. the FE
          // solution within a Newton step -- fine for the IDA/explicit split where Variables are stepped
          // explicitly, but it is why jacobian_variables below is still a no-op.
          std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
          if constexpr (Components::count_extractors() > 0)
            extract(__extracted_data, solution_global, variables, true, false, true);
          Timer timer;
          rebuild_solution_reconstruction_cache<JacobianReconstructor>(solution_global, jacobian_reconstruction_cache);
          const auto &reconstruction_cache = jacobian_reconstruction_cache;
          const auto assembly_context = make_assembly_context_view(reconstruction_cache);
          run_fv_kt_pre_assembly_hook(AssemblyStage::jacobian, assembly_context);

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_topology(cell);
            constexpr uint n_dofs = n_components;

            copy_data.reinit(cell, n_dofs, Components::count_extractors());

            fill_constant_quadrature_values(cell, solution_global, solution_global_dot, scratch_data);
            const auto &stencil = reconstruction_cache.cell_stencils[cell->active_cell_index()];

            // The source's gradient dependence makes it nonlocal -- it reaches the whole 2*dim stencil
            // through the reconstruction -- so that part cannot live in the square, cell-local
            // cell_jacobian. It goes into a rectangular block, exactly as the face workers do.
            auto &copy_data_source = copy_data.next_face_data();
            copy_data_source.reinit(cell_geometry.source_jacobian_dependencies.to_dofs,
                                    cell_geometry.source_jacobian_dependencies.from_dofs);
            const uint n_source_from = size(copy_data_source.from_dofs);
            auto &source_grad_deriv = scratch_data.source_gradient_derivatives;
            source_grad_deriv.resize(n_source_from);

            SimpleMatrix<NumberType, n_components> j_mass;
            SimpleMatrix<NumberType, n_components> j_mass_dot;
            SimpleMatrix<NumberType, n_components> j_source;
            SimpleMatrix<Tensor<1, dim, NumberType>, n_components> j_grad_source;
            for (size_t q_index = 0; q_index < cell_geometry.quadrature_points.size(); ++q_index) {
              const auto &x_q = cell_geometry.quadrature_points[q_index];
              const auto gradients = source_gradient<JacobianReconstructor>(stencil, x_q);
              const auto source_tie = fv_tie(scratch_data.solution_values[q_index], gradients, __extracted_data,
                                             variables, cell_geometry.cell_width);
              model.template jacobian_mass<0>(j_mass, x_q, scratch_data.solution_values[q_index],
                                              scratch_data.solution_dot_values[q_index]);
              model.template jacobian_mass<1>(j_mass_dot, x_q, scratch_data.solution_values[q_index],
                                              scratch_data.solution_dot_values[q_index]);
              model.template jacobian_source<0, 0>(j_source, x_q, source_tie);
              // No jacobian_source_extr counterpart: KT never assembles extractor_cell_jacobian and
              // jacobian_variables is a no-op, so extractors and variables are frozen within a Newton step.
              model.template jacobian_source_grad<1>(j_grad_source, x_q, source_tie);

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

              // d(source)/d(u_j) via the gradient: seed one stencil dof at a time, exactly as the face
              // workers do for the flux, and chain d(grad)/d(u_j) into the model's dsource/dgrad.
              for (uint j = 0; j < n_source_from; ++j) {
                const auto stencil_tagged =
                    tag_cell_stencil_dofs_from_cache(cell, stencil, solution_global, copy_data_source.from_dofs[j]);
                source_grad_deriv[j] =
                    JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                        stencil.cell.x, x_q, stencil_tagged.cell.u, stencil.neighbors.x, stencil_tagged.neighbors.u);
              }
              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = local_component_of_dof[i];
                for (uint j = 0; j < n_source_from; ++j) {
                  NumberType contribution{};
                  for (uint c = 0; c < n_components; ++c)
                    for (uint d = 0; d < dim; ++d)
                      contribution += j_grad_source(component_i, c)[d] * source_grad_deriv[j][c][d];
                  copy_data_source.cell_jacobian(i, j) += weight * cell_geometry.jxw[q_index] * contribution;
                }
              }
            }
          };
          const auto face_worker = [&](const Iterator &cell, const unsigned int &f,
                                       [[maybe_unused]] const unsigned int &sf, const Iterator &ncell,
                                       [[maybe_unused]] const unsigned int &nf,
                                       [[maybe_unused]] const unsigned int &nsf, Scratch &scratch_data,
                                       CopyData &copy_data) {
            [[maybe_unused]] const int q_face_index = 0;
            const auto x_q = cell->face(f)->center();
            const auto JxW = face_jxw(cell, f);
            const auto n_face = face_normal_from_cell(cell, f);

            const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, f);
            const auto &cell_stencil = reconstruction_cache.cell_stencils[cell->active_cell_index()];
            const auto &ncell_stencil = reconstruction_cache.cell_stencils[ncell->active_cell_index()];
            const auto &cell_neighbors = cell_stencil.neighbors;
            const auto &ncell_neighbors = ncell_stencil.neighbors;
            const auto &cell_data = cell_stencil.cell;
            const auto &ncell_data = ncell_stencil.cell;

            auto &copy_data_face = copy_data.next_face_data();
            const auto &face_dependencies = get_cell_topology(cell).face_jacobian_dependencies[f];
            copy_data_face.reinit(face_dependencies.to_dofs, face_dependencies.from_dofs);

            // Precompute reconstructed state and face-gradient derivatives for each dependency dof.
            const uint n_from = size(copy_data_face.from_dofs);
            for (auto &derivatives : scratch_data.reconstructed_derivatives) {
              if (derivatives.capacity() < n_from) derivatives.reserve(n_from);
              derivatives.resize(n_from);
            }
            auto &reconstructed_deriv = scratch_data.reconstructed_derivatives;
            auto &diffusion_deriv = scratch_data.diffusion_derivatives;
            for (auto &derivatives : diffusion_deriv) {
              if (derivatives.capacity() < n_from) derivatives.reserve(n_from);
              derivatives.resize(n_from);
            }

            for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
              const auto dof_j = copy_data_face.from_dofs[j];
              const auto cell_stencil_tagged =
                  tag_cell_stencil_dofs_from_cache(cell, cell_stencil, solution_global, dof_j);
              const auto ncell_stencil_tagged =
                  tag_cell_stencil_dofs_from_cache(ncell, ncell_stencil, solution_global, dof_j);

              // face_no=0: d(u⁻)/d(u_j) — reconstruction from cell side
              {
                reconstructed_deriv[0][j].u =
                    internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                        cell_stencil_tagged.cell.u, cell_data.x, x_q, cell_neighbors.x,
                        cell_stencil_tagged.neighbors.u);
                reconstructed_deriv[0][j].grad =
                    JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                        cell_data.x, x_q, cell_stencil_tagged.cell.u, cell_neighbors.x,
                        cell_stencil_tagged.neighbors.u);
              }

              // face_no=1: d(u⁺)/d(u_j) — reconstruction from neighbor side
              {
                reconstructed_deriv[1][j].u =
                    internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                        ncell_stencil_tagged.cell.u, ncell_data.x, x_q, ncell_neighbors.x,
                        ncell_stencil_tagged.neighbors.u);
                reconstructed_deriv[1][j].grad =
                    JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                        ncell_data.x, x_q, ncell_stencil_tagged.cell.u, ncell_neighbors.x,
                        ncell_stencil_tagged.neighbors.u);
              }

              if constexpr (dim == 1) {
                const auto third_derivative_stencil_ad =
                    internal::make_interior_third_derivative_stencil(cell_stencil_tagged, ncell_stencil_tagged, x_q);
                const auto third_derivatives =
                    JacobianReconstructor::template compute_third_derivatives_at_face_derivative<n_components>(
                        third_derivative_stencil_ad.x, third_derivative_stencil_ad.u);
                reconstructed_deriv[0][j].third_derivatives = third_derivatives;
                reconstructed_deriv[1][j].third_derivatives = third_derivatives;
              }

              const auto diffusion_face_ad =
                  internal::compute_diffusion_face_state(cell_stencil_tagged, ncell_stencil_tagged);
              const auto diffusion_face_derivatives =
                  internal::extract_diffusion_face_derivatives<dim, NumberType, n_components>(diffusion_face_ad);
              diffusion_deriv[0][j] = diffusion_face_derivatives[0];
              diffusion_deriv[1][j] = diffusion_face_derivatives[1];
            }

            const double width_minus = get_cell_topology(cell).cell_width;
            const double width_plus = get_cell_topology(ncell).cell_width;

            const auto j_numflux =
                internal::compute_kt_numflux_jacobian<WaveSpeedStrategy, Model, NumberType, dim, n_components>(
                    reconstruction.u_plus, reconstruction.u_minus, reconstruction.face_grad_plus,
                    reconstruction.face_grad_minus, x_q, width_plus, width_minus, __extracted_data, variables, model);
            const auto j_diffusion = internal::compute_diffusion_flux_jacobian<Model, NumberType, dim, n_components>(
                reconstruction.diffusion_u_minus, reconstruction.diffusion_u_plus, reconstruction.diffusion_grad_minus,
                reconstruction.diffusion_grad_plus, reconstruction.third_derivatives_minus,
                reconstruction.third_derivatives_plus, x_q, width_minus, width_plus, __extracted_data, variables,
                model);

            for (uint i = 0; i < size(copy_data_face.to_dofs); ++i) {
              const bool cell_side_i = i < n_components;
              const auto component_i = cell_side_i ? i : i - n_components;
              const auto jump_i = cell_side_i ? NumberType(1.) : NumberType(-1.);
              for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
                NumberType diffusion_contribution{};
                for (size_t face_no = 0; face_no < 2; ++face_no) {
                  NumberType advection_contribution{};
                  for (size_t c = 0; c < n_components; ++c) {
                    advection_contribution += scalar_product(j_numflux.u[face_no](component_i, c), n_face) *
                                              reconstructed_deriv[face_no][j].u[c];
                    for (size_t d_in = 0; d_in < dim; ++d_in)
                      for (size_t d_out = 0; d_out < dim; ++d_out)
                        advection_contribution += j_numflux.grad[face_no](component_i, c)[d_out][d_in] * n_face[d_out] *
                                                  reconstructed_deriv[face_no][j].grad[c][d_in];
                  }
                  copy_data_face.cell_jacobian(i, j) += weight * JxW * jump_i * advection_contribution;
                }

                for (size_t face_no = 0; face_no < 2; ++face_no)
                  for (size_t c = 0; c < n_components; ++c) {
                    diffusion_contribution += scalar_product(j_diffusion.u[face_no](component_i, c), n_face) *
                                              diffusion_deriv[face_no][j].u[c];
                    for (size_t d_in = 0; d_in < dim; ++d_in)
                      for (size_t d_out = 0; d_out < dim; ++d_out)
                        diffusion_contribution += j_diffusion.grad[face_no](component_i, c)[d_out][d_in] *
                                                  n_face[d_out] * diffusion_deriv[face_no][j].grad[c][d_in];
                    for (size_t d0 = 0; d0 < dim; ++d0)
                      for (size_t d1 = 0; d1 < dim; ++d1)
                        for (size_t d2 = 0; d2 < dim; ++d2)
                          for (size_t d_out = 0; d_out < dim; ++d_out)
                            diffusion_contribution +=
                                j_diffusion.third_derivatives[face_no](component_i, c)[d_out][d0][d1][d2] *
                                n_face[d_out] * reconstructed_deriv[face_no][j].third_derivatives[c][d0][d1][d2];
                  }

                // The residual uses [[phi_i]] * ((H + D) · n). The diffusion
                // path has two corrected side gradients, separate from the two KT
                // advective traces.
                copy_data_face.cell_jacobian(i, j) += weight * JxW * jump_i * diffusion_contribution;
              }
            }
          };

          const auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no, Scratch &scratch_data,
                                           CopyData &copy_data) {
            [[maybe_unused]] const int q_face_index = 0;
            const auto x_q = cell->face(face_no)->center();
            const auto JxW = face_jxw(cell, face_no);
            const auto n_face = face_normal_from_cell(cell, face_no);

            auto &copy_data_face = copy_data.next_face_data();
            const auto &face_dependencies = get_cell_topology(cell).face_jacobian_dependencies[face_no];
            copy_data_face.reinit(face_dependencies.to_dofs, face_dependencies.from_dofs);

            const auto boundary_stencil =
                build_boundary_reconstruction_stencil_from_cache<NumberType>(cell, face_no, solution_global);
            const auto &cell_stencil = reconstruction_cache.cell_stencils[cell->active_cell_index()];
            const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, face_no);

            // Precompute reconstructed state and face-gradient derivatives for each dependency dof.
            const uint n_from = size(copy_data_face.from_dofs);
            for (auto &derivatives : scratch_data.reconstructed_derivatives) {
              if (derivatives.capacity() < n_from) derivatives.reserve(n_from);
              derivatives.resize(n_from);
            }
            auto &reconstructed_deriv = scratch_data.reconstructed_derivatives;
            auto &diffusion_deriv = scratch_data.diffusion_derivatives;
            for (auto &derivatives : diffusion_deriv) {
              if (derivatives.capacity() < n_from) derivatives.reserve(n_from);
              derivatives.resize(n_from);
            }

            for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
              const auto dof_j = copy_data_face.from_dofs[j];
              const auto boundary_stencil_ad =
                  internal::tag_boundary_reconstruction_stencil_dofs<dim, NumberType, n_components>(boundary_stencil,
                                                                                                    dof_j);
              const auto cell_stencil_ad = tag_cell_stencil_dofs_from_cache(cell, cell_stencil, solution_global, dof_j);
              const auto [physical_stencil_ad, ghost_stencil_ad] =
                  internal::make_model_boundary_reconstruction_side_stencils(boundary_stencil_ad, cell_stencil_ad, x_q,
                                                                             model);
              reconstructed_deriv[0][j].u =
                  internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                      physical_stencil_ad.cell.u, physical_stencil_ad.cell.x, x_q, physical_stencil_ad.neighbors.x,
                      physical_stencil_ad.neighbors.u);
              reconstructed_deriv[1][j].u =
                  internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                      ghost_stencil_ad.cell.u, ghost_stencil_ad.cell.x, x_q, ghost_stencil_ad.neighbors.x,
                      ghost_stencil_ad.neighbors.u);
              reconstructed_deriv[0][j].grad =
                  JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                      physical_stencil_ad.cell.x, x_q, physical_stencil_ad.cell.u, physical_stencil_ad.neighbors.x,
                      physical_stencil_ad.neighbors.u);
              reconstructed_deriv[1][j].grad =
                  JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                      ghost_stencil_ad.cell.x, x_q, ghost_stencil_ad.cell.u, ghost_stencil_ad.neighbors.x,
                      ghost_stencil_ad.neighbors.u);

              if constexpr (dim == 1) {
                auto third_derivative_boundary_stencil_ad = boundary_stencil_ad.primary;
                internal::apply_boundary_reconstruction_stencil(third_derivative_boundary_stencil_ad, cell_stencil_ad,
                                                                model);
                const auto third_derivative_stencil_ad =
                    internal::make_boundary_third_derivative_stencil(third_derivative_boundary_stencil_ad);
                const auto third_derivatives =
                    JacobianReconstructor::template compute_third_derivatives_at_face_derivative<n_components>(
                        third_derivative_stencil_ad.x, third_derivative_stencil_ad.u);
                reconstructed_deriv[0][j].third_derivatives = third_derivatives;
                reconstructed_deriv[1][j].third_derivatives = third_derivatives;
              }

              const auto diffusion_face_ad =
                  internal::compute_diffusion_face_state(physical_stencil_ad, ghost_stencil_ad);
              const auto diffusion_face_derivatives =
                  internal::extract_diffusion_face_derivatives<dim, NumberType, n_components>(diffusion_face_ad);
              diffusion_deriv[0][j] = diffusion_face_derivatives[0];
              diffusion_deriv[1][j] = diffusion_face_derivatives[1];
            }

            // Compute numerical flux Jacobian
            const double width_minus = get_cell_topology(cell).cell_width;
            const double width_plus = width_minus;

            const auto j_numflux =
                internal::compute_kt_numflux_jacobian<WaveSpeedStrategy, Model, NumberType, dim, n_components>(
                    reconstruction.u_plus, reconstruction.u_minus, reconstruction.face_grad_plus,
                    reconstruction.face_grad_minus, x_q, width_plus, width_minus, __extracted_data, variables, model);
            const auto j_diffusion = internal::compute_diffusion_flux_jacobian<Model, NumberType, dim, n_components>(
                reconstruction.diffusion_u_minus, reconstruction.diffusion_u_plus, reconstruction.diffusion_grad_minus,
                reconstruction.diffusion_grad_plus, reconstruction.third_derivatives_minus,
                reconstruction.third_derivatives_plus, x_q, width_minus, width_plus, __extracted_data, variables,
                model);

            // Chain-rule assembly (same pattern as interior face_worker)
            for (uint i = 0; i < size(copy_data_face.to_dofs); ++i) {
              const auto component_i = i;
              for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
                NumberType diffusion_contribution{};
                for (size_t face_no = 0; face_no < 2; ++face_no) {
                  NumberType advection_contribution{};
                  for (size_t c = 0; c < n_components; ++c) {
                    advection_contribution += scalar_product(j_numflux.u[face_no](component_i, c), n_face) *
                                              reconstructed_deriv[face_no][j].u[c];
                    for (size_t d_in = 0; d_in < dim; ++d_in)
                      for (size_t d_out = 0; d_out < dim; ++d_out)
                        advection_contribution += j_numflux.grad[face_no](component_i, c)[d_out][d_in] * n_face[d_out] *
                                                  reconstructed_deriv[face_no][j].grad[c][d_in];
                  }
                  copy_data_face.cell_jacobian(i, j) += weight * JxW * advection_contribution;
                }

                for (size_t face_no = 0; face_no < 2; ++face_no)
                  for (size_t c = 0; c < n_components; ++c) {
                    diffusion_contribution += scalar_product(j_diffusion.u[face_no](component_i, c), n_face) *
                                              diffusion_deriv[face_no][j].u[c];
                    for (size_t d_in = 0; d_in < dim; ++d_in)
                      for (size_t d_out = 0; d_out < dim; ++d_out)
                        diffusion_contribution += j_diffusion.grad[face_no](component_i, c)[d_out][d_in] *
                                                  n_face[d_out] * diffusion_deriv[face_no][j].grad[c][d_in];
                    for (size_t d0 = 0; d0 < dim; ++d0)
                      for (size_t d1 = 0; d1 < dim; ++d1)
                        for (size_t d2 = 0; d2 < dim; ++d2)
                          for (size_t d_out = 0; d_out < dim; ++d_out)
                            diffusion_contribution +=
                                j_diffusion.third_derivatives[face_no](component_i, c)[d_out][d0][d1][d2] *
                                n_face[d_out] * reconstructed_deriv[face_no][j].third_derivatives[c][d0][d1][d2];
                  }

                // Boundary diffusion uses the same ghost-stencil side labels as the
                // advective boundary reconstruction, but it is differentiated as one
                // pair of corrected face-gradient operators.
                copy_data_face.cell_jacobian(i, j) += weight * JxW * diffusion_contribution;
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
                                            MeshWorker::assemble_own_interior_faces_once |
                                            MeshWorker::assemble_ghost_faces_once;

          // map() is collective and each rank visits only its own cells; see NoMapsHere.
          const NoMapsHere no_maps_during_assembly;
          const auto schedule = schedule_for(assembly_cost::momentum_integral);
          MeshWorker::mesh_loop(locally_owned_cells(dof_handler), cell_worker, copier, scratch_data, copy_data, flags,
                                boundary_worker, face_worker, schedule.queue_length, schedule.chunk_size);
          // Resolve contributions this rank made to rows it does not own. A partition-boundary
          // face is assembled by exactly one of its two neighbours (mesh_loop hands it to the
          // smaller subdomain id), and that rank writes BOTH sides -- so the other side's rows
          // arrive here. A no-op for the serial types.
          jacobian.compress(dealii::VectorOperation::add);
          timings_jacobian.push_back(timer.wall_time());
        }

        virtual void refinement_indicator([[maybe_unused]] Vector<double> &indicator,
                                          [[maybe_unused]] const VectorType &solution_global)
        {
        }

        template <typename DoFContainer>
        static void append_dofs(std::vector<types::global_dof_index> &target, const DoFContainer &source)
        {
          target.insert(target.end(), source.begin(), source.end());
        }

        template <typename DoFContainer>
        static void append_valid_dofs(std::vector<types::global_dof_index> &target, const DoFContainer &source)
        {
          for (const auto dof : source)
            if (dof != numbers::invalid_dof_index) target.push_back(dof);
        }

        static void sort_unique_dofs(std::vector<types::global_dof_index> &dofs)
        {
          std::sort(dofs.begin(), dofs.end());
          dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
        }

        void append_reconstruction_neighbor_dofs(std::vector<types::global_dof_index> &from_dofs,
                                                 const Iterator &root_cell) const
        {
          std::vector<types::global_dof_index> neighbor_dof_indices(fe.dofs_per_cell);

          const auto append_recursive = [&](const auto &self, const Iterator &cell, const unsigned int depth) -> void {
            for (const auto face_index : cell->face_indices()) {
              if (is_physical_boundary_face(cell, face_index)) continue;

              const auto neighbor = face_neighbor(cell, face_index);
              neighbor->get_dof_indices(neighbor_dof_indices);
              from_dofs.insert(from_dofs.end(), neighbor_dof_indices.begin(), neighbor_dof_indices.end());

              if (depth > 1) self(self, neighbor, depth - 1);
            }
          };

          append_recursive(append_recursive, root_cell, 2);
        }

        void append_boundary_reconstruction_dofs(std::vector<types::global_dof_index> &from_dofs,
                                                 const unsigned int cell_index, const unsigned int face_index) const
        {
          const auto append_boundary_stencil_dofs =
              [&](const internal::BoundaryStencilTopologyData<dim, n_components> &topology) {
                for (const auto &dofs : topology.dof_indices)
                  append_valid_dofs(from_dofs, dofs);
              };

          const auto &topology = cell_topology_cache[cell_index].boundary_stencils[face_index];
          append_boundary_stencil_dofs(topology.primary);
          for (size_t face = 0; face < topology.tangential_ghost_neighbor_valid.size(); ++face) {
            if (topology.tangential_ghost_neighbor_valid[face])
              append_boundary_stencil_dofs(topology.tangential_ghost_neighbors[face]);
            if (topology.corner_tangential_stencil_valid[face]) {
              for (const auto &corner_stencil : topology.corner_tangential_stencils[face])
                append_boundary_stencil_dofs(corner_stencil);
            }
          }
        }

        void build_face_jacobian_dependency_cache(const Iterator &cell, const unsigned int face_index,
                                                  FaceJacobianDependencyCacheEntry &dependencies) const
        {
          const auto &cell_topology = cell_topology_cache[cell->active_cell_index()].stencil;

          dependencies.to_dofs.clear();
          append_dofs(dependencies.to_dofs, cell_topology.cell.dof_indices);
          if (!is_physical_boundary_face(cell, face_index))
            append_dofs(dependencies.to_dofs, cell_topology.neighbors.dof_indices[face_index]);

          dependencies.from_dofs = dependencies.to_dofs;
          if (is_physical_boundary_face(cell, face_index))
            append_boundary_reconstruction_dofs(dependencies.from_dofs, cell->active_cell_index(), face_index);
          if constexpr (JacobianReconstructor::jacobian_stencil_radius > 1) {
            append_reconstruction_neighbor_dofs(dependencies.from_dofs, cell);
            if (!is_physical_boundary_face(cell, face_index))
              append_reconstruction_neighbor_dofs(dependencies.from_dofs, face_neighbor(cell, face_index));
          }
          sort_unique_dofs(dependencies.from_dofs);
        }

        /**
         * @brief The dofs the nonlocal part of the source jacobian writes to and reads from.
         *
         * Radius 1, tighter than the face dependencies: the gradient handed to model.source() is
         * reconstructed from the cell and its 2*dim face neighbours only. Across a physical boundary face
         * that neighbour is a model ghost, so the boundary stencil's dofs take its place.
         */
        void build_source_jacobian_dependency_cache(const Iterator &cell,
                                                    FaceJacobianDependencyCacheEntry &dependencies) const
        {
          const auto &cell_topology = cell_topology_cache[cell->active_cell_index()].stencil;

          dependencies.to_dofs.clear();
          append_dofs(dependencies.to_dofs, cell_topology.cell.dof_indices);

          dependencies.from_dofs = dependencies.to_dofs;
          for (const auto face_index : cell->face_indices()) {
            if (is_physical_boundary_face(cell, face_index))
              append_boundary_reconstruction_dofs(dependencies.from_dofs, cell->active_cell_index(), face_index);
            else
              append_valid_dofs(dependencies.from_dofs, cell_topology.neighbors.dof_indices[face_index]);
          }
          sort_unique_dofs(dependencies.from_dofs);
        }

        void build_sparsity(get_type::SparsityPattern<SparseMatrixType> &sparsity_pattern,
                            const DoFHandler<dim> &to_dofh, const DoFHandler<dim> &from_dofh, const int stencil = 2,
                            [[maybe_unused]] bool add_extractor_dofs = false) const
        {
          const auto &triangulation = discretization.get_triangulation();

          DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());

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
                    if (!is_physical_boundary_face(from_cell, face_no)) {
                      auto neighbor_cell = face_neighbor(from_cell, face_no);

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
          finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern, discretization.get_locally_owned_dofs(),
                                                 discretization.get_locally_relevant_dofs(),
                                                 discretization.get_communicator());
        }

        void build_cached_jacobian_sparsity(get_type::SparsityPattern<SparseMatrixType> &sparsity_pattern) const
        {
          DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            const auto &topology = get_cell_topology(cell);
            const auto &cell_dofs = topology.stencil.cell.dof_indices;
            for (const auto row : cell_dofs)
              for (const auto column : cell_dofs)
                dsp.add(row, column);

            const auto add_block = [&](const FaceJacobianDependencyCacheEntry &dependencies) {
              for (const auto row : dependencies.to_dofs)
                for (const auto column : dependencies.from_dofs)
                  if (row != numbers::invalid_dof_index && column != numbers::invalid_dof_index) dsp.add(row, column);
            };
            for (const auto face_index : cell->face_indices())
              add_block(topology.face_jacobian_dependencies[face_index]);
            // Subsumed by the face blocks in practice, but the source jacobian must not depend on that.
            add_block(topology.source_jacobian_dependencies);
          }
          finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern, discretization.get_locally_owned_dofs(),
                                                 discretization.get_locally_relevant_dofs(),
                                                 discretization.get_communicator());
        }

        void fill_boundary_topology(internal::BoundaryStencilTopologyData<dim, n_components> &boundary_topology,
                                    const Iterator &cell, const unsigned int boundary_face_no,
                                    std::vector<types::global_dof_index> &dof_indices) const
        {
          using namespace def::BoundaryStencilIndex;
          boundary_topology.lower_boundary = boundary_face_no % 2 == 0;
          boundary_topology.cell_face = boundary_face_no;
          boundary_topology.face_center = cell->face(boundary_face_no)->center();
          boundary_topology.ghost_center = boundary_topology.lower_boundary ? lower_inner : upper_inner;
          boundary_topology.ghost_left = boundary_topology.lower_boundary ? lower_outer : physical_cell;
          boundary_topology.ghost_right = boundary_topology.lower_boundary ? physical_cell : upper_outer;
          for (auto &dofs : boundary_topology.dof_indices)
            dofs.fill(numbers::invalid_dof_index);

          cell->get_dof_indices(dof_indices);
          boundary_topology.x[physical_cell] = cell->center();
          for (uint i = 0; i < n_components; ++i)
            boundary_topology.dof_indices[physical_cell][i] = dof_indices[i];

          const auto interior_face = GeometryInfo<dim>::opposite_face[boundary_face_no];
          AssertThrow(!is_physical_boundary_face(cell, interior_face),
                      ExcMessage("KT boundary stencil requires at least two interior cells behind the boundary face."));

          auto neighbor = face_neighbor(cell, interior_face);
          std::array<types::global_dof_index, n_components> first_interior_dofs{};
          neighbor->get_dof_indices(dof_indices);
          for (uint i = 0; i < n_components; ++i)
            first_interior_dofs[i] = dof_indices[i];

          AssertThrow(!is_physical_boundary_face(neighbor, interior_face),
                      ExcMessage("KT boundary stencil requires a second interior cell behind the boundary face."));
          auto next_neighbor = face_neighbor(neighbor, interior_face);
          std::array<types::global_dof_index, n_components> second_interior_dofs{};
          next_neighbor->get_dof_indices(dof_indices);
          for (uint i = 0; i < n_components; ++i)
            second_interior_dofs[i] = dof_indices[i];

          if (boundary_topology.lower_boundary) {
            boundary_topology.x[upper_inner] = neighbor->center();
            boundary_topology.x[upper_outer] = next_neighbor->center();
            boundary_topology.dof_indices[upper_inner] = first_interior_dofs;
            boundary_topology.dof_indices[upper_outer] = second_interior_dofs;
          } else {
            boundary_topology.x[lower_inner] = neighbor->center();
            boundary_topology.x[lower_outer] = next_neighbor->center();
            boundary_topology.dof_indices[lower_inner] = first_interior_dofs;
            boundary_topology.dof_indices[lower_outer] = second_interior_dofs;
          }
        }

        void rebuild_cell_topology_cache()
        {
          FEValues<dim> fe_values(mapping, fe, quadrature, update_quadrature_points | update_JxW_values);
          cell_topology_cache.clear();
          cell_topology_cache.resize(triangulation.n_active_cells());

          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          // Every cell, not just the owned ones: the stencil reaches two cells out, so an owned
          // cell at a partition boundary needs entries for cells it does not own.
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            auto &cache_entry = cell_topology_cache[cell->active_cell_index()];
            auto &stencil_topology = cache_entry.stencil;
            stencil_topology.neighbors = {};
            stencil_topology.boundary_ids.fill(numbers::invalid_boundary_id);
            stencil_topology.face_centers = {};
            for (auto &neighbor_dofs : stencil_topology.neighbors.dof_indices)
              neighbor_dofs.fill(numbers::invalid_dof_index);
            for (auto &boundary_reconstruction_topology : cache_entry.boundary_stencils) {
              boundary_reconstruction_topology.tangential_ghost_neighbor_valid.fill(false);
              boundary_reconstruction_topology.corner_tangential_stencil_valid.fill(false);
              for (auto &dofs : boundary_reconstruction_topology.primary.dof_indices)
                dofs.fill(numbers::invalid_dof_index);
              for (auto &topology : boundary_reconstruction_topology.tangential_ghost_neighbors)
                for (auto &dofs : topology.dof_indices)
                  dofs.fill(numbers::invalid_dof_index);
              for (auto &corner_stencils : boundary_reconstruction_topology.corner_tangential_stencils)
                for (auto &topology : corner_stencils)
                  for (auto &dofs : topology.dof_indices)
                    dofs.fill(numbers::invalid_dof_index);
            }

            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            stencil_topology.cell.x = cell->center();
            for (uint i = 0; i < n_components; ++i)
              stencil_topology.cell.dof_indices[i] = dof_indices[i];
            cache_entry.quadrature_points = fe_values.get_quadrature_points();
            cache_entry.jxw.assign(fe_values.get_JxW_values().begin(), fe_values.get_JxW_values().end());
            cache_entry.cell_width = DiFfRG::internal::cell_width(cell);

            for (const auto face_index : cell->face_indices()) {
              const auto face = cell->face(face_index);
              stencil_topology.face_centers[face_index] = face->center();
              if (is_physical_boundary_face(cell, face_index)) {
                stencil_topology.boundary_ids[face_index] = face->boundary_id();
                stencil_topology.neighbors.x[face_index] = face->center();
                auto &boundary_reconstruction_topology = cache_entry.boundary_stencils[face_index];
                fill_boundary_topology(boundary_reconstruction_topology.primary, cell, face_index, dof_indices);

                if constexpr (dim == 2) {
                  const unsigned int normal_axis = face_index / 2;
                  const unsigned int tangential_axis = 1U - normal_axis;
                  const unsigned int tangential_minus = 2 * tangential_axis;
                  const unsigned int tangential_plus = tangential_minus + 1;
                  const auto interior_face = GeometryInfo<dim>::opposite_face[face_index];
                  auto normal_neighbor = face_neighbor(cell, interior_face);
                  auto second_normal_neighbor = face_neighbor(normal_neighbor, interior_face);

                  for (const auto tangential_face : {tangential_minus, tangential_plus}) {
                    if (!is_physical_boundary_face(cell, tangential_face)) {
                      const auto tangential_neighbor = face_neighbor(cell, tangential_face);
                      if (is_physical_boundary_face(tangential_neighbor, face_index)) {
                        fill_boundary_topology(
                            boundary_reconstruction_topology.tangential_ghost_neighbors[tangential_face],
                            tangential_neighbor, face_index, dof_indices);
                        boundary_reconstruction_topology.tangential_ghost_neighbor_valid[tangential_face] = true;
                      }
                      continue;
                    }

                    fill_boundary_topology(
                        boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][0], cell,
                        tangential_face, dof_indices);
                    fill_boundary_topology(
                        boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][1],
                        normal_neighbor, tangential_face, dof_indices);
                    fill_boundary_topology(
                        boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][2],
                        second_normal_neighbor, tangential_face, dof_indices);
                    boundary_reconstruction_topology.corner_tangential_stencil_valid[tangential_face] = true;
                  }
                }
                continue;
              }

              const auto neighbor = face_neighbor(cell, face_index);
              neighbor->get_dof_indices(dof_indices);
              stencil_topology.neighbors.x[face_index] = neighbor->center();
              for (uint i = 0; i < n_components; ++i)
                stencil_topology.neighbors.dof_indices[face_index][i] = dof_indices[i];
            }

            for (const auto face_index : cell->face_indices())
              build_face_jacobian_dependency_cache(cell, face_index,
                                                   cache_entry.face_jacobian_dependencies[face_index]);
            build_source_jacobian_dependency_cache(cell, cache_entry.source_jacobian_dependencies);
          }
        }

        const CellTopologyCacheEntry &get_cell_topology(const Iterator &cell) const
        {
          const auto cell_index = cell->active_cell_index();
          AssertIndexRange(cell_index, cell_topology_cache.size());
          return cell_topology_cache[cell_index];
        }

        template <typename String>
          requires std::convertible_to<String, std::string>
        [[deprecated("Construct the assembler with output.log_port() and call log() instead")]] void log(String &&)
        {
          DiFfRG::internal::reject_named_assembler_log<String>();
        }

        void log()
        {
          std::stringstream ss;
          ss << "FV Assembler: " << std::endl;
          ss << "        Reinit: " << average_time_reinit() * 1000 << "ms (" << num_reinits() << ")" << std::endl;
          ss << "        Residual: " << average_time_residual_assembly() * 1000 << "ms (" << num_residuals() << ")"
             << std::endl;
          ss << "        Jacobian: " << average_time_jacobian_assembly() * 1000 << "ms (" << num_jacobians() << ")"
             << std::endl;
          log_port.info(ss.str());
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
        LogPort log_port;
        const DoFHandler<dim> &dof_handler;
        const Mapping<dim> &mapping;
        const Triangulation<dim> &triangulation;
        const FiniteElement<dim> &fe;

        /// @see FEMAssembler::schedule_for
        AssemblySchedule schedule_for(const double cost_ns) const
        {
          return make_assembly_schedule(n_owned_cells, DiFfRG::n_threads(), cost_ns, schedule_overrides);
        }

        /// @see FEMAssembler::update_assembly_schedules
        void update_assembly_schedules()
        {
          const uint n_owned = n_locally_owned_cells(discretization);
          const bool unchanged = n_owned == n_owned_cells;
          n_owned_cells = n_owned;
          if (unchanged) return;

          const uint threads = DiFfRG::n_threads();
          const auto cheap = schedule_for(assembly_cost::local_fe);
          const auto integral = schedule_for(assembly_cost::momentum_integral);
          log_port.info("FV: Assembling {} cells on {} threads -- {}x{} workers/cells for a cheap cell loop, "
                        "{}x{} for an integral one.",
                        n_owned_cells, threads, cheap.queue_length, cheap.chunk_size, integral.queue_length,
                        integral.chunk_size);
        }

        /// @see FEMAssembler::n_owned_cells
        uint n_owned_cells = 0;
        const AssemblyScheduleOverrides schedule_overrides;

        mutable Point EoM;
        mutable Iterator EoM_cell;
        Iterator old_EoM_cell;
        const Config::EoMConfig EoM_config;
        mutable std::optional<Point> EoM_minimum_guess;
        /// Mesh-dependent half of the potential reconstructions, built once and reused; see PotentialSystemCache.
        mutable DiFfRG::internal::PotentialSystemCache<dim, NumberType> potential_cache;

        const QGauss<dim> quadrature;
        const QGauss<dim - 1> quadrature_face;

        get_type::SparsityPattern<SparseMatrixType> sparsity_pattern_mass;
        get_type::SparsityPattern<SparseMatrixType> sparsity_pattern_jacobian;
        SparseMatrixType mass_matrix;

        std::vector<double> timings_reinit;
        std::vector<double> timings_residual;
        std::vector<double> timings_jacobian;
        std::array<unsigned int, n_components> local_component_of_dof{};
        std::vector<CellTopologyCacheEntry> cell_topology_cache;
        std::vector<FaceReconstructionDescriptor> face_reconstruction_descriptors;
        SolutionReconstructionCache residual_reconstruction_cache;
        SolutionReconstructionCache jacobian_reconstruction_cache;

        // Warn once per assembler when round-off from a gradient-independent flux baseline
        // exceeds this fraction of the physical flux variation; see
        // probe_diffusion_flux_conditioning(). 1e-9 sits comfortably below the tolerances any
        // fRG flow is run at, so a warning always means real digits are being lost.
        static constexpr double flux_conditioning_warn_threshold = 1e-9;
        static constexpr unsigned int flux_conditioning_max_samples = 512;
        // Set in reinit() so the collective that produces it is never behind a condition.
        double domain_diameter = 0.0;
        // Opt-in ("/discretization/diagnose_flux_conditioning"). The baseline/variation ratio
        // is a sound measure of digits lost in the flux difference, but on its own it cannot
        // tell a harmful case from a harmless one: a model may be baseline-dominated in the
        // deep UV while its diffusion term is still irrelevant to the flow, and pay nothing
        // for it. Enable this when a KT flow is inexplicably slow or stalls at tight
        // tolerances, and read the ratio as "digits available in the diffusion residual".
        const bool diagnose_flux_conditioning;
        mutable bool flux_conditioning_probed = false;
      };
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
