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
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <spdlog/spdlog.h>
#include <sstream>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FV/reconstructor/first_order_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/tvd_reconstructor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>

#include <DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh>
#include <DiFfRG/discretization/FV/assembler/flux_ties.hh>
#include <DiFfRG/discretization/FV/assembler/reconstruction_cache.hh>
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
        /**
         * @brief Class to hold data for each assembly thread, i.e. FEValues for cells, interfaces, as well as
         * pre-allocated data structures for the solutions
         */
        template <int dim, typename NumberType, size_t n_components> struct ScratchData {
          using QuadratureValue = std::array<NumberType, n_components>;

          ScratchData(const Quadrature<dim> &quadrature)
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

        template <typename NumberType> struct CopyData_ResidualDiagnostics {
          struct CopyDataFace_ResidualDiagnostics {
            Vector<NumberType> advection;
            Vector<NumberType> diffusion;
            std::vector<types::global_dof_index> joint_dof_indices;

            void reinit(const unsigned int n_face_dofs)
            {
              advection.reinit(n_face_dofs);
              diffusion.reinit(n_face_dofs);
              joint_dof_indices.resize(n_face_dofs);
            }
          };

          Vector<NumberType> advection;
          Vector<NumberType> diffusion;
          Vector<NumberType> source;
          Vector<NumberType> mass;
          std::vector<types::global_dof_index> local_dof_indices;
          std::vector<CopyDataFace_ResidualDiagnostics> face_data;
          unsigned int active_face_count = 0;

          template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell)
          {
            advection.reinit(dofs_per_cell);
            diffusion.reinit(dofs_per_cell);
            source.reinit(dofs_per_cell);
            mass.reinit(dofs_per_cell);
            local_dof_indices.resize(dofs_per_cell);
            if (face_data.size() != cell->n_faces()) face_data.resize(cell->n_faces());
            active_face_count = 0;
            cell->get_dof_indices(local_dof_indices);
          }

          CopyDataFace_ResidualDiagnostics &next_face_data()
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
          // 1. Compute flux Jacobians J (and Hessian H, if the wave-speed strategy
          // needs it) via either Real<2>-AD, Real<1>-AD + FD, or Real<1>-AD only,
          // depending on the strategy's `needs_hessian` and `hessian_via_fd` traits.
          //
          //   needs_hessian   hessian_via_fd   path used
          //   ------------    --------------   ---------
          //   false           (ignored)        compute_flux_jacobian_only          (Real<1>, H=0)
          //   true            false            compute_flux_jacobian_and_hessian   (Real<2>)
          //   true            true             compute_flux_jacobian_and_hessian_fd(Real<1> + central FD)
          auto compute_FJH = [&](const std::array<NumberType, n_components> &u) {
            if constexpr (!WaveSpeedStrategy::needs_hessian)
              return compute_flux_jacobian_only<Model, NumberType, dim, n_components>(u, x_q, model);
            else if constexpr (WaveSpeedStrategy::hessian_via_fd)
              return compute_flux_jacobian_and_hessian_fd<Model, NumberType, dim, n_components>(u, x_q, model);
            else
              return compute_flux_jacobian_and_hessian<Model, NumberType, dim, n_components>(u, x_q, model);
          };
          auto [F_plus, J_plus, H_plus] = compute_FJH(u_plus);
          auto [F_minus, J_minus, H_minus] = compute_FJH(u_minus);

          // 2. Compute a_half (max local wave speed per dimension) via the strategy
          const auto a = WaveSpeedStrategy::template compute_speeds<NumberType, dim, n_components>(J_plus, J_minus);

          // 3. Compute da_half / du_plus and da_half / du_minus for the selected wave-speed branch.
          const auto [da_plus, da_minus] =
              WaveSpeedStrategy::template compute_selected_speed_derivatives<NumberType, dim, n_components>(
                  J_plus, J_minus, H_plus, H_minus);

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

      } // namespace internal

#include <DiFfRG/discretization/FV/assembler/reconstruction_diagnostics.hh>

      template <typename Discretization_, typename Model_,
                def::HasReconstructor Reconstructor_ =
                    def::TVDReconstructor<Discretization_::dim, def::MinModLimiter, double>,
                def::HasWaveSpeed WaveSpeedStrategy_ = MaxEigenvalueWaveSpeed,
                def::HasReconstructor JacobianReconstructor_ = Reconstructor_>
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
        using JacobianReconstructor = JacobianReconstructor_;
        using NumberType = typename Discretization::NumberType;
        using VectorType = typename Discretization::VectorType;

        using Components = typename Discretization::Components;
        static constexpr uint dim = Discretization::dim;
        static_assert(Reconstructor::dim == dim, "Reconstructor dimension must match the discretization dimension.");
        static_assert(JacobianReconstructor::dim == dim,
                      "JacobianReconstructor dimension must match the discretization dimension.");
        static constexpr uint n_components = Components::count_fe_functions(0);
        static constexpr uint n_faces = GeometryInfo<dim>::faces_per_cell;
        // using CacheData = internal::Cache_Data<NumberType, dim, n_components>;
        using GradientType = internal::GradientType<dim, NumberType, n_components>;
        using Iterator = typename DoFHandler<Discretization::dim>::active_cell_iterator;
        using Point = dealii::Point<dim>;
        using Scratch = internal::ScratchData<dim, NumberType, n_components>;
        struct FaceJacobianDependencyCacheEntry {
          std::vector<types::global_dof_index> to_dofs;
          std::vector<types::global_dof_index> from_dofs;
        };
        struct CellTopologyCacheEntry {
          internal::CellStencilTopologyData<dim, n_components> stencil;
          std::array<internal::BoundaryStencilTopologyData<1, n_components>, n_faces> boundary_stencils_1d{};
          std::array<FaceJacobianDependencyCacheEntry, n_faces> face_jacobian_dependencies{};
          std::vector<Point> quadrature_points;
          std::vector<NumberType> jxw;
        };
        struct ResidualContributionDiagnostics {
          VectorType advection;
          VectorType diffusion;
          VectorType source;
          VectorType mass;
          VectorType total;
          VectorType total_minus_residual;
          bool has_residual = false;
        };

        Assembler(Discretization &discretization, Model &model, const JSONValue &json)
            : discretization(discretization), model(model), dof_handler(discretization.get_dof_handler()),
              mapping(discretization.get_mapping()), triangulation(discretization.get_triangulation()), json(json),
              fe(discretization.get_fe()), threads(json.get_uint("/discretization/threads")),
              batch_size(json.get_uint("/discretization/batch_size")),
              EoM_cell(*(dof_handler.active_cell_iterators().end())),
              old_EoM_cell(*(dof_handler.active_cell_iterators().end())),
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

          auto &fe_out = data_out.fe_output();
          fe_out.attach(dof_handler, solution, fe_function_names);
          if (dt_solution.size() > 0) fe_out.attach(dof_handler, dt_solution, fe_function_names_dot);
          if (residual.size() > 0) fe_out.attach(dof_handler, residual, fe_function_names_residual);

#ifdef H5CPP
          if (json.get_bool("/output/hdf5", true) && json.get_bool("/output/fv_reconstruction_diagnostics", false)) {
            if constexpr (dim != 1) {
              throw std::runtime_error(
                  "FV reconstruction diagnostics are currently implemented for 1D KT models only.");
            } else {
              diagnostics::write_reconstruction_maps<Assembler>(*this, data_out, dof_handler, model, solution);
            }
          }
          if (json.get_bool("/output/hdf5", true) &&
              json.get_bool("/output/fv_residual_contribution_diagnostics", false)) {
            if constexpr (dim != 1) {
              throw std::runtime_error(
                  "FV residual contribution diagnostics are currently implemented for 1D KT models only.");
            } else {
              const auto contributions = residual_contribution_diagnostics(solution, dt_solution, residual);
              diagnostics::write_residual_contribution_maps<Assembler>(data_out, dof_handler, contributions);
            }
          }
#endif

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

          constexpr uint stencil = JacobianReconstructor::jacobian_stencil_radius;
          build_sparsity(sparsity_pattern_jacobian, dof_handler, dof_handler, stencil, true);
          rebuild_cell_topology_cache();

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

        struct ReadoutSolution {
          std::array<NumberType, n_components> values{};
          GradientType gradients{};
          std::array<Tensor<2, dim, NumberType>, n_components> hessians{};
        };

        void readouts(DataOutput<dim, VectorType> &data_out, const VectorType &solution_global,
                      const VectorType &variables) const
        {
          auto helper = [&](auto EoMfun, auto outputter) {
            auto EoM_cell = this->EoM_cell;
            const auto EoM = get_fv_readout_point(EoM_cell, solution_global, EoMfun);
            this->EoM_cell = EoM_cell;

            auto solution = reconstruct_readout_solution(EoM_cell, solution_global, EoM);

            std::array<NumberType, Components::count_extractors()> extracted_data{{}};
            outputter(data_out, EoM,
                      e_tie(solution.values, solution.gradients, solution.hessians, extracted_data, variables));
          };
          model.readouts_multiple(helper, data_out);
        }

        ReadoutSolution reconstruct_readout_solution(const Iterator &cell, const VectorType &solution_global,
                                                     const Point &x) const
        {
          CellStencilData stencil;
          fill_cell_stencil(cell, solution_global, stencil);

          const auto gradients = Reconstructor::template compute_gradient<n_components>(
              stencil.cell.x, stencil.cell.u, stencil.neighbors.x, stencil.neighbors.u);

          ReadoutSolution solution;
          solution.values = internal::reconstruct_u(stencil.cell.u, stencil.cell.x, x, gradients);
          solution.gradients = Reconstructor::template compute_gradient_at_point<n_components>(
              stencil.cell.x, x, stencil.cell.u, stencil.neighbors.x, stencil.neighbors.u);
          return solution;
        }

        Iterator fallback_readout_cell() const
        {
          auto best_cell = dof_handler.begin_active();
          double best_distance = std::numeric_limits<double>::max();
          for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
            const auto distance = cell->center().distance(Point());
            if (distance < best_distance) {
              best_distance = distance;
              best_cell = cell;
            }
          }
          return best_cell;
        }

        template <typename EoMFUN> Point get_fv_readout_point(Iterator &EoM_cell, const VectorType &solution_global,
                                                             const EoMFUN &EoMfun) const
        {
          if constexpr (dim == 1) {
            const auto end_cell = dof_handler.active_cell_iterators().end();
            if (EoM_max_iter == 0) {
              EoM_cell = fallback_readout_cell();
              return EoM_cell->center();
            }

            auto evaluate = [&](const Iterator &cell, const Point &x) {
              const auto solution = reconstruct_readout_solution(cell, solution_global, x);
              return EoMfun(x, solution.values)[0];
            };

            auto find_in_cell = [&](const Iterator &cell, Point &EoM, double &EoM_value) {
              Point left = cell->face(0)->center();
              Point right = cell->face(1)->center();
              if (right[0] < left[0]) std::swap(left, right);

              double left_value = evaluate(cell, left);
              double right_value = evaluate(cell, right);
              if (std::abs(left_value) <= EoM_abs_tol) {
                EoM = left;
                EoM_value = left_value;
                return true;
              }
              if (std::abs(right_value) <= EoM_abs_tol) {
                EoM = right;
                EoM_value = right_value;
                return true;
              }
              if (left_value * right_value > 0.) return false;

              Point mid = (left + right) / 2.;
              double mid_value = evaluate(cell, mid);
              for (uint iteration = 0; iteration < EoM_max_iter; ++iteration) {
                if (std::abs(mid_value) <= EoM_abs_tol) {
                  EoM = mid;
                  EoM_value = mid_value;
                  return true;
                }

                if (left_value * mid_value <= 0.) {
                  right = mid;
                  right_value = mid_value;
                } else {
                  left = mid;
                  left_value = mid_value;
                }

                mid = (left + right) / 2.;
                mid_value = evaluate(cell, mid);
              }

              EoM = mid;
              EoM_value = mid_value;
              return true;
            };

            Point EoM;
            double EoM_value = 0.;
            if (EoM_cell != end_cell && find_in_cell(EoM_cell, EoM, EoM_value)) return EoM;

            Iterator best_cell = fallback_readout_cell();
            Point best_point = best_cell->center();
            double best_value = std::abs(evaluate(best_cell, best_point));
            for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
              if (find_in_cell(cell, EoM, EoM_value)) {
                EoM_cell = cell;
                return EoM;
              }

              const auto center = cell->center();
              const auto center_value = std::abs(evaluate(cell, center));
              if (center_value < best_value) {
                best_cell = cell;
                best_point = center;
                best_value = center_value;
              }
            }

            EoM_cell = best_cell;
            return best_point;
          } else {
            EoM_cell = fallback_readout_cell();
            return EoM_cell->center();
          }
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

          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, nullptr, nullptr, threads, batch_size);
        }

        ResidualContributionDiagnostics
        residual_contribution_diagnostics(const VectorType &solution_global, const VectorType &solution_global_dot,
                                          const VectorType &residual_reference = VectorType())
        {
          using CopyData = internal::CopyData_ResidualDiagnostics<NumberType>;
          const auto &constraints = discretization.get_constraints();

          ResidualContributionDiagnostics diagnostics;
          reinit_vector(diagnostics.advection);
          reinit_vector(diagnostics.diffusion);
          reinit_vector(diagnostics.source);
          reinit_vector(diagnostics.mass);
          reinit_vector(diagnostics.total);
          reinit_vector(diagnostics.total_minus_residual);

          VectorType zero_dot;
          const VectorType *dot = &solution_global_dot;
          if (solution_global_dot.size() == 0) {
            reinit_vector(zero_dot);
            zero_dot = 0.;
            dot = &zero_dot;
          }

          Scratch scratch_data(quadrature);
          CopyData copy_data;

          const auto reconstruction_cache = build_solution_reconstruction_cache<Reconstructor>(solution_global);

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_topology(cell);
            constexpr uint n_dofs = n_components;

            copy_data.reinit(cell, n_dofs);

            fill_constant_quadrature_values(cell, solution_global, *dot, scratch_data);

            std::array<NumberType, n_components> mass{};
            std::array<NumberType, n_components> source{};
            for (size_t q_index = 0; q_index < cell_geometry.quadrature_points.size(); ++q_index) {
              const auto &x_q = cell_geometry.quadrature_points[q_index];
              model.mass(mass, x_q, scratch_data.solution_values[q_index], scratch_data.solution_dot_values[q_index]);
              model.source(source, x_q, fv_tie(scratch_data.solution_values[q_index]));

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = local_component_of_dof[i];
                copy_data.mass(i) += cell_geometry.jxw[q_index] * mass[component_i];
                copy_data.source(i) += cell_geometry.jxw[q_index] * source[component_i];
              }
            }
          };

          const auto face_worker = [&](const Iterator &cell, const unsigned int &f,
                                       [[maybe_unused]] const unsigned int &sf, const Iterator &ncell,
                                       [[maybe_unused]] const unsigned int &nf,
                                       [[maybe_unused]] const unsigned int &nsf, [[maybe_unused]] Scratch &scratch_data,
                                       CopyData &copy_data) {
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

            const auto [F_plus, F_minus, a_half] =
                internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(reconstruction.u_plus, reconstruction.u_minus,
                                                                        x_q, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, reconstruction.u_plus,
                                                            reconstruction.u_minus);
            const auto D = internal::compute_diffusion_flux(reconstruction.u_plus, reconstruction.u_minus,
                                                            reconstruction.face_grad_plus,
                                                            reconstruction.face_grad_minus, x_q, model);

            for (uint component_i = 0; component_i < n_components; ++component_i) {
              copy_data_face.joint_dof_indices[component_i] = cell_data.dof_indices[component_i];
              copy_data_face.joint_dof_indices[n_components + component_i] = ncell_data.dof_indices[component_i];

              const auto advection_contribution = JxW * scalar_product(H[component_i], n_face);
              const auto diffusion_contribution = -JxW * scalar_product(D[component_i], n_face);
              copy_data_face.advection(component_i) += advection_contribution;
              copy_data_face.diffusion(component_i) += diffusion_contribution;
              copy_data_face.advection(n_components + component_i) -= advection_contribution;
              copy_data_face.diffusion(n_components + component_i) -= diffusion_contribution;
            }
          };

          const auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no,
                                           [[maybe_unused]] Scratch &scratch_data, CopyData &copy_data) {
            const uint n_face_dofs = n_components;

            auto &copy_data_face = copy_data.next_face_data();
            copy_data_face.reinit(n_face_dofs);

            const auto x_q = cell->face(face_no)->center();
            const auto JxW = face_jxw(cell, face_no);
            const auto n_bnd = face_normal_from_cell(cell, face_no);

            const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, face_no);
            const auto &cell_data = reconstruction_cache.cell_stencils[cell->active_cell_index()].cell;

            static_assert(dim == 1, "KT boundary-face assembly currently requires one-dimensional boundary stencils.");

            const auto [F_plus, F_minus, a_half] =
                internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(reconstruction.u_plus, reconstruction.u_minus,
                                                                        x_q, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, reconstruction.u_plus,
                                                            reconstruction.u_minus);
            const auto D = internal::compute_diffusion_flux(reconstruction.u_plus, reconstruction.u_minus,
                                                            reconstruction.face_grad_plus,
                                                            reconstruction.face_grad_minus, x_q, model);

            for (uint component_i = 0; component_i < n_components; ++component_i) {
              copy_data_face.joint_dof_indices[component_i] = cell_data.dof_indices[component_i];
              copy_data_face.advection(component_i) += JxW * scalar_product(H[component_i], n_bnd);
              copy_data_face.diffusion(component_i) -= JxW * scalar_product(D[component_i], n_bnd);
            }
          };

          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.source, c.local_dof_indices, diagnostics.source);
            constraints.distribute_local_to_global(c.mass, c.local_dof_indices, diagnostics.mass);
            for (unsigned int face_index = 0; face_index < c.active_face_count; ++face_index) {
              const auto &face_data = c.face_data[face_index];
              constraints.distribute_local_to_global(face_data.advection, face_data.joint_dof_indices,
                                                     diagnostics.advection);
              constraints.distribute_local_to_global(face_data.diffusion, face_data.joint_dof_indices,
                                                     diagnostics.diffusion);
            }
          };

          MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                            MeshWorker::assemble_own_interior_faces_once;

          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, boundary_worker, face_worker, threads, batch_size);

          for (types::global_dof_index i = 0; i < diagnostics.total.size(); ++i) {
            diagnostics.total[i] =
                diagnostics.advection[i] + diagnostics.diffusion[i] + diagnostics.source[i] + diagnostics.mass[i];
          }

          diagnostics.has_residual = residual_reference.size() == diagnostics.total.size();
          if (diagnostics.has_residual) {
            for (types::global_dof_index i = 0; i < diagnostics.total_minus_residual.size(); ++i)
              diagnostics.total_minus_residual[i] = diagnostics.total[i] - residual_reference[i];
          }

          return diagnostics;
        }

        using CellData = internal::CellData<dim, NumberType, n_components>;
        using NeighborData = internal::NeighborData<dim, NumberType, n_components>;
        using CellStencilData = internal::CellStencilData<dim, NumberType, n_components>;
        using CellTopologyData = internal::CellTopologyData<dim, n_components>;
        using CellStencilTopologyData = internal::CellStencilTopologyData<dim, n_components>;
        using FaceReconstructionState = internal::FaceReconstructionState<dim, NumberType, n_components>;
        using SolutionReconstructionCache = internal::SolutionReconstructionCache<dim, NumberType, n_components>;
        template <typename NT> using CellStencilDataT = internal::CellStencilData<dim, NT, n_components>;

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
                const bool boundary_supported = model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x,
                                                                             face->center());
                AssertThrow(
                    boundary_supported,
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
          requires(dim == 1)
        static internal::BoundaryStencilData<dim, BoundaryNumberType, n_components>
        build_boundary_stencil_1d(const Iterator &cell, const unsigned int boundary_face_no,
                                  const VectorType &solution_global,
                                  std::vector<types::global_dof_index> &scratch_dof_indices)
        {
          static_assert(dim == 1, "Paper-style boundary stencils currently support only dim=1.");

          using BoundaryIndex = internal::BoundaryStencilIndex<dim>;
          const auto interior_face = GeometryInfo<1>::opposite_face[boundary_face_no];
          AssertThrow(!cell->at_boundary(interior_face),
                      ExcMessage("KT boundary stencil requires at least two interior cells behind the boundary face."));

          internal::BoundaryStencilData<dim, BoundaryNumberType, n_components> boundary_stencil{};
          boundary_stencil.lower_boundary = boundary_face_no == 0;
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

          auto neighbor = cell->neighbor(interior_face);
          CellData first_interior;
          fill_cell_data(neighbor, solution_global, scratch_dof_indices, first_interior);

          AssertThrow(!neighbor->at_boundary(interior_face),
                      ExcMessage("KT boundary stencil requires a second interior cell behind the boundary face."));
          auto next_neighbor = neighbor->neighbor(interior_face);
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

        void fill_cell_data_from_topology(const CellTopologyData &topology, const VectorType &solution_global,
                                          CellData &data) const
        {
          internal::fill_cell_data_from_topology<dim, NumberType, n_components>(topology, solution_global, data);
        }

        template <int boundary_dim, typename BoundaryNumberType>
          requires(boundary_dim != 1)
        internal::BoundaryStencilData<boundary_dim, BoundaryNumberType, n_components>
        build_boundary_stencil_from_cache_impl(const Iterator &, const unsigned int, const VectorType &) const
        {
          static_assert(boundary_dim == 1, "Paper-style boundary stencils currently support only dim=1.");
        }

        template <int boundary_dim, typename BoundaryNumberType>
          requires(boundary_dim == 1)
        internal::BoundaryStencilData<boundary_dim, BoundaryNumberType, n_components>
        build_boundary_stencil_from_cache_impl(const Iterator &cell, const unsigned int boundary_face_no,
                                               const VectorType &solution_global) const
        {
          const auto &topology = get_cell_topology(cell).boundary_stencils_1d[boundary_face_no];
          return internal::fill_boundary_stencil_from_topology<BoundaryNumberType, boundary_dim, n_components>(
              topology, solution_global);
        }

        template <typename BoundaryNumberType>
        internal::BoundaryStencilData<dim, BoundaryNumberType, n_components>
        build_boundary_stencil_1d_from_cache(const Iterator &cell, const unsigned int boundary_face_no,
                                             const VectorType &solution_global) const
        {
          return build_boundary_stencil_from_cache_impl<dim, BoundaryNumberType>(cell, boundary_face_no,
                                                                                 solution_global);
        }

        void fill_cell_stencil(const Iterator &cell, const VectorType &solution_global, CellStencilData &stencil) const
        {
          const auto &topology = get_cell_topology(cell).stencil;

          stencil.boundary_ids = topology.boundary_ids;
          stencil.face_centers = topology.face_centers;
          fill_cell_data_from_topology(topology.cell, solution_global, stencil.cell);
          stencil.neighbors.x = topology.neighbors.x;
          stencil.neighbors.dof_indices = topology.neighbors.dof_indices;
          for (auto &neighbor_u : stencil.neighbors.u)
            neighbor_u = {};

          for (const auto face_index : cell->face_indices()) {
            if (cell->at_boundary(face_index)) {
              if constexpr (dim == 1) {
                auto boundary_stencil =
                    build_boundary_stencil_1d_from_cache<NumberType>(cell, face_index, solution_global);
                const bool boundary_supported =
                    model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x,
                                                 topology.face_centers[face_index]);
                AssertThrow(
                    boundary_supported,
                    ExcMessage("KT boundary stencil was rejected while populating a boundary-adjacent cell stencil."));

                stencil.neighbors.x[face_index] = boundary_stencil.x[boundary_stencil.ghost_center];
                stencil.neighbors.u[face_index] = boundary_stencil.u[boundary_stencil.ghost_center];
                stencil.neighbors.dof_indices[face_index] =
                    boundary_stencil.dof_indices[boundary_stencil.ghost_center];
              }
              continue;
            }

            for (unsigned int i = 0; i < n_components; ++i) {
              const auto dof = topology.neighbors.dof_indices[face_index][i];
              stencil.neighbors.u[face_index][i] = solution_global(dof);
            }
          }
        }

        unsigned int find_neighbor_face(const Iterator &cell, const Iterator &neighbor) const
        {
          for (const auto neighbor_face_index : neighbor->face_indices()) {
            if (neighbor->at_boundary(neighbor_face_index)) continue;
            if (neighbor->neighbor(neighbor_face_index) == cell) return neighbor_face_index;
          }
          AssertThrow(false, ExcMessage("Could not find reciprocal KT neighbor face."));
          return 0;
        }

        template <def::HasReconstructor ActiveReconstructor>
        SolutionReconstructionCache build_solution_reconstruction_cache(const VectorType &solution_global) const
        {
          SolutionReconstructionCache cache;
          cache.cell_stencils.resize(triangulation.n_active_cells());
          cache.face_reconstructions.resize(triangulation.n_active_cells());
          cache.face_reconstruction_valid.resize(triangulation.n_active_cells());
          for (auto &valid_faces : cache.face_reconstruction_valid)
            valid_faces.fill(false);

          for (const auto &cell : dof_handler.active_cell_iterators())
            fill_cell_stencil(cell, solution_global, cache.cell_stencils[cell->active_cell_index()]);

          for (const auto &cell : dof_handler.active_cell_iterators()) {
            const auto cell_index = cell->active_cell_index();
            for (const auto face_index : cell->face_indices()) {
              if (cache.face_reconstruction_valid[cell_index][face_index]) continue;

              const auto x_q = cell->face(face_index)->center();
              if (cell->at_boundary(face_index)) {
                auto boundary_stencil =
                    build_boundary_stencil_1d_from_cache<NumberType>(cell, face_index, solution_global);
                cache.face_reconstructions[cell_index][face_index] =
                    internal::compute_boundary_face_reconstruction_state<ActiveReconstructor>(
                        boundary_stencil, x_q, model);
                cache.face_reconstruction_valid[cell_index][face_index] = true;
                continue;
              }

              const auto neighbor = cell->neighbor(face_index);
              const auto neighbor_index = neighbor->active_cell_index();
              const auto neighbor_face_index = find_neighbor_face(cell, neighbor);

              const auto state = internal::compute_interior_face_reconstruction_state<ActiveReconstructor>(
                  cache.cell_stencils[cell_index], cache.cell_stencils[neighbor_index], x_q);
              cache.face_reconstructions[cell_index][face_index] = state;
              cache.face_reconstruction_valid[cell_index][face_index] = true;
              cache.face_reconstructions[neighbor_index][neighbor_face_index] =
                  internal::reverse_face_reconstruction(state);
              cache.face_reconstruction_valid[neighbor_index][neighbor_face_index] = true;
            }
          }

          return cache;
        }

        const FaceReconstructionState &
        get_cached_face_reconstruction(const SolutionReconstructionCache &cache, const Iterator &cell,
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
          auto boundary_stencil = build_boundary_stencil_1d_from_cache<NumberType>(cell, face_no, solution_global);
          return internal::compute_boundary_face_reconstruction_state<Reconstructor>(
              boundary_stencil, x_q, model);
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

        auto compute_boundary_jacobian_face_reconstruction_from_cache(const Iterator &cell,
                                                                      const unsigned int face_no,
                                                                      const VectorType &solution_global,
                                                                      const Point &x_q) const
        {
          auto boundary_stencil = build_boundary_stencil_1d_from_cache<NumberType>(cell, face_no, solution_global);
          return internal::compute_boundary_face_reconstruction_state<JacobianReconstructor>(
              boundary_stencil, x_q, model);
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

          Timer timer;
          const auto reconstruction_cache = build_solution_reconstruction_cache<Reconstructor>(solution_global);

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_topology(cell);
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

            const auto [F_plus, F_minus, a_half] =
                internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(reconstruction.u_plus, reconstruction.u_minus,
                                                                        x_q, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, reconstruction.u_plus,
                                                            reconstruction.u_minus);
            const auto D = internal::compute_diffusion_flux(reconstruction.u_plus, reconstruction.u_minus,
                                                            reconstruction.face_grad_plus,
                                                            reconstruction.face_grad_minus, x_q, model);

            // Sign convention: the face flux is (H + D)·n, i.e. the advection numerical
            // flux H (from KurganovTadmor_advection_flux) and the diffusion flux D (from
            // flux()) are SUMMED. Both model methods therefore return the physical flux
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

            static_assert(dim == 1, "KT boundary-face assembly currently requires one-dimensional boundary stencils.");

            const auto [F_plus, F_minus, a_half] =
                internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(reconstruction.u_plus, reconstruction.u_minus,
                                                                        x_q, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, reconstruction.u_plus,
                                                            reconstruction.u_minus);

            const auto D_bnd = internal::compute_diffusion_flux(reconstruction.u_plus, reconstruction.u_minus,
                                                                reconstruction.face_grad_plus,
                                                                reconstruction.face_grad_minus, x_q, model);

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
                                            MeshWorker::assemble_own_interior_faces_once;

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
          Timer timer;
          const auto reconstruction_cache = build_solution_reconstruction_cache<JacobianReconstructor>(solution_global);

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            const auto &cell_geometry = get_cell_topology(cell);
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
          const auto face_worker = [&](const Iterator &cell, const unsigned int &f,
                                       [[maybe_unused]] const unsigned int &sf, const Iterator &ncell,
                                       [[maybe_unused]] const unsigned int &nf,
                                       [[maybe_unused]] const unsigned int &nsf,
                                       Scratch &scratch_data, CopyData &copy_data) {
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

            for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
              const auto dof_j = copy_data_face.from_dofs[j];

              // face_no=0: d(u⁻)/d(u_j) — reconstruction from cell side
              {
                auto u_center_tagged = internal::tag_cell_dofs(cell_data, dof_j);
                auto u_n_tagged = internal::make_tagged_neighbors(cell_neighbors, dof_j);
                reconstructed_deriv[0][j].u =
                    internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                        u_center_tagged.u, cell_data.x, x_q, cell_neighbors.x, u_n_tagged.u);
                reconstructed_deriv[0][j].grad =
                    JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                        cell_data.x, x_q, u_center_tagged.u, cell_neighbors.x, u_n_tagged.u);
              }

              // face_no=1: d(u⁺)/d(u_j) — reconstruction from neighbor side
              {
                auto u_center_tagged = internal::tag_cell_dofs(ncell_data, dof_j);
                auto u_n_tagged = internal::make_tagged_neighbors(ncell_neighbors, dof_j);
                reconstructed_deriv[1][j].u =
                    internal::reconstruct_u_derivative<JacobianReconstructor, dim, NumberType, n_components>(
                        u_center_tagged.u, ncell_data.x, x_q, ncell_neighbors.x, u_n_tagged.u);
                reconstructed_deriv[1][j].grad =
                    JacobianReconstructor::template compute_gradient_at_point_derivative<n_components>(
                        ncell_data.x, x_q, u_center_tagged.u, ncell_neighbors.x, u_n_tagged.u);
              }
            }

            const auto j_numflux =
                internal::compute_kt_numflux_jacobian<WaveSpeedStrategy, Model, NumberType, dim, n_components>(
                    reconstruction.u_plus, reconstruction.u_minus, x_q, model);
            const auto j_diffusion = internal::compute_diffusion_flux_jacobian<Model, NumberType, dim, n_components>(
                reconstruction.u_plus, reconstruction.u_minus, reconstruction.face_grad_plus,
                reconstruction.face_grad_minus, x_q, model);

            for (uint i = 0; i < size(copy_data_face.to_dofs); ++i) {
              const bool cell_side_i = i < n_components;
              const auto component_i = cell_side_i ? i : i - n_components;
              const auto jump_i = cell_side_i ? NumberType(1.) : NumberType(-1.);
              for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
                NumberType diffusion_contribution{};
                for (size_t face_no = 0; face_no < 2; ++face_no) {
                  NumberType advection_contribution{};
                  for (size_t c = 0; c < n_components; ++c) {
                    advection_contribution += scalar_product(j_numflux[face_no](component_i, c), n_face) *
                                              reconstructed_deriv[face_no][j].u[c];

                    diffusion_contribution += scalar_product(j_diffusion.u[face_no](component_i, c), n_face) *
                                              reconstructed_deriv[face_no][j].u[c];
                    for (size_t d_in = 0; d_in < dim; ++d_in)
                      for (size_t d_out = 0; d_out < dim; ++d_out)
                        diffusion_contribution += j_diffusion.grad[face_no](component_i, c)[d_out][d_in] *
                                                  n_face[d_out] * reconstructed_deriv[face_no][j].grad[c][d_in];
                  }
                  copy_data_face.cell_jacobian(i, j) += weight * JxW * jump_i * advection_contribution;
                }

                // The residual uses [[phi_i]] * ((H + D) · n), so after summing the
                // contributions from both face traces into diffusion_contribution we
                // add the full diffusive chain-rule term once here.
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

            static_assert(dim == 1, "KT boundary-face Jacobians currently require one-dimensional boundary stencils.");
            const auto boundary_stencil =
                build_boundary_stencil_1d_from_cache<NumberType>(cell, face_no, solution_global);

            const auto &reconstruction = get_cached_face_reconstruction(reconstruction_cache, cell, face_no);

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
              internal::BoundaryStencilData<dim, autodiff::Real<1, NumberType>, n_components> boundary_stencil_ad{};
              boundary_stencil_ad.x = boundary_stencil.x;
              boundary_stencil_ad.u = u_stencil_tagged;
              boundary_stencil_ad.dof_indices = boundary_stencil.dof_indices;
              boundary_stencil_ad.lower_boundary = boundary_stencil.lower_boundary;
              boundary_stencil_ad.ghost_center = boundary_stencil.ghost_center;
              boundary_stencil_ad.ghost_left = boundary_stencil.ghost_left;
              boundary_stencil_ad.ghost_right = boundary_stencil.ghost_right;
              const bool boundary_supported_ad =
                  model.apply_boundary_stencil(boundary_stencil_ad.u, boundary_stencil_ad.x, x_q);
              AssertThrow(boundary_supported_ad,
                          ExcMessage("KT boundary stencil was rejected during AD reconstruction tracing."));
              const auto physical_stencil_ad =
                  internal::make_physical_boundary_side_stencil_1d<dim, autodiff::Real<1, NumberType>, n_components>(
                      boundary_stencil_ad);
              const auto ghost_stencil_ad =
                  internal::make_ghost_boundary_side_stencil_1d<dim, autodiff::Real<1, NumberType>, n_components>(
                      boundary_stencil_ad);
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
            }

            // Compute numerical flux Jacobian
            const auto j_numflux =
                internal::compute_kt_numflux_jacobian<WaveSpeedStrategy, Model, NumberType, dim, n_components>(
                    reconstruction.u_plus, reconstruction.u_minus, x_q, model);
            const auto j_diffusion = internal::compute_diffusion_flux_jacobian<Model, NumberType, dim, n_components>(
                reconstruction.u_plus, reconstruction.u_minus, reconstruction.face_grad_plus,
                reconstruction.face_grad_minus, x_q, model);

            // Chain-rule assembly (same pattern as interior face_worker)
            for (uint i = 0; i < size(copy_data_face.to_dofs); ++i) {
              const auto component_i = i;
              for (uint j = 0; j < size(copy_data_face.from_dofs); ++j) {
                NumberType diffusion_contribution{};
                for (size_t face_no = 0; face_no < 2; ++face_no) {
                  NumberType advection_contribution{};
                  for (size_t c = 0; c < n_components; ++c) {
                    advection_contribution += scalar_product(j_numflux[face_no](component_i, c), n_face) *
                                              reconstructed_deriv[face_no][j].u[c];

                    diffusion_contribution += scalar_product(j_diffusion.u[face_no](component_i, c), n_face) *
                                              reconstructed_deriv[face_no][j].u[c];
                    for (size_t d_in = 0; d_in < dim; ++d_in)
                      for (size_t d_out = 0; d_out < dim; ++d_out)
                        diffusion_contribution += j_diffusion.grad[face_no](component_i, c)[d_out][d_in] *
                                                  n_face[d_out] * reconstructed_deriv[face_no][j].grad[c][d_in];
                  }
                  copy_data_face.cell_jacobian(i, j) += weight * JxW * advection_contribution;
                }

                // Boundary faces use the same residual sign convention: [[phi_i]] *
                // ((H + D) · n). diffusion_contribution already contains both the
                // interior-side and ghost-side chain-rule pieces, so we add it
                // once after the face_no sum.
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
                                            MeshWorker::assemble_own_interior_faces_once;

          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, boundary_worker, face_worker, threads, batch_size);
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

        static void sort_unique_dofs(std::vector<types::global_dof_index> &dofs)
        {
          std::sort(dofs.begin(), dofs.end());
          dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
        }

        void append_reconstruction_neighbor_dofs(std::vector<types::global_dof_index> &from_dofs,
                                                 const Iterator &root_cell) const
        {
          std::vector<types::global_dof_index> neighbor_dof_indices(fe.dofs_per_cell);

          const auto append_recursive = [&](const auto &self, const Iterator &cell,
                                            const unsigned int depth) -> void {
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

        void build_face_jacobian_dependency_cache(const Iterator &cell, const unsigned int face_index,
                                                  FaceJacobianDependencyCacheEntry &dependencies) const
        {
          const auto &cell_topology = cell_topology_cache[cell->active_cell_index()].stencil;

          dependencies.to_dofs.clear();
          append_dofs(dependencies.to_dofs, cell_topology.cell.dof_indices);
          if (!cell->at_boundary(face_index))
            append_dofs(dependencies.to_dofs, cell_topology.neighbors.dof_indices[face_index]);

          dependencies.from_dofs = dependencies.to_dofs;
          if constexpr (JacobianReconstructor::jacobian_stencil_radius > 1) {
            append_reconstruction_neighbor_dofs(dependencies.from_dofs, cell);
            if (!cell->at_boundary(face_index))
              append_reconstruction_neighbor_dofs(dependencies.from_dofs, cell->neighbor(face_index));
          }
          sort_unique_dofs(dependencies.from_dofs);
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

        void rebuild_cell_topology_cache()
        {
          FEValues<dim> fe_values(mapping, fe, quadrature, update_quadrature_points | update_JxW_values);
          cell_topology_cache.clear();
          cell_topology_cache.resize(triangulation.n_active_cells());

          std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            auto &cache_entry = cell_topology_cache[cell->active_cell_index()];
            auto &stencil_topology = cache_entry.stencil;
            stencil_topology.neighbors = {};
            stencil_topology.boundary_ids.fill(numbers::invalid_boundary_id);
            stencil_topology.face_centers = {};
            for (auto &neighbor_dofs : stencil_topology.neighbors.dof_indices)
              neighbor_dofs.fill(numbers::invalid_dof_index);
            for (auto &boundary_topology : cache_entry.boundary_stencils_1d)
              for (auto &dofs : boundary_topology.dof_indices)
                dofs.fill(numbers::invalid_dof_index);

            fe_values.reinit(cell);
            cell->get_dof_indices(dof_indices);
            stencil_topology.cell.x = cell->center();
            for (uint i = 0; i < n_components; ++i)
              stencil_topology.cell.dof_indices[i] = dof_indices[i];
            cache_entry.quadrature_points = fe_values.get_quadrature_points();
            cache_entry.jxw.assign(fe_values.get_JxW_values().begin(), fe_values.get_JxW_values().end());

            for (const auto face_index : cell->face_indices()) {
              const auto face = cell->face(face_index);
              stencil_topology.face_centers[face_index] = face->center();
              if (cell->at_boundary(face_index)) {
                stencil_topology.boundary_ids[face_index] = face->boundary_id();
                if constexpr (dim == 1) {
                  using BoundaryIndex = internal::BoundaryStencilIndex<dim>;
                  auto &boundary_topology = cache_entry.boundary_stencils_1d[face_index];
                  boundary_topology.lower_boundary = face_index == 0;
                  boundary_topology.ghost_center =
                      boundary_topology.lower_boundary ? BoundaryIndex::lower_inner : BoundaryIndex::upper_inner;
                  boundary_topology.ghost_left =
                      boundary_topology.lower_boundary ? BoundaryIndex::lower_outer : BoundaryIndex::physical_cell;
                  boundary_topology.ghost_right =
                      boundary_topology.lower_boundary ? BoundaryIndex::physical_cell : BoundaryIndex::upper_outer;
                  boundary_topology.x[BoundaryIndex::physical_cell] = stencil_topology.cell.x;
                  boundary_topology.dof_indices[BoundaryIndex::physical_cell] = stencil_topology.cell.dof_indices;

                  const auto interior_face = GeometryInfo<1>::opposite_face[face_index];
                  AssertThrow(!cell->at_boundary(interior_face),
                              ExcMessage(
                                  "KT boundary stencil requires at least two interior cells behind the boundary face."));

                  auto neighbor = cell->neighbor(interior_face);
                  std::array<types::global_dof_index, n_components> first_interior_dofs{};
                  neighbor->get_dof_indices(dof_indices);
                  for (uint i = 0; i < n_components; ++i)
                    first_interior_dofs[i] = dof_indices[i];

                  AssertThrow(!neighbor->at_boundary(interior_face),
                              ExcMessage("KT boundary stencil requires a second interior cell behind the boundary face."));
                  auto next_neighbor = neighbor->neighbor(interior_face);
                  std::array<types::global_dof_index, n_components> second_interior_dofs{};
                  next_neighbor->get_dof_indices(dof_indices);
                  for (uint i = 0; i < n_components; ++i)
                    second_interior_dofs[i] = dof_indices[i];

                  if (boundary_topology.lower_boundary) {
                    boundary_topology.x[BoundaryIndex::upper_inner] = neighbor->center();
                    boundary_topology.x[BoundaryIndex::upper_outer] = next_neighbor->center();
                    boundary_topology.dof_indices[BoundaryIndex::upper_inner] = first_interior_dofs;
                    boundary_topology.dof_indices[BoundaryIndex::upper_outer] = second_interior_dofs;
                  } else {
                    boundary_topology.x[BoundaryIndex::lower_inner] = neighbor->center();
                    boundary_topology.x[BoundaryIndex::lower_outer] = next_neighbor->center();
                    boundary_topology.dof_indices[BoundaryIndex::lower_inner] = first_interior_dofs;
                    boundary_topology.dof_indices[BoundaryIndex::lower_outer] = second_interior_dofs;
                  }
                }
                continue;
              }

              const auto neighbor = cell->neighbor(face_index);
              neighbor->get_dof_indices(dof_indices);
              stencil_topology.neighbors.x[face_index] = neighbor->center();
              for (uint i = 0; i < n_components; ++i)
                stencil_topology.neighbors.dof_indices[face_index][i] = dof_indices[i];
            }

            for (const auto face_index : cell->face_indices())
              build_face_jacobian_dependency_cache(cell, face_index,
                                                   cache_entry.face_jacobian_dependencies[face_index]);
          }
        }

        const CellTopologyCacheEntry &get_cell_topology(const Iterator &cell) const
        {
          const auto cell_index = cell->active_cell_index();
          AssertIndexRange(cell_index, cell_topology_cache.size());
          return cell_topology_cache[cell_index];
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

        mutable Iterator EoM_cell;
        Iterator old_EoM_cell;
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
        std::vector<CellTopologyCacheEntry> cell_topology_cache;
      };
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
