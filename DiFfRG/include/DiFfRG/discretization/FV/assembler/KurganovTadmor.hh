#pragma once

// external libraries

// DiFfRG
#include <Eigen/Dense>
#include <array>
#include <autodiff/forward/real/real.hpp>
#include <cstddef>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_interface_values.h>
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
#include <petscvec.h>
#include <spdlog/spdlog.h>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>

#include <DiFfRG/discretization/common/types.hh>
#include <ranges>
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
        using GradientType = std::array<dealii::Tensor<1, dim, NumberType>, n_components>;

        template <typename... T> auto advection_flux_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions">>(std::tie(t...));
        }

        template <typename... T> auto flux_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives">>(std::tie(t...));
        }

        /**
         * @brief Class to hold data for each assembly thread, i.e. FEValues for cells, interfaces, as well as
         * pre-allocated data structures for the solutions
         */
        template <int dim, typename NumberType> struct ScratchData {
          using VectorType = Vector<NumberType>;

          ScratchData(const Mapping<dim> &mapping, const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
                      const Quadrature<dim - 1> &quadrature_face,
                      const UpdateFlags update_flags = update_values | update_gradients | update_quadrature_points |
                                                       update_JxW_values,
                      const UpdateFlags interface_update_flags = update_values | update_gradients |
                                                                 update_quadrature_points | update_JxW_values |
                                                                 update_normal_vectors)
              : fe_values(mapping, fe, quadrature, update_flags),
                fe_interface_values(mapping, fe, quadrature_face, interface_update_flags)
          {
          }

          ScratchData(const ScratchData<dim, NumberType> &scratch_data)
              : fe_values(scratch_data.fe_values.get_mapping(), scratch_data.fe_values.get_fe(),
                          scratch_data.fe_values.get_quadrature(), scratch_data.fe_values.get_update_flags()),
                fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                                    scratch_data.fe_interface_values.get_fe(),
                                    scratch_data.fe_interface_values.get_quadrature(),
                                    scratch_data.fe_interface_values.get_update_flags())

          {
          }

          FEValues<dim> fe_values;
          FEInterfaceValues<dim> fe_interface_values;
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

        // TODO fewer memory allocations
        template <typename NumberType> struct CopyData_R {
          struct CopyDataFace_R {

            Vector<NumberType> cell_residual;
            std::vector<types::global_dof_index> joint_dof_indices;

            template <int dim> void reinit(const FEInterfaceValues<dim> &fe_iv)
            {
              cell_residual.reinit(fe_iv.n_current_interface_dofs());
              joint_dof_indices = fe_iv.get_interface_dof_indices();
            }
          };

          Vector<NumberType> cell_residual;
          Vector<NumberType> cell_mass;
          std::vector<types::global_dof_index> local_dof_indices;
          std::vector<CopyDataFace_R> face_data;

          template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell)
          {
            cell_residual.reinit(dofs_per_cell);
            cell_mass.reinit(dofs_per_cell);
            local_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);
          }
        };

        // TODO fewer memory allocations
        template <typename NumberType> struct CopyData_J {
          struct CopyDataFace_J {
            FullMatrix<NumberType> cell_jacobian;
            FullMatrix<NumberType> extractor_cell_jacobian;
            std::vector<types::global_dof_index> joint_dof_indices;

            template <int dim> void reinit(const FEInterfaceValues<dim> &fe_iv, uint n_extractors)
            {
              uint dofs_per_cell = fe_iv.n_current_interface_dofs();
              cell_jacobian.reinit(dofs_per_cell, dofs_per_cell);
              if (n_extractors > 0) extractor_cell_jacobian.reinit(dofs_per_cell, n_extractors);
              joint_dof_indices = fe_iv.get_interface_dof_indices();
            }
          };

          FullMatrix<NumberType> cell_jacobian;
          FullMatrix<NumberType> extractor_cell_jacobian;
          FullMatrix<NumberType> cell_mass_jacobian;
          std::vector<types::global_dof_index> local_dof_indices;
          std::vector<CopyDataFace_J> face_data;

          template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell, uint n_extractors)
          {
            cell_jacobian.reinit(dofs_per_cell, dofs_per_cell);
            if (n_extractors > 0) extractor_cell_jacobian.reinit(dofs_per_cell, n_extractors);
            cell_mass_jacobian.reinit(dofs_per_cell, dofs_per_cell);
            local_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);
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
         * @brief This function computes the gradient of u using a minmod limiter
         *
         * The gradient computed in this function fulfills the Total Variation Diminishing Principle.
         *
         * @tparam NumberType the numeric type used for the computation
         * @tparam dim the spatial dimension
         * @tparam n_components the number of components in u
         *
         * @param center_pos the position of the cell center where the gradient is computed
         * @param u_center the value of u at the cell center
         * @param x_n the positions of the neighboring cell centers. It is assumed that the ordering is such that the
         * first two entries correspond to the neighbors in the first dimension, the next two entries to the second
         * dimension, and so on.
         * @param u_n the values of u at the neighboring cell centers, ordered in the same way as x_n.
         *
         * @return GradientType<dim, NumberType, n_components> the computed gradient.
         */
        template <typename NumberType, int dim, int n_components>
        GradientType<dim, NumberType, n_components>
        compute_gradient(const Point<dim> &center_pos, const std::array<NumberType, n_components> &u_center,
                         const std::array<Point<dim>, 2 * dim> &x_n,
                         const std::array<std::array<NumberType, n_components>, 2 * dim> &u_n)
        {
          GradientType<dim, NumberType, n_components> u_grad{};
          for (size_t c = 0; c < n_components; c++) {
            const NumberType &u_val = u_center[c];
            for (int d = 0, i_n_1 = 0, i_n_2 = 1; d < dim; d++, i_n_1 += 2, i_n_2 += 2) {

              const auto &u_n_1 = u_n[i_n_1][c];
              const auto &u_n_2 = u_n[i_n_2][c];

              const auto dx_1 = x_n[i_n_1] - center_pos;
              const NumberType du_1 = (u_n_1 - u_val) / dx_1[d];
              const auto dx_2 = x_n[i_n_2] - center_pos;
              const NumberType du_2 = (u_n_2 - u_val) / dx_2[d];

              const NumberType du_smaller = 0.5 * (sgn(du_1) + sgn(du_2)) * std::min(std::abs(du_1), std::abs(du_2));

              u_grad[c][d] = du_smaller;
            }
          }

          return u_grad;
        }

        /**
         * @brief Result struct for compute_kt_flux_and_speeds.
         */
        template <int dim, typename NumberType, size_t n_components> struct KTFluxData {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F_plus;
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F_minus;
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> a_half;
        };

        /**
         * @brief Compute the advection fluxes and local wave speeds for the Kurganov-Tadmor scheme using
         * automatic differentiation.
         *
         * @tparam Model the model type providing KurganovTadmor_advection_flux
         * @tparam NumberType the numeric type
         * @tparam dim the spatial dimension
         * @tparam n_components the number of solution components
         *
         * @param u_plus the reconstructed state on the "+" side of the interface
         * @param u_minus the reconstructed state on the "-" side of the interface
         * @param x_q the quadrature point position at the face
         * @param model the model providing the advection flux function
         *
         * @return KTFluxData containing F_plus, F_minus, and the local wave speeds a_half.
         */
        template <typename Model, typename NumberType, int dim, size_t n_components>
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

          using JacobianMatrix = std::array<std::array<NumberType, n_components>, n_components>;
          std::array<JacobianMatrix, dim> J_plus{}, J_minus{};

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

          for (size_t d = 0; d < dim; ++d) {
            NumberType max_eig_plus = 0.0;
            NumberType max_eig_minus = 0.0;

            if constexpr (n_components == 1) {
              max_eig_plus = std::abs(J_plus[d][0][0]);
              max_eig_minus = std::abs(J_minus[d][0][0]);
            } else {
              Eigen::Matrix<NumberType, n_components, n_components> J_plus_eigen, J_minus_eigen;
              for (size_t i = 0; i < n_components; ++i)
                for (size_t j = 0; j < n_components; ++j) {
                  J_plus_eigen(i, j) = J_plus[d][i][j];
                  J_minus_eigen(i, j) = J_minus[d][i][j];
                }
              Eigen::EigenSolver<Eigen::Matrix<NumberType, n_components, n_components>> es_plus(J_plus_eigen);
              Eigen::EigenSolver<Eigen::Matrix<NumberType, n_components, n_components>> es_minus(J_minus_eigen);
              max_eig_plus = es_plus.eigenvalues().cwiseAbs().maxCoeff();
              max_eig_minus = es_minus.eigenvalues().cwiseAbs().maxCoeff();
            }

            NumberType a = std::max(max_eig_plus, max_eig_minus);

            for (size_t component : std::views::iota(0u, n_components)) {
              result.a_half[component][d] = a;
            }
          }

          return result;
        }

        /**
         * @brief Compute the Kurganov-Tadmor numerical flux from the left/right fluxes and wave speeds.
         *
         * H = 0.5 * (F_plus + F_minus) - 0.5 * a_half * (u_plus - u_minus)
         *
         * @tparam dim the spatial dimension
         * @tparam NumberType the numeric type
         * @tparam n_components the number of solution components
         *
         * @param F_plus flux evaluated at the "+" state
         * @param F_minus flux evaluated at the "-" state
         * @param a_half local wave speeds per component
         * @param u_plus the "+" state values
         * @param u_minus the "-" state values
         *
         * @return the numerical flux per component as an array of Tensor<1,dim>.
         */
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
        /**
         * @brief Compute the averaged diffusion flux at a face for the Kurganov-Tadmor scheme.
         *
         * Evaluates model.flux on both the left (u_minus) and right (u_plus) reconstructed states
         * using a two-point gradient approximation, then returns their average:
         * D = 0.5 * (flux(u_minus, grad) + flux(u_plus, grad))
         *
         * @tparam Model the model type providing the flux method
         * @tparam NumberType the numeric type
         * @tparam dim the spatial dimension
         * @tparam n_components the number of solution components
         *
         * @param u_plus the reconstructed state on the "+" side of the interface
         * @param u_minus the reconstructed state on the "-" side of the interface
         * @param x_left the center of the left cell
         * @param x_right the center of the right cell (or ghost position)
         * @param u_left the cell-average value of the left cell
         * @param u_right the cell-average value of the right cell (or ghost value)
         * @param normal the outward unit normal at the face
         * @param x_q the quadrature point position at the face
         * @param model the model providing the flux function
         *
         * @return the averaged diffusion flux per component as an array of Tensor<1,dim>.
         */
        template <typename Model, typename NumberType, int dim, size_t n_components>
        std::array<dealii::Tensor<1, dim, NumberType>, n_components>
        compute_diffusion_flux(const std::array<NumberType, n_components> &u_plus,
                               const std::array<NumberType, n_components> &u_minus, const dealii::Point<dim> &x_left,
                               const dealii::Point<dim> &x_right, const std::array<NumberType, n_components> &u_left,
                               const std::array<NumberType, n_components> &u_right,
                               const dealii::Tensor<1, dim> &normal, const dealii::Point<dim> &x_q, const Model &model)
        {
          const dealii::Tensor<1, dim> dx_vec = x_right - x_left;
          const NumberType dx_n = scalar_product(dx_vec, normal);

          GradientType<dim, NumberType, n_components> face_gradient{};
          for (size_t c = 0; c < n_components; ++c)
            face_gradient[c] = normal * ((u_right[c] - u_left[c]) / dx_n);

          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D_minus{}, D_plus{};
          model.flux(D_minus, x_q, flux_tie(u_minus, face_gradient));
          model.flux(D_plus, x_q, flux_tie(u_plus, face_gradient));

          std::array<dealii::Tensor<1, dim, NumberType>, n_components> D{};
          for (size_t c = 0; c < n_components; ++c)
            D[c] = 0.5 * (D_minus[c] + D_plus[c]);
          return D;
        }
      } // namespace internal

      template <typename Discretization_, typename Model_>
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
        using NumberType = typename Discretization::NumberType;
        using VectorType = typename Discretization::VectorType;

        using Components = typename Discretization::Components;
        static constexpr uint dim = Discretization::dim;
        static constexpr uint n_components = Components::count_fe_functions(0);
        static constexpr uint n_faces = GeometryInfo<dim>::faces_per_cell;
        // using CacheData = internal::Cache_Data<NumberType, dim, n_components>;
        using GradientType = internal::GradientType<dim, NumberType, n_components>;
        using Iterator = typename DoFHandler<Discretization::dim>::active_cell_iterator;
        using Point = dealii::Point<dim>;

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

          reinit();
        }

        virtual void reinit_vector(VectorType &vec) const override
        {
          const auto block_structure = discretization.get_block_structure();
          vec.reinit(block_structure[0]);
        }

        virtual IndexSet get_differential_indices() const override { return IndexSet(); }

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

          // Mass sparsity pattern
          {
            DynamicSparsityPattern dsp(dof_handler.n_dofs());
            DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                            /*keep_constrained_dofs = */ true);
            sparsity_pattern_mass.copy_from(dsp);
            mass_matrix.reinit(sparsity_pattern_mass);
            MatrixCreator::create_mass_matrix(dof_handler, quadrature, mass_matrix,
                                              static_cast<Function<dim, NumberType> *>(nullptr),
                                              discretization.get_constraints());
          }
          // // Jacobian sparsity pattern
          // {
          //   DynamicSparsityPattern dsp(dof_handler.n_dofs());
          //   DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
          //                                        /*keep_constrained_dofs = */ true);
          //   sparsity_pattern_jacobian.copy_from(dsp);
          // }

          constexpr uint stencil = 1;
          build_sparsity(sparsity_pattern_jacobian, dof_handler, dof_handler, stencil, true);

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

        virtual void jacobian_variables(FullMatrix<NumberType> &jacobian, const VectorType &variables,
                                        const VectorType &) override {
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
          using Scratch = internal::ScratchData<dim, NumberType>;
          using CopyData = internal::CopyData_R<NumberType>;
          const auto &constraints = discretization.get_constraints();

          Scratch scratch_data(mapping, discretization.get_fe(), quadrature, quadrature_face);
          CopyData copy_data;

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            scratch_data.fe_values.reinit(cell);
            const auto &fe_v = scratch_data.fe_values;
            const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();

            copy_data.reinit(cell, n_dofs);

            const auto &JxW = fe_v.get_JxW_values();
            const auto &q_points = fe_v.get_quadrature_points();
            const auto &q_indices = fe_v.quadrature_point_indices();

            std::vector<VectorType> solution(quadrature.size(), VectorType(n_components));
            std::vector<VectorType> solution_dot(quadrature.size(), VectorType(n_components));

            fe_v.get_function_values(solution_global, solution);
            fe_v.get_function_values(solution_global_dot, solution_dot);

            std::array<NumberType, n_components> mass_values{};
            for (const auto &q_index : q_indices) {
              const auto &x_q = q_points[q_index];
              model.mass(mass_values, x_q, solution[q_index], solution_dot[q_index]);

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = fe_v.get_fe().system_to_component_index(i).first;
                copy_data.cell_mass(i) += weight * JxW[q_index] * fe_v.shape_value_component(i, q_index, component_i) *
                                          mass_values[component_i]; // +phi_i(x_q) * mass(x_q, u_q)
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

        static std::pair<Point, std::array<NumberType, n_components>> get_cell_value(const Iterator &cell,
                                                                                     const VectorType &solution_global)
        {
          Point x_cell = cell->center();
          std::array<NumberType, n_components> u_cell;
          std::vector<types::global_dof_index> global_cell_dof_indices(n_components);
          cell->get_dof_indices(global_cell_dof_indices);
          for (unsigned int i = 0; i < n_components; ++i)
            u_cell[i] = solution_global(global_cell_dof_indices[i]);

          return std::make_pair(x_cell, u_cell);
        }

        static std::pair<std::array<Point, n_faces>, std::array<std::array<NumberType, n_components>, n_faces>>
        get_neighboring_cell_data(const Iterator &cell, const VectorType &solution_global, const Model &model)
        {
          std::array<Point, n_faces> x_n;
          std::array<std::array<NumberType, n_components>, n_faces> u_n;

          std::array<types::boundary_id, n_faces> boundary_ids;
          std::array<Point, n_faces> face_centers;

          for (const auto face_index : cell->face_indices()) {
            if (cell->at_boundary(face_index)) {
              const auto face = cell->face(face_index);
              boundary_ids[face_index] = face->boundary_id();
              face_centers[face_index] = face->center();
              // u_n/x_n will be computed by model
            } else {
              boundary_ids[face_index] = numbers::invalid_boundary_id;
              const auto neighbor = cell->neighbor(face_index);
              auto [x, u] = get_cell_value(neighbor, solution_global);
              x_n[face_index] = x;
              u_n[face_index] = u;
            }
          }

          auto [x_cell, u_cell] = get_cell_value(cell, solution_global);

          model.apply_boundary_conditions(u_n, x_n, boundary_ids, face_centers, u_cell, x_cell);

          return std::make_pair(x_n, u_n);
        }

        virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                              const VectorType &solution_global_dot, NumberType weight_mass,
                              const VectorType & /* variables */ = VectorType()) override
        {
          using Scratch = internal::ScratchData<dim, NumberType>;
          using CopyData = internal::CopyData_R<NumberType>;
          const auto &constraints = discretization.get_constraints();

          Scratch scratch_data(mapping, discretization.get_fe(), quadrature, quadrature_face);
          CopyData copy_data;

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            scratch_data.fe_values.reinit(cell);
            const auto &fe_v = scratch_data.fe_values;
            const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();

            copy_data.reinit(cell, n_dofs);

            const auto &JxW = fe_v.get_JxW_values();

            const auto &q_points = fe_v.get_quadrature_points();
            const auto &q_indices = fe_v.quadrature_point_indices();

            std::vector<VectorType> solution(quadrature.size(), VectorType(n_components));
            std::vector<VectorType> solution_dot(quadrature.size(), VectorType(n_components));

            fe_v.get_function_values(solution_global, solution);
            fe_v.get_function_values(solution_global_dot, solution_dot);

            std::array<NumberType, n_components> mass{};
            std::array<NumberType, n_components> source{};
            for (const auto &q_index : q_indices) {
              const auto &x_q = q_points[q_index];
              model.mass(mass, x_q, solution[q_index], solution_dot[q_index]);
              model.source(source, x_q, fv_tie(solution[q_index]));

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = fe_v.get_fe().system_to_component_index(i).first;
                copy_data.cell_mass(i) += weight_mass * JxW[q_index] *
                                          fe_v.shape_value_component(i, q_index, component_i) *
                                          mass[component_i];          // +phi_i(x_q) * mass(x_q, u_q)
                copy_data.cell_residual(i) += JxW[q_index] * weight * // dx *
                                              (fe_v.shape_value_component(i, q_index, component_i) *
                                               source[component_i]); // -phi_i(x_q) * source(x_q, u_q)
              }
            }
          };

          const auto face_worker = [&](const Iterator &cell, const unsigned int &f, const unsigned int &sf,
                                       const Iterator &ncell, const unsigned int &nf, const unsigned int &nsf,
                                       Scratch &scratch_data, CopyData &copy_data) {
            scratch_data.fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);
            const FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;

            const auto &face_q_points = fe_iv.get_quadrature_points();
            const int q_face_index = 0; // only one quadrature point per face for FV (constant FE)
            const auto &x_q = face_q_points[q_face_index];

            const uint n_face_dofs = fe_iv.n_current_interface_dofs();

            copy_data.face_data.emplace_back();
            auto &copy_data_face = copy_data.face_data.back();
            copy_data_face.reinit(fe_iv);

            auto [x_cell_n, u_cell_n] = get_neighboring_cell_data(cell, solution_global, model);
            auto [x_ncell_n, u_ncell_n] = get_neighboring_cell_data(ncell, solution_global, model);
            auto [x_cell, u_cell] = get_cell_value(cell, solution_global);
            auto [x_ncell, u_ncell] = get_cell_value(ncell, solution_global);

            const GradientType u_grad_cell =
                internal::compute_gradient<NumberType, dim, n_components>(x_cell, u_cell, x_cell_n, u_cell_n);
            const GradientType u_grad_ncell =
                internal::compute_gradient<NumberType, dim, n_components>(x_ncell, u_ncell, x_ncell_n, u_ncell_n);

            const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

            const std::array<NumberType, n_components> u_minus =
                internal::reconstruct_u(u_cell, cell->center(), x_q, u_grad_cell);
            const std::array<NumberType, n_components> u_plus =
                internal::reconstruct_u(u_ncell, ncell->center(), x_q, u_grad_ncell);

            const auto [F_plus, F_minus, a_half] = internal::compute_kt_flux_and_speeds(u_plus, u_minus, x_q, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);

            const auto &n_face = normals[q_face_index];
            const auto D =
                internal::compute_diffusion_flux(u_plus, u_minus, x_cell, x_ncell, u_cell, u_ncell, n_face, x_q, model);

            const auto JxW = fe_iv.get_JxW_values();

            for (uint dof = 0; dof < n_face_dofs; ++dof) {
              const auto &cd_i = fe_iv.interface_dof_to_dof_indices(dof);
              const auto component_i = cd_i[0] == numbers::invalid_unsigned_int
                                           ? fe.system_to_component_index(cd_i[1]).first
                                           : fe.system_to_component_index(cd_i[0]).first;
              copy_data_face.cell_residual(dof) +=
                  weight * JxW[q_face_index] *
                  (fe_iv.jump_in_shape_values(dof, q_face_index, component_i) *
                   (scalar_product(H[component_i], n_face) -
                    scalar_product(D[component_i], n_face))); // [[phi_i]] * (H - D) · n
            }
          };

          const auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no, Scratch &scratch_data,
                                           CopyData &copy_data) {
            scratch_data.fe_interface_values.reinit(cell, face_no);
            const auto &fe_fv = scratch_data.fe_interface_values.get_fe_face_values(0);
            const FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
            const uint n_face_dofs = fe_iv.n_current_interface_dofs();

            copy_data.face_data.emplace_back();
            auto &copy_data_face = copy_data.face_data.back();
            copy_data_face.reinit(fe_iv);

            const auto &JxW = fe_fv.get_JxW_values();
            const auto &q_points = fe_fv.get_quadrature_points();
            const int q_face_index = 0; // only one quadrature point per face for FV
            const auto &x_q = q_points[q_face_index];

            // Get cell data and ghost neighbor data via apply_boundary_conditions
            auto [x_cell_n, u_cell_n] = get_neighboring_cell_data(cell, solution_global, model);
            auto [x_cell, u_cell] = get_cell_value(cell, solution_global);

            const GradientType u_grad_cell =
                internal::compute_gradient<NumberType, dim, n_components>(x_cell, u_cell, x_cell_n, u_cell_n);

            const std::vector<Tensor<1, dim>> &normals = fe_fv.get_normal_vectors();

            // Interior reconstructed state at face
            const std::array<NumberType, n_components> u_minus =
                internal::reconstruct_u(u_cell, cell->center(), x_q, u_grad_cell);

            // Ghost cell value and position from apply_boundary_conditions
            const std::array<NumberType, n_components> &u_ghost = u_cell_n[face_no];
            const Point &x_ghost = x_cell_n[face_no];

            // Ghost gradient via model interface (allows second-order reconstruction at boundaries)
            const auto boundary_id = cell->face(face_no)->boundary_id();
            GradientType u_grad_ghost{};
            model.boundary_ghost_gradient(u_grad_ghost, boundary_id, normals[q_face_index], x_q, u_ghost, x_ghost,
                                          u_cell, x_cell, u_grad_cell);

            // Ghost reconstructed state at face
            const std::array<NumberType, n_components> u_plus =
                internal::reconstruct_u(u_ghost, x_ghost, x_q, u_grad_ghost);

            const auto [F_plus, F_minus, a_half] = internal::compute_kt_flux_and_speeds(u_plus, u_minus, x_q, model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);

            const auto &n_bnd = normals[q_face_index];
            const auto D_bnd =
                internal::compute_diffusion_flux(u_plus, u_minus, x_cell, x_ghost, u_cell, u_ghost, n_bnd, x_q, model);

            for (uint dof = 0; dof < n_face_dofs; ++dof) {
              const auto &cd_i = fe_iv.interface_dof_to_dof_indices(dof);
              const auto component_i = cd_i[0] == numbers::invalid_unsigned_int
                                           ? fe.system_to_component_index(cd_i[1]).first
                                           : fe.system_to_component_index(cd_i[0]).first;
              copy_data_face.cell_residual(dof) +=
                  weight * JxW[q_face_index] *
                  (fe_iv.jump_in_shape_values(dof, q_face_index, component_i) *
                   (scalar_product(H[component_i], n_bnd) - scalar_product(D_bnd[component_i], n_bnd)));
            }
          };

          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, residual);
            constraints.distribute_local_to_global(c.cell_mass, c.local_dof_indices, residual);
            for (const auto &face_data : c.face_data) {
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
          using Scratch = internal::ScratchData<dim, NumberType>;
          using CopyData = internal::CopyData_J<NumberType>;
          const auto &constraints = discretization.get_constraints();

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            scratch_data.fe_values.reinit(cell);
            const auto &fe_v = scratch_data.fe_values;
            const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();

            copy_data.reinit(cell, n_dofs, Components::count_extractors());
            const auto &JxW = fe_v.get_JxW_values();
            const auto &q_points = fe_v.get_quadrature_points();
            const auto &q_indices = fe_v.quadrature_point_indices();

            std::vector<VectorType> solution(quadrature.size(), VectorType(n_components));
            std::vector<VectorType> solution_dot(quadrature.size(), VectorType(n_components));
            fe_v.get_function_values(solution_global, solution);
            fe_v.get_function_values(solution_global_dot, solution_dot);

            SimpleMatrix<NumberType, n_components> j_mass;
            SimpleMatrix<NumberType, n_components> j_mass_dot;
            for (const auto &q_index : q_indices) {
              const auto &x_q = q_points[q_index];
              model.template jacobian_mass<0>(j_mass, x_q, solution[q_index], solution_dot[q_index]);
              model.template jacobian_mass<1>(j_mass_dot, x_q, solution[q_index], solution_dot[q_index]);

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = fe_v.get_fe().system_to_component_index(i).first;
                for (uint j = 0; j < n_dofs; ++j) {
                  const auto component_j = fe_v.get_fe().system_to_component_index(j).first;
                  copy_data.cell_jacobian(i, j) +=
                      JxW[q_index] * fe_v.shape_value_component(j, q_index, component_j) *
                      fe_v.shape_value_component(i, q_index, component_i) *
                      (alpha * j_mass_dot(component_i, component_j) + beta * j_mass(component_i, component_j));
                }
              }
            }
          };
          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_jacobian, c.local_dof_indices, jacobian);
          };

          Scratch scratch_data(mapping, discretization.get_fe(), quadrature, quadrature_face);
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
          using Scratch = internal::ScratchData<dim, NumberType>;
          using CopyData = internal::CopyData_J<NumberType>;
          const auto &constraints = discretization.get_constraints();

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            scratch_data.fe_values.reinit(cell);
            const auto &fe_v = scratch_data.fe_values;
            const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();

            copy_data.reinit(cell, n_dofs, Components::count_extractors());
            const auto &JxW = fe_v.get_JxW_values();
            const auto &q_points = fe_v.get_quadrature_points();
            const auto &q_indices = fe_v.quadrature_point_indices();

            std::vector<VectorType> solution(quadrature.size(), VectorType(n_components));
            std::vector<VectorType> solution_dot(quadrature.size(), VectorType(n_components));
            fe_v.get_function_values(solution_global, solution);
            fe_v.get_function_values(solution_global_dot, solution_dot);

            SimpleMatrix<NumberType, n_components> j_mass;
            SimpleMatrix<NumberType, n_components> j_mass_dot;
            SimpleMatrix<NumberType, n_components> j_source;
            for (const auto &q_index : q_indices) {
              const auto &x_q = q_points[q_index];
              model.template jacobian_mass<0>(j_mass, x_q, solution[q_index], solution_dot[q_index]);
              model.template jacobian_mass<1>(j_mass_dot, x_q, solution[q_index], solution_dot[q_index]);
              model.template jacobian_source<0, 0>(j_source, x_q, fv_tie(solution[q_index]));

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = fe_v.get_fe().system_to_component_index(i).first;
                for (uint j = 0; j < n_dofs; ++j) {
                  const auto component_j = fe_v.get_fe().system_to_component_index(j).first;
                  copy_data.cell_jacobian(i, j) += weight * JxW[q_index] * // dx * phi_j
                                                   fe_v.shape_value_component(j, q_index, component_j) *
                                                   (fe_v.shape_value_component(i, q_index, component_i) *
                                                    j_source(component_i, component_j)); // -phi_i * jsource
                  copy_data.cell_mass_jacobian(i, j) +=
                      JxW[q_index] * fe_v.shape_value_component(j, q_index, component_j) *
                      fe_v.shape_value_component(i, q_index, component_i) *
                      (alpha * j_mass_dot(component_i, component_j) + beta * j_mass(component_i, component_j));
                }
              }
            }
          };
          const auto face_worker = [&](const Iterator &cell, const unsigned int &f, const unsigned int &sf,
                                       const Iterator &ncell, const unsigned int &nf, const unsigned int &nsf,
                                       Scratch &scratch_data, CopyData &copy_data) {};

          const auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no, Scratch &scratch_data,
                                           CopyData &copy_data) {
            // pass
          };

          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_jacobian, c.local_dof_indices, jacobian);
            constraints.distribute_local_to_global(c.cell_mass_jacobian, c.local_dof_indices, jacobian);
            for (const auto &face_data : c.face_data)
              constraints.distribute_local_to_global(face_data.cell_jacobian, face_data.joint_dof_indices, jacobian);
          };

          Scratch scratch_data(mapping, discretization.get_fe(), quadrature, quadrature_face);
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
                            bool add_extractor_dofs = false) const
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
      };
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG