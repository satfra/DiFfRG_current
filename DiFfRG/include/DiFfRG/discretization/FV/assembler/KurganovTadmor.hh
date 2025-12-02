#pragma once

// external libraries

// DiFfRG
#include <cstddef>
#include <deal.II/base/multithread_info.h>
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
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <functional>
#include <iostream>
#include <petscvec.h>
#include <spdlog/spdlog.h>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>

#include <DiFfRG/discretization/common/types.hh>
#include <iterator>
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

        template <int dim, typename NumberType> struct Cache_Data {
          dealii::Point<1> position;
          NumberType u;
          NumberType du_dx_half;
          NumberType reconstructed_du;
          NumberType u_plus, u_minus;
          std::optional<std::reference_wrapper<const Cache_Data<dim, NumberType>>> left_neighbor;
          std::optional<std::reference_wrapper<const Cache_Data<dim, NumberType>>> right_neighbor;
        };

        /**
         * @brief Class to hold data for each assembly thread, i.e. FEValues for cells, interfaces, as well as
         * pre-allocated data structures for the solutions
         */
        template <int dim, typename NumberType> struct ScratchData {
          using VectorType = Vector<NumberType>;

          ScratchData(const Mapping<dim> &mapping, const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
                      const VectorType &solution_global,
                      const UpdateFlags update_flags = update_values | update_gradients | update_quadrature_points |
                                                       update_JxW_values)
              : n_components(fe.n_components()), fe_values(mapping, fe, quadrature, update_flags),
                solution_global(solution_global)
          {
          }

          ScratchData(const ScratchData<dim, NumberType> &scratch_data)
              : n_components(scratch_data.fe_values.get_fe().n_components()),
                fe_values(scratch_data.fe_values.get_mapping(), scratch_data.fe_values.get_fe(),
                          scratch_data.fe_values.get_quadrature(), scratch_data.fe_values.get_update_flags()),
                solution_global(scratch_data.solution_global)
          {
          }

          const uint n_components;

          FEValues<dim> fe_values;

          const VectorType &solution_global;
        };

        template <int dim, typename NumberType> class GhostLayer
        {
          using boundary_condition_type =
              std::function<void(std::vector<internal::Cache_Data<dim, NumberType>> &,
                                 internal::Cache_Data<dim, NumberType> &, internal::Cache_Data<dim, NumberType> &)>;

        public:
          GhostLayer() = delete;
          GhostLayer(std::vector<internal::Cache_Data<dim, NumberType>> &cache_data,
                     boundary_condition_type left_boundary_condition, boundary_condition_type right_boundary_condition)
              : _cache_data(cache_data)
          {
            left_boundary_condition(_cache_data, _left_left_boundary_cell, _left_boundary_cell);
            right_boundary_condition(_cache_data, _right_boundary_cell, _right_right_boundary_cell);
          }

          size_t size() const { return _cache_data.size() + 4; }

          void execute_parallel_function(std::function<void(Cache_Data<dim, NumberType> &)> func)
          {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, this->size()),

                              // 2. The Lambda function (receives a sub-range 'r')
                              [&](const tbb::blocked_range<size_t> &r) {
                                // 3. Iterate over the sub-range assigned to this thread
                                for (size_t i = r.begin(); i != r.end(); ++i) {
                                  func((*this)[i]);
                                }
                              });
          }

          Cache_Data<dim, NumberType> &operator[](size_t i)
          {
            if (i == 0) return _left_left_boundary_cell;
            if (i == 1) return _left_boundary_cell;
            if (i < _cache_data.size() + 2) return _cache_data[i - 2];
            if (i == _cache_data.size() + 2) return _right_boundary_cell;
            return _right_right_boundary_cell;
          }

        private:
          std::vector<internal::Cache_Data<dim, NumberType>> &_cache_data;
          Cache_Data<dim, NumberType> _left_left_boundary_cell, _left_boundary_cell, _right_boundary_cell,
              _right_right_boundary_cell;
        };

        template <int dim, typename NumberType, typename CopyDataType> class copy_from_cell_to_array
        {
          using Cache_Vector = std::vector<Cache_Data<dim, NumberType>>;

        public:
          copy_from_cell_to_array(Cache_Vector &cache_data) : cache_data(cache_data) {}
          copy_from_cell_to_array() = delete;
          void operator()(const typename DoFHandler<dim>::active_cell_iterator &cell,
                          internal::ScratchData<dim, NumberType> &scratch_data, CopyDataType & /* copy_data */)
          {
            Cache_Data<dim, NumberType> &user_cell = cache_data[cell->index()];
            user_cell.position = cell->center();

            scratch_data.fe_values.reinit(cell);
            const auto &fe_v = scratch_data.fe_values;
            const auto n_dofs = scratch_data.fe_values.get_fe().n_dofs_per_cell();
            std::vector<NumberType> solution(n_dofs);
            fe_v.get_function_values(scratch_data.solution_global, solution);
            user_cell.u = solution[0];
            const auto &left_neighbor = cell->neighbor(0);
            const auto &right_neighbor = cell->neighbor(1);
            if (left_neighbor->state() == dealii::IteratorState::valid)
              user_cell.left_neighbor = std::ref(cache_data[left_neighbor->index()]);
            if (right_neighbor->state() == dealii::IteratorState::valid)
              user_cell.right_neighbor = std::ref(cache_data[right_neighbor->index()]);
          }

        private:
          Cache_Vector &cache_data;
        };

        template <typename NumberType> NumberType limiter(NumberType r)
        {
          return std::max(NumberType{0}, std::min(NumberType{1}, r));
        }

        template <int dim, typename NumberType> class compute_intermediate_derivates
        {
        public:
          compute_intermediate_derivates() {}
          void operator()(Cache_Data<dim, NumberType> &cache_data) const
          {
            static_assert(dim == 1, "not implemented for dim > 1");
            compute_half_derivative(cache_data);
            compute_reconstruction(cache_data);
          }

        private:
          static void compute_half_derivative(Cache_Data<dim, NumberType> &cache_data)
          {
            if (cache_data.right_neighbor.has_value()) {
              const auto &right_neighbor = cache_data.right_neighbor->get();
              cache_data.du_dx_half =
                  (cache_data.u - right_neighbor.u) / (cache_data.position[0] - right_neighbor.position[0]);
            }
          }

          static void compute_reconstruction(Cache_Data<dim, NumberType> &cache_data)
          {
            if (cache_data.left_neighbor.has_value() && cache_data.right_neighbor.has_value()) {
              auto left_cell = cache_data.left_neighbor->get();
              auto right_cell = cache_data.right_neighbor->get();
              auto r_value = (cache_data.u - left_cell.u) / (right_cell.u - cache_data.u);
              cache_data.reconstructed_du = cache_data.du_dx_half * internal::limiter(r_value);

              auto upper_cell_boundary = (right_cell.position + cache_data.position) / 2.0;
              auto lower_cell_boundary = (left_cell.position + cache_data.position) / 2.0;

              auto dx = upper_cell_boundary - lower_cell_boundary;

              static_assert(dim == 1, "not implemented for dim > 1");

              cache_data.u_plus = cache_data.u + dx[0] / 2.0 * cache_data.reconstructed_du;
              cache_data.u_minus = cache_data.u - dx[0] / 2.0 * cache_data.reconstructed_du;
            }
          }
        };

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
      } // namespace internal

      template <int dim, typename NumberType>
      void LeftAntisymmetricBoundary(std::vector<internal::Cache_Data<dim, NumberType>> &cache_data,
                                     internal::Cache_Data<dim, NumberType> &left_left_boundary_cell,
                                     internal::Cache_Data<dim, NumberType> &left_boundary_cell)
      {
        using internal::Cache_Data;
        left_boundary_cell.u = -cache_data[0].u;
        left_left_boundary_cell.u = -cache_data[1].u;

        double dx = cache_data[1].position[0] - cache_data[0].position[0];
        left_boundary_cell.position = cache_data[0].position - Point<dim>(dx);
        left_left_boundary_cell.position = cache_data[0].position - 2.0 * Point<dim>(dx);

        left_boundary_cell.left_neighbor = std::ref(left_left_boundary_cell);
        left_boundary_cell.right_neighbor = std::ref(cache_data[0]);
        left_left_boundary_cell.right_neighbor = std::ref(left_boundary_cell);

        cache_data[0].left_neighbor = std::ref(left_boundary_cell);
      }

      template <int dim, typename NumberType>
      void RightExtrapolationBoundary(std::vector<internal::Cache_Data<dim, NumberType>> &cache_data,
                                      internal::Cache_Data<dim, NumberType> &right_boundary_cell,
                                      internal::Cache_Data<dim, NumberType> &right_right_boundary_cell)
      {
        using internal::Cache_Data;
        static_assert(dim == 1, "not implemented for dim > 1");
        const size_t N = cache_data.size();
        double dx = cache_data[N - 1].position[0] - cache_data[N - 2].position[0];
        double du = cache_data[N - 1].u - cache_data[N - 2].u;
        right_boundary_cell.u = cache_data[N - 1].u + du;
        right_right_boundary_cell.u = cache_data[N - 1].u + 2 * du;
        right_boundary_cell.position = cache_data[N - 1].position + Point<dim>(dx);
        right_right_boundary_cell.position = cache_data[N - 1].position + 2 * Point<dim>(dx);

        right_boundary_cell.left_neighbor = std::ref(cache_data[N - 1]);
        right_boundary_cell.right_neighbor = std::ref(right_right_boundary_cell);
        right_right_boundary_cell.left_neighbor = std::ref(right_boundary_cell);

        cache_data[N - 1].right_neighbor = std::ref(right_boundary_cell);
      }

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

        Assembler(Discretization &discretization, Model &model, const JSONValue &json)
            : discretization(discretization), model(model), dof_handler(discretization.get_dof_handler()),
              mapping(discretization.get_mapping()), triangulation(discretization.get_triangulation()), json(json),
              threads(json.get_uint("/discretization/threads")),
              batch_size(json.get_uint("/discretization/batch_size")),
              EoM_abs_tol(json.get_double("/discretization/EoM_abs_tol")),
              EoM_max_iter(json.get_uint("/discretization/EoM_max_iter")),
              quadrature(1 + json.get_uint("/discretization/overintegration"))
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

          // Mass sparsity pattern is diagonal
          auto dynamic_sparsity_pattern_mass = DynamicSparsityPattern(dof_handler.n_dofs());
          for (uint i = 0; i < dof_handler.n_dofs(); ++i)
            dynamic_sparsity_pattern_mass.add(i, i);
          sparsity_pattern_mass.copy_from(dynamic_sparsity_pattern_mass);

          mass_matrix = SparseMatrix<NumberType>(sparsity_pattern_mass, IdentityMatrix(dof_handler.n_dofs()));

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
                                        const VectorType &) override
        {
          model.template jacobian_variables<0>(jacobian, fv_tie(variables));
        };

        void readouts(DataOutput<dim, VectorType> &data_out, const VectorType &, const VectorType &variables) const
        {
          auto helper = [&](auto EoMfun, auto outputter) {
            (void)EoMfun;
            outputter(data_out, Point<0>(), fv_tie(variables));
          };
          model.template readouts_multiple<0>(helper, data_out);
        }

        virtual void mass(VectorType &mass, const VectorType &solution_global, const VectorType &solution_global_dot,
                          NumberType weight) override
        {
          using Iterator = typename DoFHandler<dim>::active_cell_iterator;
          using Scratch = internal::ScratchData<dim, NumberType>;
          using CopyData = internal::CopyData_R<NumberType>;
          const auto &constraints = discretization.get_constraints();

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            scratch_data.fe_values.reinit(cell);
            const auto &fe_v = scratch_data.fe_values;
            const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();

            copy_data.reinit(cell, n_dofs);
            const auto &JxW = fe_v.get_JxW_values();
            const auto &q_points = fe_v.get_quadrature_points();
            const auto &q_indices = fe_v.quadrature_point_indices();

            auto &solution = scratch_data.solution_global;
            auto &solution_dot = scratch_data.solution_dot;
            fe_v.get_function_values(solution_global, solution);
            fe_v.get_function_values(solution_global_dot, solution_dot);

            std::array<NumberType, n_components> mass{};
            for (const auto &q_index : q_indices) {
              const auto &x_q = q_points[q_index];
              model.mass(mass, x_q, solution[q_index], solution_dot[q_index]);

              for (uint i = 0; i < n_dofs; ++i) {
                const auto component_i = fe_v.get_fe().system_to_component_index(i).first;
                copy_data.cell_residual(i) += weight * JxW[q_index] *
                                              fe_v.shape_value_component(i, q_index, component_i) *
                                              mass[component_i]; // +phi_i(x_q) * mass(x_q, u_q)
              }
            }
          };
          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, mass);
          };

          Scratch scratch_data(mapping, discretization.get_fe(), quadrature);
          CopyData copy_data;
          MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells;

          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, nullptr, nullptr, threads, batch_size);
        }

        virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                              const VectorType &solution_global_dot, NumberType weight_mass,
                              const VectorType & /* variables */ = VectorType()) override
        {
          using Iterator = typename DoFHandler<dim>::active_cell_iterator;
          using Scratch = internal::ScratchData<dim, NumberType>;
          using CopyData = internal::CopyData_R<NumberType>;
          const auto &constraints = discretization.get_constraints();
          const auto number_of_cells = discretization.get_triangulation().n_active_cells();
          std::array<internal::Cache_Data<dim, NumberType>, n_components> cached_array{};
          Scratch scratch_data(mapping, discretization.get_fe(), quadrature, solution_global);

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
            scratch_data.fe_values.reinit(cell); // this sort of reinitializes the FEValues for a given cell
            const auto &fe_v = scratch_data.fe_values;
            const uint n_dofs = fe_v.get_fe().n_dofs_per_cell(); // should this be one?

            copy_data.reinit(cell, n_dofs); // this prepares the local data structures, although I think this is
                                            // also clear, since we use a zeroth order FV method
            const auto &JxW = fe_v.get_JxW_values();             // these are the quardrature weight times the jacobian
                                                                 // determinat, which would exist in a volume integral
            const auto &q_points = fe_v.get_quadrature_points(); // these are the quadrateure points
            const auto &q_indices =
                fe_v.quadrature_point_indices(); // these are the indices, so one can loop over them later on

            auto &solution = scratch_data.solution_global;
            auto &solution_dot = scratch_data.solution_dot;
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
          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, residual);
            constraints.distribute_local_to_global(c.cell_mass, c.local_dof_indices, residual);
          };

          CopyData copy_data;
          MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells;

          Timer timer;
          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, nullptr, nullptr, threads, batch_size);
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

            auto &solution = scratch_data.solution_global;
            auto &solution_dot = scratch_data.solution_dot;
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

          Scratch scratch_data(mapping, discretization.get_fe(), quadrature);
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

            auto &solution = scratch_data.solution_global;
            auto &solution_dot = scratch_data.solution_dot;
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
                  copy_data.cell_jacobian(i, j) +=
                      weight * JxW[q_index] * fe_v.shape_value_component(j, q_index, component_j) * // dx * phi_j * (
                      (fe_v.shape_value_component(i, q_index, component_i) *
                       j_source(component_i, component_j)); // -phi_i * jsource)
                  copy_data.cell_mass_jacobian(i, j) +=
                      JxW[q_index] * fe_v.shape_value_component(j, q_index, component_j) *
                      fe_v.shape_value_component(i, q_index, component_i) *
                      (alpha * j_mass_dot(component_i, component_j) + beta * j_mass(component_i, component_j));
                }
              }
            }
          };
          const auto copier = [&](const CopyData &c) {
            constraints.distribute_local_to_global(c.cell_jacobian, c.local_dof_indices, jacobian);
            constraints.distribute_local_to_global(c.cell_mass_jacobian, c.local_dof_indices, jacobian);
          };

          Scratch scratch_data(mapping, discretization.get_fe(), quadrature);
          CopyData copy_data;
          MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells;

          Timer timer;
          MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                                copy_data, flags, nullptr, nullptr, threads, batch_size);
          timings_jacobian.push_back(timer.wall_time());
        }

        void build_sparsity(SparsityPattern &sparsity_pattern, const DoFHandler<dim> &to_dofh,
                            const DoFHandler<dim> &from_dofh, const int stencil = 1,
                            bool /* add_extractor_dofs */ = false) const
        {
          const auto &triangulation = discretization.get_triangulation();

          DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

          const auto to_dofs_per_cell = to_dofh.get_fe().dofs_per_cell;
          const auto from_dofs_per_cell = from_dofh.get_fe().dofs_per_cell;

          for (const auto &t_cell : triangulation.active_cell_iterators()) {
            std::vector<types::global_dof_index> to_dofs(to_dofs_per_cell);
            std::vector<types::global_dof_index> from_dofs(from_dofs_per_cell);
            const auto to_cell = t_cell->as_dof_handler_iterator(to_dofh);
            const auto from_cell = t_cell->as_dof_handler_iterator(from_dofh);
            to_cell->get_dof_indices(to_dofs);
            from_cell->get_dof_indices(from_dofs);

            std::function<void(decltype(from_cell) &, const int)> add_all_neighbor_dofs = [&](const auto &from_cell,
                                                                                              const int stencil_level) {
              for (const auto face_no : from_cell->face_indices()) {
                const auto face = from_cell->face(face_no);
                if (!face->at_boundary()) {
                  auto neighbor_cell = from_cell->neighbor(face_no);

                  if (dim == 1)
                    while (neighbor_cell->has_children())
                      neighbor_cell = neighbor_cell->child(face_no == 0 ? 1 : 0);

                  // add all children
                  else if (neighbor_cell->has_children()) {
                    throw std::runtime_error("not yet implemented lol");
                  }

                  if (!neighbor_cell->is_active()) continue;

                  std::vector<types::global_dof_index> tmp(from_dofs_per_cell);
                  neighbor_cell->get_dof_indices(tmp);

                  from_dofs.insert(std::end(from_dofs), std::begin(tmp), std::end(tmp));

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
          //   for (uint row = 0; row < dsp.n_rows(); ++row)
          //     dsp.add_row_entries(row, extractor_dof_indices);
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

        const JSONValue &json;

        uint threads;
        const uint batch_size;

        mutable typename Triangulation<dim>::cell_iterator EoM_cell;
        typename Triangulation<dim>::cell_iterator old_EoM_cell;
        const double EoM_abs_tol;
        const uint EoM_max_iter;

        const QGauss<dim> quadrature;

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