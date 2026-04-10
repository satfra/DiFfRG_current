#pragma once

// standard library
#include <sstream>

// DiFfRG
#include <DiFfRG/discretization/FEM/assembler/common.hh>

namespace DiFfRG
{
  namespace CG
  {
    using namespace dealii;
    using std::array;

    template <typename... T> auto fe_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>,
                         StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors", "variables">>(
          std::tie(t...));
    }

    template <typename... T> auto i_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives", "fe_hessians">>(
          std::tie(t...));
    }

    namespace internal
    {
      /**
       * @brief Class to hold data for each assembly thread, i.e. FEValues for cells, interfaces, as well as
       * pre-allocated data structures for the solutions
       *
       * @tparam Model Model describing the assembled problem.
       */
      template <typename Discretization> struct ScratchData {
        static constexpr uint dim = Discretization::dim;
        using NumberType = typename Discretization::NumberType;

        ScratchData(const Mapping<dim> &mapping, const FiniteElement<dim> &fe,
                    const dealii::Quadrature<dim> &quadrature, const dealii::Quadrature<dim - 1> &quadrature_face,
                    const UpdateFlags update_flags = update_values | update_gradients | update_quadrature_points |
                                                     update_JxW_values | update_hessians,
                    const UpdateFlags interface_update_flags = update_values | update_gradients |
                                                               update_quadrature_points | update_JxW_values |
                                                               update_normal_vectors | update_hessians)
            : n_components(fe.n_components()), fe_values(mapping, fe, quadrature, update_flags),
              fe_interface_values(mapping, fe, quadrature_face, interface_update_flags)
        {
          solution.resize(quadrature.size(), Vector<NumberType>(n_components));
          solution_grad.resize(quadrature.size(), std::vector<Tensor<1, dim, NumberType>>(n_components));
          solution_hess.resize(quadrature.size(), std::vector<Tensor<2, dim, NumberType>>(n_components));
          solution_dot.resize(quadrature.size(), Vector<NumberType>(n_components));
          solution_interface[0].resize(quadrature_face.size(), Vector<NumberType>(n_components));
          solution_interface[1].resize(quadrature_face.size(), Vector<NumberType>(n_components));
          solution_grad_interface[0].resize(quadrature_face.size(),
                                            std::vector<Tensor<1, dim, NumberType>>(n_components));
          solution_grad_interface[1].resize(quadrature_face.size(),
                                            std::vector<Tensor<1, dim, NumberType>>(n_components));
          solution_hess_interface[0].resize(quadrature_face.size(),
                                            std::vector<Tensor<2, dim, NumberType>>(n_components));
          solution_hess_interface[1].resize(quadrature_face.size(),
                                            std::vector<Tensor<2, dim, NumberType>>(n_components));
          const uint n_dofs = fe.n_dofs_per_cell();
          comp.resize(n_dofs);
          cached_shape_values.resize(n_dofs);
          cached_shape_grads.resize(n_dofs);
          cached_shape_hessians.resize(n_dofs);
        }

        ScratchData(const ScratchData<Discretization> &scratch_data)
            : n_components(scratch_data.fe_values.get_fe().n_components()),
              fe_values(scratch_data.fe_values.get_mapping(), scratch_data.fe_values.get_fe(),
                        scratch_data.fe_values.get_quadrature(), scratch_data.fe_values.get_update_flags()),
              fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                                  scratch_data.fe_interface_values.get_fe(),
                                  scratch_data.fe_interface_values.get_quadrature(),
                                  scratch_data.fe_interface_values.get_update_flags())
        {
          const uint q_size = scratch_data.fe_values.get_quadrature().size();
          const uint q_face_size = scratch_data.fe_interface_values.get_quadrature().size();

          solution.resize(q_size, Vector<NumberType>(n_components));
          solution_grad.resize(q_size, std::vector<Tensor<1, dim, NumberType>>(n_components));
          solution_hess.resize(q_size, std::vector<Tensor<2, dim, NumberType>>(n_components));
          solution_dot.resize(q_size, Vector<NumberType>(n_components));
          solution_interface[0].resize(q_face_size, Vector<NumberType>(n_components));
          solution_interface[1].resize(q_face_size, Vector<NumberType>(n_components));
          solution_grad_interface[0].resize(q_face_size, std::vector<Tensor<1, dim, NumberType>>(n_components));
          solution_grad_interface[1].resize(q_face_size, std::vector<Tensor<1, dim, NumberType>>(n_components));
          solution_hess_interface[0].resize(q_face_size, std::vector<Tensor<2, dim, NumberType>>(n_components));
          solution_hess_interface[1].resize(q_face_size, std::vector<Tensor<2, dim, NumberType>>(n_components));
          const uint n_dofs_copy = scratch_data.comp.size();
          comp.resize(n_dofs_copy);
          cached_shape_values.resize(n_dofs_copy);
          cached_shape_grads.resize(n_dofs_copy);
          cached_shape_hessians.resize(n_dofs_copy);
        }

        const uint n_components;

        FEValues<dim> fe_values;
        FEInterfaceValues<dim> fe_interface_values;

        std::vector<Vector<NumberType>> solution;
        std::vector<std::vector<Tensor<1, dim, NumberType>>> solution_grad;
        std::vector<std::vector<Tensor<2, dim, NumberType>>> solution_hess;
        std::vector<Vector<NumberType>> solution_dot;
        array<std::vector<Vector<NumberType>>, 2> solution_interface;
        array<std::vector<std::vector<Tensor<1, dim, NumberType>>>, 2> solution_grad_interface;
        array<std::vector<std::vector<Tensor<2, dim, NumberType>>>, 2> solution_hess_interface;

        std::vector<uint> comp;

        // Cached per-DoF shape function data for jacobian assembly
        std::vector<double> cached_shape_values;
        std::vector<Tensor<1, dim>> cached_shape_grads;
        std::vector<Tensor<2, dim>> cached_shape_hessians;
      };

      template <typename NumberType> struct CopyData_R {
        Vector<NumberType> cell_residual;
        std::vector<types::global_dof_index> local_dof_indices;

        template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell)
        {
          cell_residual.reinit(dofs_per_cell);
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);
        }
      };

      template <typename NumberType> struct CopyData_J {
        FullMatrix<NumberType> cell_jacobian;
        FullMatrix<NumberType> extractor_cell_jacobian;
        std::vector<types::global_dof_index> local_dof_indices;

        template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell, uint n_extractors)
        {
          cell_jacobian.reinit(dofs_per_cell, dofs_per_cell);
          if (n_extractors > 0) extractor_cell_jacobian.reinit(dofs_per_cell, n_extractors);
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

    /**
     * @brief The basic assembler that can be used for any standard CG scheme with flux and source.
     *
     * @tparam Model The model class which contains the physical equations.
     */
    template <typename Discretization_, typename Model_> class Assembler : public FEMAssembler<Discretization_, Model_>
    {
      using Base = FEMAssembler<Discretization_, Model_>;

    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;

      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;

      Assembler(Discretization &discretization, Model &model, const JSONValue &json)
          : Base(discretization, model, json),
            quadrature(fe.degree + 1 + json.get_uint("/discretization/overintegration")),
            quadrature_face(fe.degree + 1 + json.get_uint("/discretization/overintegration"))
      {
        static_assert(Components::count_fe_subsystems() == 1, "A CG model cannot have multiple submodels!");
        reinit();
      }

      virtual void reinit_vector(VectorType &vec) const override { vec.reinit(dof_handler.n_dofs()); }

      virtual void reinit() override
      {
        Timer timer;

        Base::reinit();

        // Mass sparsity pattern
        {
          DynamicSparsityPattern dsp(dof_handler.n_dofs());
          DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                          /*keep_constrained_dofs = */ true);
          sparsity_pattern_mass.copy_from(dsp);
          mass_matrix.reinit(sparsity_pattern_mass);
          MatrixCreator::create_mass_matrix(dof_handler, quadrature, mass_matrix, (Function<dim, NumberType> *)nullptr,
                                            discretization.get_constraints());
        }
        // Jacobian sparsity pattern
        {
          DynamicSparsityPattern dsp(dof_handler.n_dofs());
          DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                          /*keep_constrained_dofs = */ true);
          sparsity_pattern_jacobian.copy_from(dsp);
        }
        timings_reinit.push_back(timer.wall_time());

        // Boundary dofs
        std::vector<IndexSet> component_boundary_dofs(Components::count_fe_functions());
        for (uint c = 0; c < Components::count_fe_functions(); ++c) {
          ComponentMask component_mask(Components::count_fe_functions(), false);
          component_mask.set(c, true);
          component_boundary_dofs[c] = DoFTools::extract_boundary_dofs(dof_handler, component_mask);
        }
        std::vector<std::vector<Point<dim>>> component_boundary_points(Components::count_fe_functions());
        for (uint c = 0; c < Components::count_fe_functions(); ++c) {
          component_boundary_points[c].resize(component_boundary_dofs[c].n_elements());
          for (uint i = 0; i < component_boundary_dofs[c].n_elements(); ++i)
            component_boundary_points[c][i] =
                discretization.get_support_point(component_boundary_dofs[c].nth_index_in_set(i));
        }

        auto &constraints = discretization.get_constraints();
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        model.affine_constraints(constraints, component_boundary_dofs, component_boundary_points);
        constraints.close();
      }

      virtual void rebuild_jacobian_sparsity() override
      {
        // Jacobian sparsity pattern
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                        /*keep_constrained_dofs = */ true);
        for (uint row = 0; row < dsp.n_rows(); ++row)
          for (const auto &col : extractor_dof_indices)
            dsp.add(row, col);
        sparsity_pattern_jacobian.copy_from(dsp);
      }

      virtual const SparsityPattern &get_sparsity_pattern_jacobian() const override
      {
        return sparsity_pattern_jacobian;
      }
      virtual const SparseMatrix<NumberType> &get_mass_matrix() const override { return mass_matrix; }

      /**
       * @brief refinement indicator for adaptivity. Only calls the model's cell_indicator function, as in CG schemes
       * the cell boundary is not discontinuous.
       *
       * @param indicator The vector to store the refinement indicator in.
       * @param solution_global The global solution vector.
       */
      virtual void refinement_indicator(Vector<double> &indicator, const VectorType &solution_global) override
      {
        using Iterator = typename DoFHandler<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_I<NumberType>;

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          scratch_data.fe_values.reinit(t_cell);
          const auto &fe_v = scratch_data.fe_values;
          copy_data.cell_index = t_cell->active_cell_index();
          copy_data.value = 0;

          const auto &JxW = fe_v.get_JxW_values();
          const auto &q_points = fe_v.get_quadrature_points();
          const auto &q_indices = fe_v.quadrature_point_indices();

          auto &solution = scratch_data.solution;
          auto &solution_grad = scratch_data.solution_grad;
          auto &solution_hess = scratch_data.solution_hess;
          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_gradients(solution_global, solution_grad);
          fe_v.get_function_hessians(solution_global, solution_hess);

          double local_indicator = 0.;
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.cell_indicator(local_indicator, x_q,
                                 i_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index]));
            copy_data.value += JxW[q_index] * local_indicator;
          }
        };
        const auto copier = [&](const CopyData &c) { indicator[c.cell_index] += c.value; };

        Scratch scratch_data(mapping, fe, quadrature, quadrature_face);
        CopyData copy_data;
        MeshWorker::AssembleFlags assemble_flags = MeshWorker::assemble_own_cells;

        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, assemble_flags, nullptr, nullptr, threads, batch_size);
      }

      virtual void mass(VectorType &mass, const VectorType &solution_global, const VectorType &solution_global_dot,
                        NumberType weight) override
      {
        using Iterator = typename DoFHandler<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
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

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe_v.get_fe().system_to_component_index(i).first;

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;
          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_values(solution_global_dot, solution_dot);

          array<NumberType, Components::count_fe_functions()> mass{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.mass(mass, x_q, solution[q_index], solution_dot[q_index]);

            for (uint i = 0; i < n_dofs; ++i) {
              copy_data.cell_residual(i) += weight * JxW[q_index] * fe_v.shape_value_component(i, q_index, comp[i]) *
                                            mass[comp[i]]; // +phi_i(x_q) * mass(x_q, u_q)
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, mass);
        };

        Scratch scratch_data(mapping, fe, quadrature, quadrature_face);
        CopyData copy_data;
        MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells;

        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, flags, nullptr, nullptr, threads, batch_size);
      }

      virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                            const VectorType &solution_global_dot, NumberType weight_mass,
                            const VectorType &variables = VectorType()) override
      {
        using Iterator = typename DoFHandler<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_R<NumberType>;
        const auto &constraints = discretization.get_constraints();

        // Find the EoM and extract whatever data is needed for the model.
        std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
        if constexpr (Components::count_extractors() > 0)
          this->extract(__extracted_data, solution_global, variables, true, false, true);
        const auto &extracted_data = __extracted_data;

        const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
          scratch_data.fe_values.reinit(cell);
          const auto &fe_v = scratch_data.fe_values;
          const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();

          copy_data.reinit(cell, n_dofs);
          const auto &JxW = fe_v.get_JxW_values();
          const auto &q_points = fe_v.get_quadrature_points();
          const auto &q_indices = fe_v.quadrature_point_indices();

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

          auto &solution = scratch_data.solution;
          auto &solution_grad = scratch_data.solution_grad;
          auto &solution_hess = scratch_data.solution_hess;
          auto &solution_dot = scratch_data.solution_dot;
          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_gradients(solution_global, solution_grad);
          fe_v.get_function_hessians(solution_global, solution_hess);
          fe_v.get_function_values(solution_global_dot, solution_dot);

          array<Tensor<1, dim, NumberType>, Components::count_fe_functions()> flux{};
          array<NumberType, Components::count_fe_functions()> source{};
          array<NumberType, Components::count_fe_functions()> mass{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.flux(
                flux, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.source(
                source, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.mass(mass, x_q, solution[q_index], solution_dot[q_index]);

            for (uint i = 0; i < n_dofs; ++i) {
              const auto &ci = comp[i];
              copy_data.cell_residual(i) +=
                  JxW[q_index] * weight * // dx *
                  (-scalar_product(fe_v.shape_grad_component(i, q_index, ci),
                                   flux[ci])                                   // -dphi_i(x_q) * flux(x_q, u_q)
                   + fe_v.shape_value_component(i, q_index, ci) * source[ci]); // -phi_i(x_q) * source(x_q, u_q)
              copy_data.cell_residual(i) += weight_mass * JxW[q_index] * fe_v.shape_value_component(i, q_index, ci) *
                                            mass[ci]; // +phi_i(x_q) * mass(x_q, u_q)
            }
          }
        };
        const auto boundary_worker = [&](const Iterator &cell, const uint &face_no, Scratch &scratch_data,
                                         CopyData &copy_data) {
          scratch_data.fe_interface_values.reinit(cell, face_no);
          const auto &fe_fv = scratch_data.fe_interface_values.get_fe_face_values(0);
          const uint n_dofs = fe_fv.get_fe().n_dofs_per_cell();

          const auto &JxW = fe_fv.get_JxW_values();
          const auto &q_points = fe_fv.get_quadrature_points();
          const auto &q_indices = fe_fv.quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_fv.get_normal_vectors();

          auto &solution = scratch_data.solution_interface[0];
          auto &solution_grad = scratch_data.solution_grad_interface[0];
          auto &solution_hess = scratch_data.solution_hess_interface[0];
          fe_fv.get_function_values(solution_global, solution);
          fe_fv.get_function_gradients(solution_global, solution_grad);
          fe_fv.get_function_hessians(solution_global, solution_hess);

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

          array<Tensor<1, dim, NumberType>, Components::count_fe_functions()> numflux{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.boundary_numflux(
                numflux, normals[q_index], x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));

            for (uint i = 0; i < n_dofs; ++i) {
              const auto &ci = comp[i];
              copy_data.cell_residual(i) +=
                  weight * JxW[q_index] * // dx
                  (fe_fv.shape_value_component(i, q_index, ci) *
                   scalar_product(numflux[ci], normals[q_index])); // phi_i(x_q) * numflux(x_q, u_q) * n(x_q)
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, residual);
        };

        Scratch scratch_data(mapping, fe, quadrature, quadrature_face);
        CopyData copy_data;
        MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces;

        Timer timer;
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, flags, boundary_worker, nullptr, threads, batch_size);
        timings_residual.push_back(timer.wall_time());
      }

      virtual void jacobian_mass(SparseMatrix<NumberType> &jacobian, const VectorType &solution_global,
                                 const VectorType &solution_global_dot, NumberType alpha, NumberType beta) override
      {
        using Iterator = typename DoFHandler<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
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

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;

          SimpleMatrix<NumberType, Components::count_fe_functions()> j_mass;
          SimpleMatrix<NumberType, Components::count_fe_functions()> j_mass_dot;

          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_values(solution_global_dot, solution_dot);
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template jacobian_mass<0>(j_mass, x_q, solution[q_index], solution_dot[q_index]);
            model.template jacobian_mass<1>(j_mass_dot, x_q, solution[q_index], solution_dot[q_index]);

            for (uint i = 0; i < n_dofs; ++i) {
              for (uint j = 0; j < n_dofs; ++j) {
                copy_data.cell_jacobian(i, j) +=
                    JxW[q_index] * fe_v.shape_value_component(j, q_index, comp[j]) *
                    fe_v.shape_value_component(i, q_index, comp[i]) *
                    (alpha * j_mass_dot(comp[i], comp[j]) + beta * j_mass(comp[i], comp[j]));
              }
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_jacobian, c.local_dof_indices, jacobian);
        };

        Scratch scratch_data(mapping, fe, quadrature, quadrature_face);
        CopyData copy_data;
        MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells;

        Timer timer;
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, flags, nullptr, nullptr, threads, batch_size);
        timings_jacobian.push_back(timer.wall_time());
      }

      virtual void jacobian(SparseMatrix<NumberType> &jacobian, const VectorType &solution_global, NumberType weight,
                            const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                            const VectorType &variables = VectorType()) override
      {
        using Iterator = typename DoFHandler<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_J<NumberType>;
        const auto &constraints = discretization.get_constraints();

        // Find the EoM and extract whatever data is needed for the model.
        std::array<NumberType, Components::count_extractors()> extracted_data{{}};
        if constexpr (Components::count_extractors() > 0) {
          this->extract(extracted_data, solution_global, variables, true, true, true);
          if (this->jacobian_extractors(this->extractor_jacobian, solution_global, variables))
            jacobian.reinit(sparsity_pattern_jacobian);
        }

        const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
          scratch_data.fe_values.reinit(cell);
          const auto &fe_v = scratch_data.fe_values;
          const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();

          copy_data.reinit(cell, n_dofs, Components::count_extractors());
          const auto &JxW = fe_v.get_JxW_values();
          const auto &q_points = fe_v.get_quadrature_points();
          const auto &q_indices = fe_v.quadrature_point_indices();

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;
          auto &solution_grad = scratch_data.solution_grad;
          auto &solution_hess = scratch_data.solution_hess;

          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_gradients(solution_global, solution_grad);
          fe_v.get_function_hessians(solution_global, solution_hess);
          fe_v.get_function_values(solution_global_dot, solution_dot);

          SimpleMatrix<Tensor<1, dim, NumberType>, Components::count_fe_functions()> j_flux;
          SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NumberType>>, Components::count_fe_functions()> j_grad_flux;
          SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NumberType>>, Components::count_fe_functions()> j_hess_flux;
          SimpleMatrix<Tensor<1, dim, NumberType>, Components::count_fe_functions(), Components::count_extractors()>
              j_extr_flux;
          SimpleMatrix<NumberType, Components::count_fe_functions()> j_source;
          SimpleMatrix<Tensor<1, dim, NumberType>, Components::count_fe_functions()> j_grad_source;
          SimpleMatrix<Tensor<2, dim, NumberType>, Components::count_fe_functions()> j_hess_source;
          SimpleMatrix<NumberType, Components::count_fe_functions(), Components::count_extractors()> j_extr_source;
          SimpleMatrix<NumberType, Components::count_fe_functions()> j_mass;
          SimpleMatrix<NumberType, Components::count_fe_functions()> j_mass_dot;

          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template jacobian_flux_source<0, 0>(
                j_flux, j_source, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_flux_source_grad<1>(
                j_grad_flux, j_grad_source, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_flux_source_hess<2>(
                j_hess_flux, j_hess_source, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            if constexpr (Components::count_extractors() > 0) {
              model.template jacobian_flux_source_extr<3>(
                  j_extr_flux, j_extr_source, x_q,
                  fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            }
            model.template jacobian_mass<0>(j_mass, x_q, solution[q_index], solution_dot[q_index]);
            model.template jacobian_mass<1>(j_mass_dot, x_q, solution[q_index], solution_dot[q_index]);

            // Cache per-DoF shape function data for this quadrature point
            auto &sv = scratch_data.cached_shape_values;
            auto &sg = scratch_data.cached_shape_grads;
            auto &sh = scratch_data.cached_shape_hessians;
            for (uint k = 0; k < n_dofs; ++k) {
              sv[k] = fe_v.shape_value_component(k, q_index, comp[k]);
              sg[k] = fe_v.shape_grad_component(k, q_index, comp[k]);
              sh[k] = fe_v.shape_hessian_component(k, q_index, comp[k]);
            }

            const auto do_work = [&](uint i_begin, uint i_end, uint j_begin, uint j_end) {
              for (uint i = i_begin; i < i_end; ++i) {
                const auto &ci = comp[i];
                const auto &sv_i = sv[i];
                const auto &sg_i = sg[i];
                for (uint j = j_begin; j < j_end; ++j) {
                  const auto &cj = comp[j];
                  NumberType contribution = weight * JxW[q_index] *
                                            (sv[j] * (-scalar_product(sg_i, j_flux(ci, cj)) + sv_i * j_source(ci, cj)) +
                                             scalar_product(sg[j], -scalar_product(sg_i, j_grad_flux(ci, cj)) +
                                                                       sv_i * j_grad_source(ci, cj)) +
                                             scalar_product(sh[j], -scalar_product(sg_i, j_hess_flux(ci, cj)) +
                                                                       sv_i * j_hess_source(ci, cj)));
                  contribution += JxW[q_index] * sv[j] * sv_i * (alpha * j_mass_dot(ci, cj) + beta * j_mass(ci, cj));
                  copy_data.cell_jacobian(i, j) += contribution;
                }
              }
            };

            if (n_dofs * n_dofs < 64)
              do_work(0, n_dofs, 0, n_dofs);
            else
              tbb::parallel_for(
                  tbb::blocked_range2d<uint>(0, n_dofs, 0, n_dofs), [&](const tbb::blocked_range2d<uint> &range) {
                    do_work(range.rows().begin(), range.rows().end(), range.cols().begin(), range.cols().end());
                  });

            // extractor contribution
            if constexpr (Components::count_extractors() > 0) {
              for (uint i = 0; i < n_dofs; ++i) {
                for (uint e = 0; e < Components::count_extractors(); ++e)
                  copy_data.extractor_cell_jacobian(i, e) +=
                      weight * JxW[q_index] * // dx * phi_j * (
                      (-scalar_product(fe_v.shape_grad_component(i, q_index, comp[i]),
                                       j_extr_flux(comp[i], e)) // -dphi_i * jflux
                       + fe_v.shape_value_component(i, q_index, comp[i]) *
                             j_extr_source(comp[i], e)); // -phi_i * jsource)
              }
            }
          }
        };

        const auto boundary_worker = [&](const Iterator &cell, const uint &face_no, Scratch &scratch_data,
                                         CopyData &copy_data) {
          scratch_data.fe_interface_values.reinit(cell, face_no);
          const auto &fe_fv = scratch_data.fe_interface_values.get_fe_face_values(0);
          const uint n_dofs = fe_fv.get_fe().n_dofs_per_cell();

          const auto &JxW = fe_fv.get_JxW_values();
          const auto &q_points = fe_fv.get_quadrature_points();
          const auto &q_indices = fe_fv.quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_fv.get_normal_vectors();

          auto &solution = scratch_data.solution_interface[0];
          auto &solution_grad = scratch_data.solution_grad_interface[0];
          auto &solution_hess = scratch_data.solution_hess;
          fe_fv.get_function_values(solution_global, solution);
          fe_fv.get_function_gradients(solution_global, solution_grad);
          fe_fv.get_function_hessians(solution_global, solution_hess);

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

          SimpleMatrix<Tensor<1, dim, NumberType>, Components::count_fe_functions()> j_boundary_numflux;
          SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NumberType>>, Components::count_fe_functions()>
              j_grad_boundary_numflux;
          SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NumberType>>, Components::count_fe_functions()>
              j_hess_boundary_numflux;
          SimpleMatrix<Tensor<1, dim, NumberType>, Components::count_fe_functions(), Components::count_extractors()>
              j_extr_boundary_numflux;
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template jacobian_boundary_numflux<0, 0>(
                j_boundary_numflux, normals[q_index], x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_boundary_numflux_grad<1>(
                j_grad_boundary_numflux, normals[q_index], x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_boundary_numflux_hess<2>(
                j_hess_boundary_numflux, normals[q_index], x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            if constexpr (Components::count_extractors() > 0) {
              model.template jacobian_boundary_numflux_extr<3>(
                  j_extr_boundary_numflux, normals[q_index], x_q,
                  fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            }

            // Cache per-DoF shape function data for boundary
            auto &bsv = scratch_data.cached_shape_values;
            auto &bsg = scratch_data.cached_shape_grads;
            auto &bsh = scratch_data.cached_shape_hessians;
            for (uint k = 0; k < n_dofs; ++k) {
              bsv[k] = fe_fv.shape_value_component(k, q_index, comp[k]);
              bsg[k] = fe_fv.shape_grad_component(k, q_index, comp[k]);
              bsh[k] = fe_fv.shape_hessian_component(k, q_index, comp[k]);
            }

            const auto do_bnd_work = [&](uint i_begin, uint i_end, uint j_begin, uint j_end) {
              for (uint i = i_begin; i < i_end; ++i) {
                const auto &ci = comp[i];
                const auto &sv_i = bsv[i];
                for (uint j = j_begin; j < j_end; ++j) {
                  const auto &cj = comp[j];
                  const auto n_dot_jnf = scalar_product(j_boundary_numflux(ci, cj), normals[q_index]);
                  const auto n_dot_jgnf = scalar_product(j_grad_boundary_numflux(ci, cj), normals[q_index]);
                  const auto n_dot_jhnf = scalar_product(j_hess_boundary_numflux(ci, cj), normals[q_index]);
                  copy_data.cell_jacobian(i, j) +=
                      weight * JxW[q_index] *
                      (bsv[j] * sv_i * n_dot_jnf + scalar_product(bsg[j], sv_i * n_dot_jgnf) +
                       scalar_product(bsh[j], sv_i * n_dot_jhnf));
                }
              }
            };

            if (n_dofs * n_dofs < 64)
              do_bnd_work(0, n_dofs, 0, n_dofs);
            else
              tbb::parallel_for(
                  tbb::blocked_range2d<uint>(0, n_dofs, 0, n_dofs), [&](const tbb::blocked_range2d<uint> &range) {
                    do_bnd_work(range.rows().begin(), range.rows().end(), range.cols().begin(), range.cols().end());
                  });

            // extractor contribution
            if constexpr (Components::count_extractors() > 0) {
              for (uint i = 0; i < n_dofs; ++i) {
                for (uint e = 0; e < Components::count_extractors(); ++e)
                  copy_data.extractor_cell_jacobian(i, e) +=
                      weight * JxW[q_index] * // dx * phi_j(x_q)
                      (fe_fv.shape_value_component(i, q_index, comp[i]) *
                       scalar_product(j_extr_boundary_numflux(comp[i], e),
                                      normals[q_index])); // phi_i(x_q) * j_numflux(x_q, u_q) * n(x_q)
              }
            }
          }
        };

        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_jacobian, c.local_dof_indices, jacobian);
          if constexpr (Components::count_extractors() > 0) {
            FullMatrix<NumberType> extractor_dependence(c.local_dof_indices.size(), extractor_dof_indices.size());
            c.extractor_cell_jacobian.mmult(extractor_dependence, this->extractor_jacobian);
            constraints.distribute_local_to_global(extractor_dependence, c.local_dof_indices, extractor_dof_indices,
                                                   jacobian);
          }
        };

        Scratch scratch_data(mapping, fe, quadrature, quadrature_face);
        CopyData copy_data;
        MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces;

        Timer timer;
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, flags, boundary_worker, nullptr, threads, batch_size);
        timings_jacobian.push_back(timer.wall_time());
      }

      void log(const std::string logger)
      {
        std::stringstream ss;
        ss << "CG Assembler: " << std::endl;
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

      double average_time_residual_assembly()
      {
        double t = 0.;
        double n = timings_residual.size();
        for (const auto &t_ : timings_residual)
          t += t_ / n;
        return t;
      }
      uint num_residuals() const { return timings_residual.size(); }

      double average_time_jacobian_assembly()
      {
        double t = 0.;
        double n = timings_jacobian.size();
        for (const auto &t_ : timings_jacobian)
          t += t_ / n;
        return t;
      }
      uint num_jacobians() const { return timings_jacobian.size(); }

    protected:
      using Base::discretization;
      using Base::dof_handler;
      using Base::fe;
      using Base::mapping;
      using Base::model;

      QGauss<dim> quadrature;
      QGauss<dim - 1> quadrature_face;
      using Base::batch_size;
      using Base::threads;

      SparsityPattern sparsity_pattern_mass;
      SparsityPattern sparsity_pattern_jacobian;
      SparseMatrix<NumberType> mass_matrix;

      std::vector<double> timings_reinit;
      std::vector<double> timings_residual;
      std::vector<double> timings_jacobian;

      using Base::extractor_dof_indices;
    };
  } // namespace CG
} // namespace DiFfRG