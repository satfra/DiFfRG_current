#pragma once

// standard library
#include <sstream>

// DiFfRG
#include <DiFfRG/discretization/FEM/assembler/common.hh>

namespace DiFfRG
{
  namespace dDG
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

        ScratchData(const Mapping<dim> &mapping, const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
                    const Quadrature<dim - 1> &quadrature_face,
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
          comp.resize(fe.n_dofs_per_cell());
          cached_shape_values.resize(fe.n_dofs_per_cell());
          cached_shape_grads.resize(fe.n_dofs_per_cell());
          cached_shape_hessians.resize(fe.n_dofs_per_cell());
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
          comp.resize(scratch_data.fe_values.get_fe().n_dofs_per_cell());
          cached_shape_values.resize(scratch_data.fe_values.get_fe().n_dofs_per_cell());
          cached_shape_grads.resize(scratch_data.fe_values.get_fe().n_dofs_per_cell());
          cached_shape_hessians.resize(scratch_data.fe_values.get_fe().n_dofs_per_cell());
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
        std::vector<double> cached_shape_values;
        std::vector<Tensor<1, dim>> cached_shape_grads;
        std::vector<Tensor<2, dim>> cached_shape_hessians;
      };

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
        std::vector<types::global_dof_index> local_dof_indices;
        std::vector<CopyDataFace_R> face_data;

        template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell)
        {
          cell_residual.reinit(dofs_per_cell);
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);
          face_data.clear();
          face_data.reserve(6);
        }
      };

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
        std::vector<types::global_dof_index> local_dof_indices;
        std::vector<CopyDataFace_J> face_data;

        template <class Iterator> void reinit(const Iterator &cell, uint dofs_per_cell, uint n_extractors)
        {
          cell_jacobian.reinit(dofs_per_cell, dofs_per_cell);
          if (n_extractors > 0) extractor_cell_jacobian.reinit(dofs_per_cell, n_extractors);
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);
          face_data.clear();
          face_data.reserve(6);
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
     * @brief The basic assembler that can be used for any standard DG scheme with flux and source.
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
        static_assert(Components::count_fe_subsystems() == 1, "A dDG model cannot have multiple submodels!");
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
          DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                               /*keep_constrained_dofs = */ true);
          sparsity_pattern_jacobian.copy_from(dsp);
        }

        timings_reinit.push_back(timer.wall_time());
      }

      virtual void rebuild_jacobian_sparsity() override
      {
        // Jacobian sparsity pattern
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
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
       * @brief refinement indicator for adaptivity. Calls the model's cell_indicator and face_indicator functions.
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
        const auto face_worker = [&](const Iterator &t_cell, const uint &f, const uint &sf, const Iterator &t_ncell,
                                     const uint &nf, const unsigned int &nsf, Scratch &scratch_data,
                                     CopyData &copy_data) {
          scratch_data.fe_interface_values.reinit(t_cell, f, sf, t_ncell, nf, nsf);
          const auto &fe_iv = scratch_data.fe_interface_values;
          const auto &fe_iv_s = scratch_data.fe_interface_values.get_fe_face_values(0);
          const auto &fe_iv_n = scratch_data.fe_interface_values.get_fe_face_values(1);

          auto &copy_data_face = copy_data.face_data.emplace_back();
          copy_data_face.cell_indices[0] = t_cell->active_cell_index();
          copy_data_face.cell_indices[1] = t_ncell->active_cell_index();
          copy_data_face.values[0] = 0;
          copy_data_face.values[1] = 0;

          const auto &JxW = fe_iv.get_JxW_values();
          const auto &q_points = fe_iv.get_quadrature_points();
          const auto &q_indices = fe_iv.quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

          auto &solution_s = scratch_data.solution_interface[0];
          auto &solution_n = scratch_data.solution_interface[1];
          auto &solution_grad_s = scratch_data.solution_grad_interface[0];
          auto &solution_grad_n = scratch_data.solution_grad_interface[1];
          auto &solution_hess_s = scratch_data.solution_hess_interface[0];
          auto &solution_hess_n = scratch_data.solution_hess_interface[1];
          fe_iv_s.get_function_values(solution_global, solution_s);
          fe_iv_n.get_function_values(solution_global, solution_n);
          fe_iv_s.get_function_gradients(solution_global, solution_grad_s);
          fe_iv_n.get_function_gradients(solution_global, solution_grad_n);
          fe_iv_s.get_function_hessians(solution_global, solution_hess_s);
          fe_iv_n.get_function_hessians(solution_global, solution_hess_n);

          array<double, 2> local_indicator{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.face_indicator(local_indicator, normals[q_index], x_q,
                                 i_tie(solution_s[q_index], solution_grad_s[q_index], solution_hess_s[q_index]),
                                 i_tie(solution_n[q_index], solution_grad_n[q_index], solution_hess_n[q_index]));

            copy_data_face.values[0] += JxW[q_index] * local_indicator[0] * (1. + t_cell->at_boundary());
            copy_data_face.values[1] += JxW[q_index] * local_indicator[1] * (1. + t_ncell->at_boundary());
          }
        };
        const auto copier = [&](const CopyData &c) {
          for (auto &cdf : c.face_data)
            for (uint j = 0; j < 2; ++j)
              indicator[cdf.cell_indices[j]] += cdf.values[j];
          indicator[c.cell_index] += c.value;
        };

        Scratch scratch_data(mapping, fe, quadrature, quadrature_face);
        CopyData copy_data;
        MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_own_interior_faces_once;

        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, flags, nullptr, face_worker, threads, batch_size);
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

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;
          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_values(solution_global_dot, solution_dot);

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

          array<NumberType, Components::count_fe_functions()> mass{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.mass(mass, x_q, solution[q_index], solution_dot[q_index]);

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = comp[i];
              copy_data.cell_residual(i) += weight * JxW[q_index] *
                                            fe_v.shape_value_component(i, q_index, component_i) *
                                            mass[component_i]; // +phi_i(x_q) * mass(x_q, u_q)
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

          auto &solution = scratch_data.solution;
          auto &solution_grad = scratch_data.solution_grad;
          auto &solution_hess = scratch_data.solution_hess;
          auto &solution_dot = scratch_data.solution_dot;
          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_gradients(solution_global, solution_grad);
          fe_v.get_function_hessians(solution_global, solution_hess);
          fe_v.get_function_values(solution_global_dot, solution_dot);

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

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
              const auto component_i = comp[i];
              copy_data.cell_residual(i) += JxW[q_index] * weight * // dx *
                                            (-scalar_product(fe_v.shape_grad_component(i, q_index, component_i),
                                                             flux[component_i]) // -dphi_i(x_q) * flux(x_q, u_q)
                                             + fe_v.shape_value_component(i, q_index, component_i) *
                                                   source[component_i]); // -phi_i(x_q) * source(x_q, u_q)
              copy_data.cell_residual(i) += weight_mass * JxW[q_index] *
                                            fe_v.shape_value_component(i, q_index, component_i) *
                                            mass[component_i]; // +phi_i(x_q) * mass(x_q, u_q)
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
              const auto component_i = comp[i];
              copy_data.cell_residual(i) +=
                  weight * JxW[q_index] * // dx
                  (fe_fv.shape_value_component(i, q_index, component_i) *
                   scalar_product(numflux[component_i], normals[q_index])); // phi_i(x_q) * numflux(x_q, u_q) * n(x_q)
            }
          }
        };
        const auto face_worker = [&](const Iterator &cell, const uint &f, const unsigned int &sf, const Iterator &ncell,
                                     const unsigned int &nf, const unsigned int &nsf, Scratch &scratch_data,
                                     CopyData &copy_data) {
          scratch_data.fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);
          const auto &fe_iv = scratch_data.fe_interface_values;
          const auto &fe_iv_s = scratch_data.fe_interface_values.get_fe_face_values(0);
          const auto &fe_iv_n = scratch_data.fe_interface_values.get_fe_face_values(1);
          const uint n_dofs = fe_iv.n_current_interface_dofs();

          copy_data.face_data.emplace_back();
          auto &copy_data_face = copy_data.face_data.back();
          copy_data_face.reinit(fe_iv);

          const auto &JxW = fe_iv.get_JxW_values();
          const auto &q_points = fe_iv.get_quadrature_points();
          const auto &q_indices = fe_iv.quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

          auto &solution_s = scratch_data.solution_interface[0];
          auto &solution_n = scratch_data.solution_interface[1];
          auto &solution_grad_s = scratch_data.solution_grad_interface[0];
          auto &solution_grad_n = scratch_data.solution_grad_interface[1];
          auto &solution_hess_s = scratch_data.solution_hess_interface[0];
          auto &solution_hess_n = scratch_data.solution_hess_interface[1];
          fe_iv_s.get_function_values(solution_global, solution_s);
          fe_iv_n.get_function_values(solution_global, solution_n);
          fe_iv_s.get_function_gradients(solution_global, solution_grad_s);
          fe_iv_n.get_function_gradients(solution_global, solution_grad_n);
          fe_iv_s.get_function_hessians(solution_global, solution_hess_s);
          fe_iv_n.get_function_hessians(solution_global, solution_hess_n);

          auto &comp = scratch_data.comp;
          comp.resize(n_dofs);
          for (uint i = 0; i < n_dofs; ++i) {
            const auto &cd_i = fe_iv.interface_dof_to_dof_indices(i);
            comp[i] = cd_i[0] == numbers::invalid_unsigned_int ? fe.system_to_component_index(cd_i[1]).first
                                                               : fe.system_to_component_index(cd_i[0]).first;
          }

          array<Tensor<1, dim, NumberType>, Components::count_fe_functions()> numflux{};
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.numflux(numflux, normals[q_index], x_q,
                          fe_tie(solution_s[q_index], solution_grad_s[q_index], solution_hess_s[q_index],
                                 extracted_data, variables),
                          fe_tie(solution_n[q_index], solution_grad_n[q_index], solution_hess_n[q_index],
                                 extracted_data, variables));

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = comp[i];
              copy_data_face.cell_residual(i) +=
                  weight * JxW[q_index] * // dx
                  (fe_iv.jump_in_shape_values(i, q_index, component_i) *
                   scalar_product(numflux[component_i],
                                  normals[q_index])); // [[phi_i(x_q)]] * numflux(x_q, u_q) * n(x_q)
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_residual, c.local_dof_indices, residual);
          for (auto &cdf : c.face_data)
            constraints.distribute_local_to_global(cdf.cell_residual, cdf.joint_dof_indices, residual);
        };

        Scratch scratch_data(mapping, fe, quadrature, quadrature_face);
        CopyData copy_data;
        MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                          MeshWorker::assemble_own_interior_faces_once;

        Timer timer;
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, flags, boundary_worker, face_worker, threads, batch_size);
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
          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;

          SimpleMatrix<NumberType, Components::count_fe_functions()> j_mass;
          SimpleMatrix<NumberType, Components::count_fe_functions()> j_mass_dot;

          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_values(solution_global_dot, solution_dot);

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template jacobian_mass<0>(j_mass, x_q, solution[q_index], solution_dot[q_index]);
            model.template jacobian_mass<1>(j_mass_dot, x_q, solution[q_index], solution_dot[q_index]);

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = comp[i];
              for (uint j = 0; j < n_dofs; ++j) {
                const auto component_j = comp[j];
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
        std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
        if constexpr (Components::count_extractors() > 0) {
          this->extract(__extracted_data, solution_global, variables, true, true, true);
          if (this->jacobian_extractors(this->extractor_jacobian, solution_global, variables))
            jacobian.reinit(sparsity_pattern_jacobian);
        }
        const auto &extracted_data = __extracted_data;

        const auto cell_worker = [&](const Iterator &cell, Scratch &scratch_data, CopyData &copy_data) {
          scratch_data.fe_values.reinit(cell);
          const auto &fe_v = scratch_data.fe_values;
          const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();

          copy_data.reinit(cell, n_dofs, Components::count_extractors());
          const auto &JxW = fe_v.get_JxW_values();
          const auto &q_points = fe_v.get_quadrature_points();
          const auto &q_indices = fe_v.quadrature_point_indices();

          auto &solution = scratch_data.solution;
          auto &solution_dot = scratch_data.solution_dot;
          auto &solution_grad = scratch_data.solution_grad;
          auto &solution_hess = scratch_data.solution_hess;

          fe_v.get_function_values(solution_global, solution);
          fe_v.get_function_values(solution_global_dot, solution_dot);
          fe_v.get_function_gradients(solution_global, solution_grad);
          fe_v.get_function_hessians(solution_global, solution_hess);

          auto &comp = scratch_data.comp;
          for (uint i = 0; i < n_dofs; ++i)
            comp[i] = fe.system_to_component_index(i).first;

          auto &cached_shape_values = scratch_data.cached_shape_values;
          auto &cached_shape_grads = scratch_data.cached_shape_grads;
          auto &cached_shape_hessians = scratch_data.cached_shape_hessians;

          SimpleMatrix<Tensor<1, dim>, Components::count_fe_functions()> j_flux;
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
            model.template jacobian_flux<0, 0>(
                j_flux, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_source<0, 0>(
                j_source, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_flux_grad<1>(
                j_grad_flux, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_source_grad<1>(
                j_grad_source, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_flux_hess<2>(
                j_hess_flux, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            model.template jacobian_source_hess<2>(
                j_hess_source, x_q,
                fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            if constexpr (Components::count_extractors() > 0) {
              model.template jacobian_flux_extr<3>(
                  j_extr_flux, x_q,
                  fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
              model.template jacobian_source_extr<3>(
                  j_extr_source, x_q,
                  fe_tie(solution[q_index], solution_grad[q_index], solution_hess[q_index], extracted_data, variables));
            }
            model.template jacobian_mass<0>(j_mass, x_q, solution[q_index], solution_dot[q_index]);
            model.template jacobian_mass<1>(j_mass_dot, x_q, solution[q_index], solution_dot[q_index]);

            // Cache shape values, gradients, and hessians for all DoFs at this quadrature point
            for (uint k = 0; k < n_dofs; ++k) {
              cached_shape_values[k] = fe_v.shape_value_component(k, q_index, comp[k]);
              cached_shape_grads[k] = fe_v.shape_grad_component(k, q_index, comp[k]);
              cached_shape_hessians[k] = fe_v.shape_hessian_component(k, q_index, comp[k]);
            }

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = comp[i];
              const auto &shape_value_i = cached_shape_values[i];
              const auto &shape_grad_i = cached_shape_grads[i];
              for (uint j = 0; j < n_dofs; ++j) {
                const auto component_j = comp[j];
                const auto &shape_value_j = cached_shape_values[j];
                const auto &shape_grad_j = cached_shape_grads[j];
                const auto &shape_hessian_j = cached_shape_hessians[j];
                // consolidated contribution: scalar + gradient + hessian + mass
                copy_data.cell_jacobian(i, j) +=
                    weight * JxW[q_index] *
                    (shape_value_j * // dx * phi_j * (
                         (-scalar_product(shape_grad_i,
                                          j_flux(component_i, component_j)) // -dphi_i * jflux
                          + shape_value_i * j_source(component_i, component_j)) // -phi_i * jsource)
                     + scalar_product(shape_grad_j, // gradient contribution
                                      -scalar_product(shape_grad_i,
                                                      j_grad_flux(component_i, component_j)) // -dphi_i * jflux
                                          + shape_value_i *
                                                j_grad_source(component_i, component_j)) // -phi_i * jsource
                     + scalar_product(shape_hessian_j, // hessian contribution
                                      -scalar_product(shape_grad_i,
                                                      j_hess_flux(component_i, component_j)) +
                                          shape_value_i * j_hess_source(component_i, component_j))) +
                    JxW[q_index] * shape_value_j * shape_value_i * // mass contribution
                        (alpha * j_mass_dot(component_i, component_j) + beta * j_mass(component_i, component_j));
              }
              // extractor contribution
              if constexpr (Components::count_extractors() > 0)
                for (uint e = 0; e < Components::count_extractors(); ++e)
                  copy_data.extractor_cell_jacobian(i, e) +=
                      weight * JxW[q_index] * // dx * phi_j * (
                      (-scalar_product(shape_grad_i,
                                       j_extr_flux(component_i, e)) // -dphi_i * jflux
                       + shape_value_i *
                             j_extr_source(component_i, e)); // -phi_i * jsource)
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

          auto &cached_shape_values = scratch_data.cached_shape_values;
          auto &cached_shape_grads = scratch_data.cached_shape_grads;
          auto &cached_shape_hessians = scratch_data.cached_shape_hessians;

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

            // Cache shape values, gradients, and hessians for all DoFs at this quadrature point
            for (uint k = 0; k < n_dofs; ++k) {
              cached_shape_values[k] = fe_fv.shape_value_component(k, q_index, comp[k]);
              cached_shape_grads[k] = fe_fv.shape_grad_component(k, q_index, comp[k]);
              cached_shape_hessians[k] = fe_fv.shape_hessian_component(k, q_index, comp[k]);
            }

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = comp[i];
              const auto &shape_value_i = cached_shape_values[i];
              for (uint j = 0; j < n_dofs; ++j) {
                const auto component_j = comp[j];
                const auto &shape_value_j = cached_shape_values[j];
                const auto &shape_grad_j = cached_shape_grads[j];
                const auto &shape_hessian_j = cached_shape_hessians[j];
                // consolidated contribution: scalar + gradient + hessian
                copy_data.cell_jacobian(i, j) +=
                    weight * JxW[q_index] *
                    (shape_value_j * // dx * phi_j(x_q)
                         (shape_value_i *
                          scalar_product(j_boundary_numflux(component_i, component_j),
                                         normals[q_index])) // phi_i(x_q) * j_numflux(x_q, u_q) * n(x_q)
                     + scalar_product(shape_grad_j, // gradient contribution
                                      shape_value_i *
                                          scalar_product(j_grad_boundary_numflux(component_i, component_j),
                                                         normals[q_index]))
                     + scalar_product(shape_hessian_j, // hessian contribution
                                      shape_value_i *
                                          scalar_product(j_hess_boundary_numflux(component_i, component_j),
                                                         normals[q_index])));
              }
              // extractor contribution
              if constexpr (Components::count_extractors() > 0)
                for (uint e = 0; e < Components::count_extractors(); ++e)
                  copy_data.extractor_cell_jacobian(i, e) +=
                      weight * JxW[q_index] * // dx * phi_j(x_q)
                      (shape_value_i *
                       scalar_product(j_extr_boundary_numflux(component_i, e),
                                      normals[q_index])); // phi_i(x_q) * j_numflux(x_q, u_q) * n(x_q)
            }
          }
        };
        const auto face_worker = [&](const Iterator &cell, const uint &f, const uint &sf, const Iterator &ncell,
                                     const uint &nf, const uint &nsf, Scratch &scratch_data, CopyData &copy_data) {
          scratch_data.fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);
          const auto &fe_iv = scratch_data.fe_interface_values;
          const auto &fe_iv_s = scratch_data.fe_interface_values.get_fe_face_values(0);
          const auto &fe_iv_n = scratch_data.fe_interface_values.get_fe_face_values(1);
          const uint n_dofs = fe_iv.n_current_interface_dofs();

          copy_data.face_data.emplace_back();
          auto &copy_data_face = copy_data.face_data.back();
          copy_data_face.reinit(fe_iv, Components::count_extractors());

          const auto &JxW = fe_iv.get_JxW_values();
          const auto &q_points = fe_iv.get_quadrature_points();
          const auto &q_indices = fe_iv.quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

          auto &solution_s = scratch_data.solution_interface[0];
          auto &solution_n = scratch_data.solution_interface[1];
          auto &solution_grad_s = scratch_data.solution_grad_interface[0];
          auto &solution_grad_n = scratch_data.solution_grad_interface[1];
          auto &solution_hess_s = scratch_data.solution_hess_interface[0];
          auto &solution_hess_n = scratch_data.solution_hess_interface[1];
          fe_iv_s.get_function_values(solution_global, solution_s);
          fe_iv_n.get_function_values(solution_global, solution_n);
          fe_iv_s.get_function_gradients(solution_global, solution_grad_s);
          fe_iv_n.get_function_gradients(solution_global, solution_grad_n);
          fe_iv_s.get_function_hessians(solution_global, solution_hess_s);
          fe_iv_n.get_function_hessians(solution_global, solution_hess_n);

          // Pre-compute interface DoF component indices and face numbers
          auto &comp = scratch_data.comp;
          comp.resize(n_dofs);
          std::vector<uint> face_no_arr(n_dofs);
          std::vector<uint> local_dof_arr(n_dofs);
          for (uint i = 0; i < n_dofs; ++i) {
            const auto &cd_i = fe_iv.interface_dof_to_dof_indices(i);
            face_no_arr[i] = cd_i[0] == numbers::invalid_unsigned_int ? 1 : 0;
            local_dof_arr[i] = cd_i[face_no_arr[i]];
            comp[i] = fe.system_to_component_index(local_dof_arr[i]).first;
          }

          auto &cached_shape_values = scratch_data.cached_shape_values;
          auto &cached_shape_grads = scratch_data.cached_shape_grads;
          auto &cached_shape_hessians = scratch_data.cached_shape_hessians;
          cached_shape_values.resize(n_dofs);
          cached_shape_grads.resize(n_dofs);
          cached_shape_hessians.resize(n_dofs);

          array<SimpleMatrix<Tensor<1, dim, NumberType>, Components::count_fe_functions()>, 2> j_numflux;
          array<SimpleMatrix<Tensor<1, dim, Tensor<1, dim, NumberType>>, Components::count_fe_functions()>, 2>
              j_grad_numflux;
          array<SimpleMatrix<Tensor<1, dim, Tensor<2, dim, NumberType>>, Components::count_fe_functions()>, 2>
              j_hess_numflux;
          array<SimpleMatrix<Tensor<1, dim, NumberType>, Components::count_fe_functions(),
                             Components::count_extractors()>,
                2>
              j_extr_numflux;
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            model.template jacobian_numflux<0, 0>(j_numflux, normals[q_index], x_q,
                                                  fe_tie(solution_s[q_index], solution_grad_s[q_index],
                                                         solution_hess_s[q_index], extracted_data, variables),
                                                  fe_tie(solution_n[q_index], solution_grad_n[q_index],
                                                         solution_hess_n[q_index], extracted_data, variables));
            model.template jacobian_numflux_grad<1>(j_grad_numflux, normals[q_index], x_q,
                                                    fe_tie(solution_s[q_index], solution_grad_s[q_index],
                                                           solution_hess_s[q_index], extracted_data, variables),
                                                    fe_tie(solution_n[q_index], solution_grad_n[q_index],
                                                           solution_hess_n[q_index], extracted_data, variables));
            model.template jacobian_numflux_hess<2>(j_hess_numflux, normals[q_index], x_q,
                                                    fe_tie(solution_s[q_index], solution_grad_s[q_index],
                                                           solution_hess_s[q_index], extracted_data, variables),
                                                    fe_tie(solution_n[q_index], solution_grad_n[q_index],
                                                           solution_hess_n[q_index], extracted_data, variables));
            if constexpr (Components::count_extractors() > 0) {
              model.template jacobian_numflux_extr<3>(j_extr_numflux, normals[q_index], x_q,
                                                      fe_tie(solution_s[q_index], solution_grad_s[q_index],
                                                             solution_hess_s[q_index], extracted_data, variables),
                                                      fe_tie(solution_n[q_index], solution_grad_n[q_index],
                                                             solution_hess_n[q_index], extracted_data, variables));
            }

            // Cache shape values, gradients, and hessians for all interface DoFs at this quadrature point
            for (uint k = 0; k < n_dofs; ++k) {
              cached_shape_values[k] =
                  fe_iv.get_fe_face_values(face_no_arr[k]).shape_value_component(local_dof_arr[k], q_index, comp[k]);
              cached_shape_grads[k] =
                  fe_iv.get_fe_face_values(face_no_arr[k]).shape_grad_component(local_dof_arr[k], q_index, comp[k]);
              cached_shape_hessians[k] =
                  fe_iv.get_fe_face_values(face_no_arr[k]).shape_hessian_component(local_dof_arr[k], q_index, comp[k]);
            }

            for (uint i = 0; i < n_dofs; ++i) {
              const auto component_i = comp[i];
              const auto face_no_i = face_no_arr[i];
              const auto jump_i = fe_iv.jump_in_shape_values(i, q_index, component_i);
              for (uint j = 0; j < n_dofs; ++j) {
                const auto component_j = comp[j];
                const auto face_no_j = face_no_arr[j];
                const auto &shape_value_j = cached_shape_values[j];
                const auto &shape_grad_j = cached_shape_grads[j];
                const auto &shape_hessian_j = cached_shape_hessians[j];
                // consolidated contribution: scalar + gradient + hessian
                copy_data_face.cell_jacobian(i, j) +=
                    weight * JxW[q_index] *
                    (shape_value_j * // dx * phi_j(x_q)
                         (jump_i *
                          scalar_product(j_numflux[face_no_j](component_i, component_j),
                                         normals[q_index])) // [[phi_i(x_q)]] * j_numflux(x_q, u_q)
                     + scalar_product(shape_grad_j, // gradient contribution
                                      jump_i *
                                          scalar_product(j_grad_numflux[face_no_j](component_i, component_j),
                                                         normals[q_index]))
                     + scalar_product(shape_hessian_j, // hessian contribution
                                      jump_i *
                                          scalar_product(j_hess_numflux[face_no_j](component_i, component_j),
                                                         normals[q_index])));
              }
              // extractor contribution
              if constexpr (Components::count_extractors() > 0)
                for (uint e = 0; e < Components::count_extractors(); ++e)
                  copy_data_face.extractor_cell_jacobian(i, e) +=
                      weight * JxW[q_index] * // dx * phi_j(x_q)
                      (jump_i *
                       scalar_product(j_extr_numflux[face_no_i](component_i, e),
                                      normals[q_index])); // [[phi_i(x_q)]] * j_numflux(x_q, u_q)
            }
          }
        };
        const auto copier = [&](const CopyData &c) {
          constraints.distribute_local_to_global(c.cell_jacobian, c.local_dof_indices, jacobian);
          for (auto &cdf : c.face_data) {
            constraints.distribute_local_to_global(cdf.cell_jacobian, cdf.joint_dof_indices, jacobian);
            if constexpr (Components::count_extractors() > 0) {
              FullMatrix<NumberType> extractor_dependence(cdf.joint_dof_indices.size(), extractor_dof_indices.size());
              cdf.extractor_cell_jacobian.mmult(extractor_dependence, this->extractor_jacobian);
              constraints.distribute_local_to_global(extractor_dependence, cdf.joint_dof_indices, extractor_dof_indices,
                                                     jacobian);
            }
          }
          if constexpr (Components::count_extractors() > 0) {
            FullMatrix<NumberType> extractor_dependence(c.local_dof_indices.size(), extractor_dof_indices.size());
            c.extractor_cell_jacobian.mmult(extractor_dependence, this->extractor_jacobian);
            constraints.distribute_local_to_global(extractor_dependence, c.local_dof_indices, extractor_dof_indices,
                                                   jacobian);
          }
        };

        Scratch scratch_data(mapping, fe, quadrature, quadrature_face);
        CopyData copy_data;
        MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                          MeshWorker::assemble_own_interior_faces_once;

        Timer timer;
        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(), cell_worker, copier, scratch_data,
                              copy_data, flags, boundary_worker, face_worker, threads, batch_size);
        timings_jacobian.push_back(timer.wall_time());
      }

      void log(const std::string logger)
      {
        std::stringstream ss;
        ss << "dDG Assembler: " << std::endl;
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
  } // namespace dDG
} // namespace DiFfRG