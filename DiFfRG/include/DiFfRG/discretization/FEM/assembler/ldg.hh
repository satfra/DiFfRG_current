#pragma once

// standard library
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>

// external libraries
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>
#include <DiFfRG/discretization/common/assembly_schedule.hh>
#include <DiFfRG/discretization/common/batched_scatter.hh>
#include <DiFfRG/discretization/common/cell_geometry.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/model/abs_tolerances.hh>
#include <DiFfRG/model/batch.hh>

namespace DiFfRG
{
  namespace LDG
  {
    using namespace dealii;
    using std::array, std::vector, std::unique_ptr;

    template <typename Discretization_, typename Model_>
    class LDGAssemblerBase : public AbstractAssembler<typename Discretization_::VectorType,
                                                      typename Discretization_::SparseMatrixType, Discretization_::dim>
    {
    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;
      using SparseMatrixType = typename Discretization::SparseMatrixType;

      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;
      LDGAssemblerBase(Discretization &discretization, Model &model, const ConfigTree &config)
          : discretization(discretization), model(model), report_port(discretization.report_port()),
            fe(discretization.get_fe()), dof_handler(discretization.get_dof_handler()),
            mapping(discretization.get_mapping()), schedule_overrides(AssemblyScheduleOverrides::from_config(config)),
            EoM_cell(*(dof_handler.active_cell_iterators().end())),
            old_EoM_cell(*(dof_handler.active_cell_iterators().end())),
            old_extractor_cell(*(dof_handler.active_cell_iterators().end())),
            EoM_config(DiFfRG::internal::resolve_eom_config(dof_handler, Config::EoMConfig(config)))
      {
        // reinit() refreshes this, but a derived assembler is not obliged to call it before its
        // first mesh_loop, and an unset schedule would be a zero queue length.
        update_assembly_schedules();
      }

      virtual IndexSet get_differential_indices() const override
      {
        ComponentMask component_mask(model.template differential_components<dim>());
        return DoFTools::extract_dofs(dof_handler, component_mask);
      }

      /**
       * @brief Per-dof absolute tolerances from Model::abs_tolerances, evaluated at each dof's support point on
       * the FE solution (values, gradients, hessians of the primary FE functions; not the LDG auxiliaries).
       * @see AbstractAssembler::local_abs_tolerances. Serial vectors only.
       */
      virtual bool local_abs_tolerances(VectorType &atol, const VectorType &solution, double abs_tol,
                                        double rel_tol) const override
      {
        constexpr uint n = Components::count_fe_functions(0);
        using Values = std::array<NumberType, n>;
        using Gradients = std::array<Tensor<1, dim, NumberType>, n>;
        using Hessians = std::array<Tensor<2, dim, NumberType>, n>;
        using Solution = named_tuple<std::tuple<Values &, Gradients &, Hessians &>,
                                     StringSet<"fe_functions", "fe_derivatives", "fe_hessians">>;
        if constexpr (!def::HasAbsTolerances<Model, dim, Solution, n> ||
                      !std::is_same_v<VectorType, dealii::Vector<NumberType>>)
          return false;
        else {
          atol.reinit(dof_handler.n_dofs());
          const dealii::Quadrature<dim> support(fe.get_unit_support_points());
          FEValues<dim> fe_v(mapping, fe, support,
                             update_values | update_gradients | update_hessians | update_quadrature_points);
          std::vector<Vector<NumberType>> vals(support.size(), Vector<NumberType>(n));
          std::vector<std::vector<Tensor<1, dim, NumberType>>> grads(support.size(),
                                                                     std::vector<Tensor<1, dim, NumberType>>(n));
          std::vector<std::vector<Tensor<2, dim, NumberType>>> hess(support.size(),
                                                                    std::vector<Tensor<2, dim, NumberType>>(n));
          std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
          Values v{};
          Gradients g{};
          Hessians h{};
          std::array<double, n> point_atol{};
          for (const auto &cell : dof_handler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;
            fe_v.reinit(cell);
            fe_v.get_function_values(solution, vals);
            fe_v.get_function_gradients(solution, grads);
            fe_v.get_function_hessians(solution, hess);
            cell->get_dof_indices(dofs);
            // the quadrature is the list of dof support points, so point q belongs to dof q
            for (uint q = 0; q < support.size(); ++q) {
              for (uint c = 0; c < n; ++c) {
                v[c] = vals[q][c];
                g[c] = grads[q][c];
                h[c] = hess[q][c];
              }
              point_atol.fill(abs_tol);
              model.abs_tolerances(point_atol, fe_v.quadrature_point(q), Solution(std::tie(v, g, h)), abs_tol, rel_tol);
              atol[dofs[q]] = point_atol[fe.system_to_component_index(q).first];
            }
          }
          return true;
        }
      }

      const auto &get_discretization() const { return discretization; }
      auto &get_discretization() { return discretization; }

      virtual SnapshotSpatialState capture_snapshot_state(const VectorType &spatial_replica) const override
      {
        return DiFfRG::internal::capture_cellwise_state(dof_handler, spatial_replica);
      }

      virtual void restore_snapshot_state(const SnapshotSpatialState &state, VectorType &spatial) override
      {
        DiFfRG::internal::restore_spatial_state(state, discretization, *this, spatial);
      }

      virtual void save_model_state(ModelState &state) const override
      {
        DiFfRG::internal::save_model_state(model, state);
      }

      virtual bool load_model_state(const ModelState &state) override
      {
        return DiFfRG::internal::load_model_state(model, state);
      }

      virtual void reinit() override
      {
        const auto metadata = DiFfRG::internal::build_affine_constraint_metadata<Components, dim>(discretization);
        const AffineConstraintContext<Components, dim> context(metadata);

        auto &constraints = discretization.get_constraints();
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        DiFfRG::internal::apply_model_affine_constraints(model, constraints, context);
        constraints.close();

        update_assembly_schedules();
      }

      virtual void rebuild_jacobian_sparsity() = 0;

      virtual void set_time(double t) override { model.set_time(t); }

      virtual void refinement_indicator(Vector<double> & /*indicator*/, const VectorType & /*solution*/) = 0;

      double average_time_variable_residual_assembly()
      {
        double t = 0.;
        double n = timings_variable_residual.size();
        for (const auto &t_ : timings_variable_residual)
          t += t_ / n;
        return t;
      }
      uint num_variable_residuals() const { return timings_variable_residual.size(); }

      double average_time_variable_jacobian_assembly()
      {
        double t = 0.;
        double n = timings_variable_jacobian.size();
        for (const auto &t_ : timings_variable_jacobian)
          t += t_ / n;
        return t;
      }
      uint num_variable_jacobians() const { return timings_variable_jacobian.size(); }

    protected:
      Discretization &discretization;
      Model &model;
      ReportPort report_port;
      const FiniteElement<dim> &fe;
      const DoFHandler<dim> &dof_handler;
      const Mapping<dim> &mapping;

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
        report_port.info("FEM: Assembling {} cells on {} threads -- {}x{} workers/cells for a cheap cell loop, "
                         "{}x{} for an integral one.",
                         n_owned_cells, threads, cheap.queue_length, cheap.chunk_size, integral.queue_length,
                         integral.chunk_size);
      }

      /// @see FEMAssembler::extractor_raw_potential
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

      /// @see FEMAssembler::resolve_extractor_point
      std::pair<Point<dim>, typename DoFHandler<dim>::cell_iterator>
      resolve_extractor_point(const Point<dim> &EoM_point, const typename DoFHandler<dim>::cell_iterator &EoM_cell_,
                              [[maybe_unused]] const VectorType &solution_global) const
      {
        if constexpr (HasExtractorPoint<Model, dim, NumberType>) {
          const auto sample = make_solution_sample(solution_global, dof_handler, mapping);
          const auto point = model.template extractor_point<dim, NumberType>(EoM_point, sample);
          if (point == EoM_point) return {EoM_point, EoM_cell_};
          return {point, GridTools::find_active_cell_around_point(dof_handler, point)};
        } else
          return {EoM_point, EoM_cell_};
      }

      /// @see FEMAssembler::n_owned_cells
      uint n_owned_cells = 0;
      const AssemblyScheduleOverrides schedule_overrides;

      mutable typename DoFHandler<dim>::cell_iterator EoM_cell;
      typename DoFHandler<dim>::cell_iterator old_EoM_cell;
      /// @see FEMAssembler::old_extractor_cell
      typename DoFHandler<dim>::cell_iterator old_extractor_cell;
      const Config::EoMConfig EoM_config;
      mutable Point<dim> EoM;
      mutable std::optional<Point<dim>> EoM_minimum_guess;
      /// @see FEMAssembler::potential_cache
      mutable DiFfRG::internal::PotentialSystemCache<dim, NumberType> potential_cache;
      FullMatrix<NumberType> extractor_jacobian;
      FullMatrix<NumberType> extractor_jacobian_u;
      FullMatrix<NumberType> extractor_jacobian_du;
      FullMatrix<NumberType> extractor_jacobian_ddu;
      std::vector<types::global_dof_index> extractor_dof_indices;

      std::vector<double> timings_variable_residual;
      std::vector<double> timings_variable_jacobian;
    };

    namespace internal
    {
      /**
       * @brief Class to hold data for each assembly thread, i.e. FEValues for cells, interfaces, as well as
       * pre-allocated data structures for the solutions
       */
      template <typename Discretization> struct ScratchData {
        static constexpr uint dim = Discretization::dim;
        using NumberType = typename Discretization::NumberType;
        static constexpr uint n_fe_subsystems = Discretization::Components::count_fe_subsystems();
        using Iterator = typename DoFHandler<dim>::active_cell_iterator;
        using t_Iterator = typename Triangulation<dim>::active_cell_iterator;

        ScratchData(const Mapping<dim> &mapping, const vector<const DoFHandler<dim> *> &dofh,
                    const dealii::Quadrature<dim> &quadrature, const dealii::Quadrature<dim - 1> &quadrature_face,
                    const UpdateFlags update_flags = update_values | update_gradients | update_quadrature_points |
                                                     update_JxW_values,
                    const UpdateFlags interface_update_flags = update_values | update_gradients |
                                                               update_quadrature_points | update_JxW_values |
                                                               update_normal_vectors)
        {
          AssertThrow(dofh.size() >= n_fe_subsystems,
                      StandardExceptions::ExcDimensionMismatch(dofh.size(), n_fe_subsystems));

          for (uint i = 0; i < n_fe_subsystems; ++i) {
            const auto &fe = dofh[i]->get_fe();
            fe_values[i] = std::make_unique<FEValues<dim>>(mapping, fe, quadrature, update_flags);
            fe_interface_values[i] =
                std::make_unique<FEInterfaceValues<dim>>(mapping, fe, quadrature_face, interface_update_flags);
            fe_boundary_values[i] =
                std::make_unique<FEFaceValues<dim>>(mapping, fe, quadrature_face, interface_update_flags);

            n_components[i] = fe.n_components();
            solution[i].resize(quadrature.size(), Vector<NumberType>(n_components[i]));
            solution_interface[0][i].resize(quadrature_face.size(), Vector<NumberType>(n_components[i]));
            solution_interface[1][i].resize(quadrature_face.size(), Vector<NumberType>(n_components[i]));

            const uint n_dofs_per_cell = fe.n_dofs_per_cell();
            comp[i].resize(n_dofs_per_cell);
            for (uint d = 0; d < n_dofs_per_cell; ++d)
              comp[i][d] = fe.system_to_component_index(d).first;

            cell[i] = dofh[i]->begin_active();
            ncell[i] = dofh[i]->begin_active();
          }
          solution_dot.resize(quadrature.size(), Vector<NumberType>(n_components[0]));
        }

        ScratchData(const ScratchData<Discretization> &scratch_data)
        {
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            const auto &old_fe = scratch_data.fe_values[i];
            const auto &old_fe_i = scratch_data.fe_interface_values[i];
            const auto &old_fe_b = scratch_data.fe_boundary_values[i];

            fe_values[i] = unique_ptr<FEValues<dim>>(new FEValues<dim>(
                old_fe->get_mapping(), old_fe->get_fe(), old_fe->get_quadrature(), old_fe->get_update_flags()));
            fe_interface_values[i] = unique_ptr<FEInterfaceValues<dim>>(new FEInterfaceValues<dim>(
                old_fe_i->get_mapping(), old_fe_i->get_fe(), old_fe_i->get_quadrature(), old_fe_i->get_update_flags()));
            fe_boundary_values[i] = unique_ptr<FEFaceValues<dim>>(new FEFaceValues<dim>(
                old_fe_b->get_mapping(), old_fe_b->get_fe(), old_fe_b->get_quadrature(), old_fe_b->get_update_flags()));

            n_components[i] = scratch_data.n_components[i];
            comp[i] = scratch_data.comp[i];
            solution[i].resize(scratch_data.solution[i].size(), Vector<NumberType>(n_components[i]));
            solution_interface[0][i].resize(scratch_data.solution_interface[0][i].size(),
                                            Vector<NumberType>(n_components[i]));
            solution_interface[1][i].resize(scratch_data.solution_interface[1][i].size(),
                                            Vector<NumberType>(n_components[i]));

            cell[i] = scratch_data.cell[i];
            ncell[i] = scratch_data.ncell[i];
          }
          solution_dot.resize(scratch_data.solution_dot.size(), Vector<NumberType>(n_components[0]));
        }

        const auto &new_fe_values(const t_Iterator &t_cell)
        {
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            cell[i]->copy_from(*t_cell);
            fe_values[i]->reinit(cell[i]);
          }
          return fe_values;
        }
        const auto &new_fe_interface_values(const t_Iterator &t_cell, uint f, uint sf, const t_Iterator &t_ncell,
                                            uint nf, unsigned int nsf)
        {
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            cell[i]->copy_from(*t_cell);
            ncell[i]->copy_from(*t_ncell);
            fe_interface_values[i]->reinit(cell[i], f, sf, ncell[i], nf, nsf);
          }
          return fe_interface_values;
        }
        const auto &new_fe_boundary_values(const t_Iterator &t_cell, uint face_no)
        {
          for (uint i = 0; i < n_fe_subsystems; ++i) {
            cell[i]->copy_from(*t_cell);
            fe_boundary_values[i]->reinit(cell[i], face_no);
          }
          return fe_boundary_values;
        }

        array<uint, n_fe_subsystems> n_components;
        array<Iterator, n_fe_subsystems> cell;
        array<Iterator, n_fe_subsystems> ncell;

        array<unique_ptr<FEValues<dim>>, n_fe_subsystems> fe_values;
        array<unique_ptr<FEInterfaceValues<dim>>, n_fe_subsystems> fe_interface_values;
        array<unique_ptr<FEFaceValues<dim>>, n_fe_subsystems> fe_boundary_values;

        array<std::vector<uint>, n_fe_subsystems> comp;

        array<vector<Vector<NumberType>>, n_fe_subsystems> solution;
        vector<Vector<NumberType>> solution_dot;
        array<array<vector<Vector<NumberType>>, n_fe_subsystems>, 2> solution_interface;
      };

      template <typename NumberType> struct CopyData_I {
        struct CopyFaceData_I {
          std::array<uint, 2> cell_indices;
          std::array<double, 2> values;
        };
        std::vector<CopyFaceData_I> face_data;
        double value = 0.;
        uint cell_index = 0;
      };
    } // namespace internal

    /**
     * @brief The LDG assembler: the FE functions (level 0) plus up to three LDG levels, level k built from
     * level k - 1 by the model's ldg_flux / ldg_source and a mass-matrix solve.
     *
     * Each level, and the main level, is assembled in three phases, as in CG::Assembler:
     *  1. gather: the values of the previous level (main level: of all levels) at every quadrature point of the
     *     cells, boundary faces and interior faces (both traces), into a batch;
     *  2. evaluate: ldg_flux_source_batch<k> (main level: evaluate_batch) and the numerical fluxes over the whole
     *     batch, or their seed-stacked AD jacobians, see model/batch.hh; each interior face once;
     *  3. scatter: every cell contracts its cell term and its share of each of its faces into its own rows
     *     (internal::ColoredCells).
     *
     * The jacobian is J = weight (J_uu + sum_k J_ug[k] J_gu[k]) + mass terms, with J_gu[k] = d(level k)/du built from
     * the level jacobians by the chain rule and cached while the model declares them constant.
     *
     * Config: /discretization/batched/max_stacked_points bounds the size of one AD evaluation of the
     * jacobian (default: 256 MB of AD inputs), see DiFfRG::internal::seed_stacked_jacobian.
     */
    template <typename Discretization_,
              typename Model_ = typename DiFfRG::internal::assembler_model_of<Discretization_>::type>
    class Assembler : public LDGAssemblerBase<Discretization_, Model_>
    {
      using Base = LDGAssemblerBase<Discretization_, Model_>;

    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;
      using SparseMatrixType = typename Discretization::SparseMatrixType;

      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;
      static constexpr uint stencil = Components::count_fe_subsystems();
      static constexpr uint n_levels = stencil;
      static constexpr size_t n_fe = Components::count_fe_functions(0);
      static constexpr size_t n_extr = Components::count_extractors();
      using Extractors = std::array<NumberType, n_extr>;
      /// The batch of the main level: the values of all levels.
      using MainBatch = LDGPointBatch<dim, NumberType, Components, Extractors, VectorType>;
      static constexpr size_t n_all = MainBatch::n_fe_functions;
      /// The batch level `from` hands to the construction of level from + 1: its values, nothing shared.
      template <uint from>
      using LevelBatch =
          PointBatch<dim, NumberType, Components::count_fe_functions(from), std::array<NumberType, 0>, VectorType>;

      /// Wall time of the assembly phases of the main level, summed over all calls. `gather` includes building
      /// the LDG levels (and, for the jacobian, their jacobians).
      struct PhaseTimes {
        double gather = 0., evaluate = 0., scatter = 0.;
        uint calls = 0;
      };

    private:
      template <typename... T> auto fe_more_conv(std::tuple<T &...> &t) const
      {
        if constexpr (stencil == 2)
          return named_tuple<std::tuple<T &...>,
                             StringSet<"fe_functions", "LDG1", "fe_derivatives", "fe_hessians", "extractors",
                                       "variables", "potential", "potential_gradient", "potential_hessian">>(t);
        else if constexpr (stencil == 3)
          return named_tuple<std::tuple<T &...>,
                             StringSet<"fe_functions", "LDG1", "LDG2", "fe_derivatives", "fe_hessians", "extractors",
                                       "variables", "potential", "potential_gradient", "potential_hessian">>(t);
        else if constexpr (stencil == 4)
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "LDG1", "LDG2", "LDG3", "fe_derivatives",
                                                           "fe_hessians", "extractors", "variables", "potential",
                                                           "potential_gradient", "potential_hessian">>(t);
        else
          throw std::runtime_error("Only <= 3 LDG subsystems are supported.");
      }

      template <typename... T> auto ref_conv(std::tuple<T &...> &t) const
      {
        if constexpr (stencil == 2)
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "LDG1">>(t);
        else if constexpr (stencil == 3)
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "LDG1", "LDG2">>(t);
        else if constexpr (stencil == 4)
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "LDG1", "LDG2", "LDG3">>(t);
        else
          throw std::runtime_error("Only <= 3 LDG subsystems are supported.");
      }

    public:
      Assembler(Discretization &discretization, Model &model, const ConfigTree &config)
          : Base(discretization, model, config),
            quadrature(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
            quadrature_face(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
            dof_handler_list(discretization.get_dof_handler_list()),
            max_stacked_points(config.get_uint("/discretization/batched/max_stacked_points", default_stacked_points()))
      {
        static_assert(Components::count_fe_subsystems() > 1, "LDG must have a submodel with index 1.");
        reinit();
      }

      virtual void reinit_vector(VectorType &vec) const override { vec.reinit(dof_handler.n_dofs()); }
      // LDG stays on the serial policy (see LDG::Discretization for why), so this is the plain
      // pattern-based reinit rather than a policy call.
      virtual void reinit_matrix(SparseMatrixType &matrix) const override
      {
        matrix.reinit(get_sparsity_pattern_jacobian());
      }
      virtual MPI_Comm get_communicator() const override { return discretization.get_communicator(); }
      virtual void reinit_solution_view(SolutionView<VectorType> &view) const override
      {
        view.reinit(discretization.get_locally_owned_dofs(), discretization.get_locally_relevant_dofs(),
                    discretization.get_communicator());
      }

      /**
       * @brief Attach all intermediate (ldg) vectors to the data output
       *
       * @param data_out The scoped output frame
       * @param sol The current global solution
       */
      virtual void attach_data_output(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                                      const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                      const VectorType &residual = VectorType()) override
      {
        rebuild_ldg_vectors(solution);
        readouts(data_out, solution, variables);

        const auto fe_function_names = Components::FEFunction_Descriptor::get_names_vector();
        std::vector<std::string> fe_function_names_residual;
        for (const auto &name : fe_function_names)
          fe_function_names_residual.push_back(name + "_residual");
        std::vector<std::string> fe_function_names_dot;
        for (const auto &name : fe_function_names)
          fe_function_names_dot.push_back(name + "_dot");

        auto fe_out = data_out.fields();
        fe_out.attach(*dof_handler_list[0], solution, fe_function_names);
        if (dt_solution.size() > 0) fe_out.attach(dof_handler, dt_solution, fe_function_names_dot);
        if (residual.size() > 0) fe_out.attach(dof_handler, residual, fe_function_names_residual);
        for (uint k = 1; k < Components::count_fe_subsystems(); ++k) {
          sol_vector_vec_tmp[k] = sol_vector[k];
          fe_out.attach(*dof_handler_list[k], sol_vector_vec_tmp[k], "LDG" + std::to_string(k));
        }
      }

      virtual void reinit() override
      {
        const auto init_mass = [&](uint i) {
          // build the sparsity of the mass matrix and mass matrix of all ldg levels
          auto dofs_per_component = DoFTools::count_dofs_per_fe_component(*(dof_handler_list[i]));

          if (i == 0) {
            auto n_fe = dofs_per_component.size();
            for (uint j = 1; j < n_fe; ++j)
              if (dofs_per_component[j] != dofs_per_component[0])
                throw std::runtime_error("For LDG the FE basis of all systems must be equal!");

            BlockDynamicSparsityPattern dsp(n_fe, n_fe);
            for (uint i = 0; i < n_fe; ++i)
              for (uint j = 0; j < n_fe; ++j)
                dsp.block(i, j).reinit(dofs_per_component[0], dofs_per_component[0]);
            dsp.collect_sizes();
            DoFTools::make_sparsity_pattern(*(dof_handler_list[0]), dsp, discretization.get_constraints(0), true);
            sparsity_pattern_mass.copy_from(dsp);

            component_mass_matrix_inverse.reinit(sparsity_pattern_mass.block(0, 0));
            mass_matrix.reinit(sparsity_pattern_mass);

            MatrixCreator::create_mass_matrix(*(dof_handler_list[0]), quadrature, mass_matrix,
                                              (Function<dim, NumberType> *)nullptr, discretization.get_constraints(0));
            build_inverse(mass_matrix.block(0, 0), component_mass_matrix_inverse);
            sol_block.reinit(dofs_per_component);
          } else {
            sol_vector[i].reinit(dofs_per_component);
            sol_vector_tmp[i].reinit(dofs_per_component);
            ldg_matrix_built[i] = false;
          }
        };

        const auto init_jacobian = [&](uint i) {
          // build the jacobian and subjacobians
          if (i == 0) {
            build_ldg_sparsity(sparsity_pattern_jacobian, *(dof_handler_list[0]), *(dof_handler_list[0]), stencil,
                               true);
            for (uint k = 1; k < Components::count_fe_subsystems(); ++k)
              jacobian_tmp[k].reinit(sparsity_pattern_jacobian);
          } else {
            build_ldg_sparsity(sparsity_pattern_ug[i], *(dof_handler_list[0]), *(dof_handler_list[i]), 1);
            j_ug[i].reinit(sparsity_pattern_ug[i]);
          }
        };

        auto init_ldg = [&](uint i) {
          // build the subjacobian sparsity patterns of all matrices that contribute to the jacobian = uu + ug*gu
          build_ldg_sparsity(sparsity_pattern_gu[i], *(dof_handler_list[i]), *(dof_handler_list[0]), i);
          j_gu[i].reinit(sparsity_pattern_gu[i]);

          // these are the "in-between" dependencies of the ldg levels
          build_ldg_sparsity(sparsity_pattern_wg[i], *(dof_handler_list[i]), *(dof_handler_list[i - 1]), 1);
          j_wg[i].reinit(sparsity_pattern_wg[i]);
          j_wg_tmp[i].reinit(sparsity_pattern_wg[i]);
        };

        Timer timer;

        Base::reinit();

        vector<std::thread> init_threads;
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i)
          init_threads.emplace_back(std::thread(init_mass, i));
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i)
          init_threads.emplace_back(std::thread(init_jacobian, i));
        for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
          init_threads.emplace_back(std::thread(init_ldg, i));
        for (auto &t : init_threads)
          t.join();

        setup_cells();
        timings_reinit.push_back(timer.wall_time());
      }

      virtual void rebuild_jacobian_sparsity() override
      {
        build_ldg_sparsity(sparsity_pattern_jacobian, *(dof_handler_list[0]), *(dof_handler_list[0]), stencil, true);
        for (uint k = 1; k < Components::count_fe_subsystems(); ++k)
          jacobian_tmp[k].reinit(sparsity_pattern_jacobian);
      }

      virtual void refinement_indicator(Vector<double> &indicator, const VectorType &solution_global) override
      {
        using Iterator = typename Triangulation<dim>::active_cell_iterator;
        using Scratch = internal::ScratchData<Discretization>;
        using CopyData = internal::CopyData_I<NumberType>;

        const auto cell_worker = [&](const Iterator &t_cell, Scratch &scratch_data, CopyData &copy_data) {
          const auto &fe_v = scratch_data.new_fe_values(t_cell);
          copy_data.cell_index = t_cell->active_cell_index();
          copy_data.value = 0;

          const auto &JxW = fe_v[0]->get_JxW_values();
          const auto &q_points = fe_v[0]->get_quadrature_points();
          const auto &q_indices = fe_v[0]->quadrature_point_indices();

          auto &solution = scratch_data.solution;
          fe_v[0]->get_function_values(solution_global, solution[0]);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i)
            fe_v[i]->get_function_values(sol_vector[i], solution[i]);

          double local_indicator = 0.;
          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            auto sol_q = local_sol_q(solution, q_index);
            model.cell_indicator(local_indicator, x_q, ref_conv(sol_q));

            copy_data.value += JxW[q_index] * local_indicator;
          }
        };
        const auto face_worker = [&](const Iterator &t_cell, const uint &f, const uint &sf, const Iterator &t_ncell,
                                     const uint &nf, const unsigned int &nsf, Scratch &scratch_data,
                                     CopyData &copy_data) {
          const auto &fe_iv = scratch_data.new_fe_interface_values(t_cell, f, sf, t_ncell, nf, nsf);

          auto &copy_data_face = copy_data.face_data.emplace_back();
          copy_data_face.cell_indices[0] = t_cell->active_cell_index();
          copy_data_face.cell_indices[1] = t_ncell->active_cell_index();
          copy_data_face.values[0] = 0;
          copy_data_face.values[1] = 0;

          const auto &JxW = fe_iv[0]->get_JxW_values();
          const auto &q_points = fe_iv[0]->get_quadrature_points();
          const auto &q_indices = fe_iv[0]->quadrature_point_indices();
          const std::vector<Tensor<1, dim>> &normals = fe_iv[0]->get_normal_vectors();
          array<double, 2> local_indicator{};

          auto &solution = scratch_data.solution_interface;
          fe_iv[0]->get_fe_face_values(0).get_function_values(solution_global, solution[0][0]);
          fe_iv[0]->get_fe_face_values(1).get_function_values(solution_global, solution[1][0]);
          for (uint i = 1; i < Components::count_fe_subsystems(); ++i) {
            fe_iv[i]->get_fe_face_values(0).get_function_values(sol_vector[i], solution[0][i]);
            fe_iv[i]->get_fe_face_values(1).get_function_values(sol_vector[i], solution[1][i]);
          }

          for (const auto &q_index : q_indices) {
            const auto &x_q = q_points[q_index];
            auto sol_q_s = local_sol_q(solution[0], q_index);
            auto sol_q_n = local_sol_q(solution[1], q_index);
            model.face_indicator(local_indicator, normals[q_index], x_q, ref_conv(sol_q_s), ref_conv(sol_q_n));

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

        const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values;
        Scratch scratch_data(mapping, dof_handler_list, quadrature, quadrature_face, update_flags);
        CopyData copy_data;
        MeshWorker::AssembleFlags assemble_flags =
            MeshWorker::assemble_own_cells | MeshWorker::assemble_own_interior_faces_once;

        rebuild_ldg_vectors(solution_global);
        const auto schedule = schedule_for(assembly_cost::local_fe);
        MeshWorker::mesh_loop(locally_owned_cells(dof_handler), cell_worker, copier, scratch_data, copy_data,
                              assemble_flags, nullptr, face_worker, schedule.queue_length, schedule.chunk_size);
      }

      virtual const BlockSparsityPattern &get_sparsity_pattern_jacobian() const override
      {
        return sparsity_pattern_jacobian;
      }
      virtual const BlockSparseMatrix<NumberType> &get_mass_matrix() const override { return mass_matrix; }

      /**
       * @brief Construct the mass
       *
       * @param residual The result is stored here.
       * @param solution_global The current global solution.
       * @param weight A factor to multiply the whole residual with.
       */
      virtual void mass(VectorType &residual, const VectorType &solution_global, const VectorType &solution_global_dot,
                        NumberType weight) override
      {
        scatter_residual(residual, [&](const size_t, Scratch &s, Vector<NumberType> &r) {
          const auto &fe_v = *s.fe_v[0];
          fe_v.get_function_values(solution_global, s.values[0]);
          fe_v.get_function_values(solution_global_dot, s.values_dot);
          array<NumberType, n_fe> m{};
          for (const auto &q : fe_v.quadrature_point_indices()) {
            model.mass(m, fe_v.quadrature_point(q), s.values[0][q], s.values_dot[q]);
            for (uint i = 0; i < r.size(); ++i)
              r(i) += weight * fe_v.JxW(q) * fe_v.shape_value_component(i, q, s.comp[0][i]) * m[s.comp[0][i]];
          }
        });
      }

      /**
       * @brief Construct the system residual, i.e. Res = grad(flux) - source
       *
       * @param residual The result is stored here.
       * @param solution_global The current global solution.
       * @param weight A factor to multiply the whole residual with.
       */
      virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                            const VectorType &solution_global_dot, NumberType weight_mass,
                            const VectorType &variables = VectorType()) override
      {
        Timer timer, phase;
        // Find the EoM and extract whatever data is needed for the model; extract() builds the LDG levels.
        Extractors extracted_data{{}};
        if constexpr (n_extr > 0)
          this->extract(extracted_data, solution_global, variables, true, false, true);
        else
          rebuild_ldg_vectors(solution_global);
        gather_main(solution_global, extracted_data, variables);
        residual_times.gather += phase.wall_time();

        phase.restart();
        cell_result.reinit(cell_batch.size(), Term::flux | Term::source);
        model.evaluate_batch(cell_result, cell_batch);
        boundary_result.reinit(boundary_batch.size(), Term::flux);
        evaluate_boundary_numflux(model, boundary_result, boundary_normals, boundary_batch);
        face_result.reinit(face_batch[0].size(), Term::flux);
        evaluate_numflux(model, face_result, face_normals, face_batch[0], face_batch[1]);
        residual_times.evaluate += phase.wall_time();

        phase.restart();
        scatter_residual(residual, [&](const size_t k, Scratch &s, Vector<NumberType> &r) {
          const auto &fe_v = *s.fe_v[0];
          const auto &comp = s.comp[0];
          fe_v.get_function_values(solution_global, s.values[0]);
          fe_v.get_function_values(solution_global_dot, s.values_dot);
          array<NumberType, n_fe> mass{};
          for (const auto &q : fe_v.quadrature_point_indices()) {
            const size_t p = k * n_q + q;
            model.mass(mass, fe_v.quadrature_point(q), s.values[0][q], s.values_dot[q]);
            for (uint i = 0; i < r.size(); ++i) {
              const auto c = comp[i];
              Tensor<1, dim, NumberType> flux;
              for (uint d = 0; d < dim; ++d)
                flux[d] = cell_result.flux(c, d)[p];
              r(i) += fe_v.JxW(q) * (weight * (-scalar_product(fe_v.shape_grad_component(i, q, c), flux) +
                                               fe_v.shape_value_component(i, q, c) * cell_result.source(c)[p]) +
                                     weight_mass * fe_v.shape_value_component(i, q, c) * mass[c]);
            }
          }
          add_face_terms(s, k, 0, r, boundary_result, face_result, weight);
        });
        residual_times.scatter += phase.wall_time();
        ++residual_times.calls;
        timings_residual.push_back(timer.wall_time());
      }

      virtual void jacobian_mass(BlockSparseMatrix<NumberType> &jacobian, const VectorType &solution_global,
                                 const VectorType &solution_global_dot, NumberType alpha = 1.,
                                 NumberType beta = 1.) override
      {
        scatter_jacobian(jacobian, false, [&](const size_t, Scratch &s, LocalData &out) {
          add_mass_jacobian(s, out.blocks[0][0], solution_global, solution_global_dot, alpha, beta);
        });
      }

      /**
       * @brief Construct the system jacobian, i.e. dRes/du
       *
       * @param jacobian The result is stored here.
       * @param solution_global The current global solution.
       * @param weight A factor to multiply the whole jacobian with.
       */
      virtual void jacobian(BlockSparseMatrix<NumberType> &jacobian, const VectorType &solution_global,
                            NumberType weight, const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                            const VectorType &variables = VectorType()) override
      {
        Timer timer, phase;
        // Find the EoM and extract whatever data is needed for the model; extract() builds the LDG levels.
        Extractors extracted_data{{}};
        if constexpr (n_extr > 0) {
          this->extract(extracted_data, solution_global, variables, true, true, true);
          if (this->jacobian_extractors(this->extractor_jacobian, solution_global, variables))
            jacobian.reinit(sparsity_pattern_jacobian);
        } else
          rebuild_ldg_vectors(solution_global);
        // The levels are current now, so the jacobians built from them are, too.
        constexpr_for<1, n_levels, 1>([&](auto k) {
          if (!ldg_matrix_built[k] || !model.get_components().jacobians_constant(k, k - 1))
            rebuild_ldg_jacobian<k>(solution_global);
        });
        gather_main(solution_global, extracted_data, variables);
        jacobian_times.gather += phase.wall_time();

        phase.restart();
        evaluate_flux_source_jacobian(model, cell_jacobians, cell_batch, max_stacked_points, cell_workspace);
        evaluate_boundary_numflux_jacobian(model, boundary_jacobians, boundary_normals, boundary_batch,
                                           max_stacked_points, boundary_workspace);
        evaluate_numflux_jacobian(model, face_jacobians, face_normals, face_batch[0], face_batch[1], max_stacked_points,
                                  face_workspace);
        require_finite(cell_jacobians);
        require_finite(boundary_jacobians);
        require_finite(face_jacobians);
        jacobian_times.evaluate += phase.wall_time();

        phase.restart();
        for (uint k = 1; k < n_levels; ++k)
          j_ug[k] = 0;
        scatter_jacobian(jacobian, true, [&](const size_t k, Scratch &s, LocalData &out) {
          add_mass_jacobian(s, out.blocks[0][0], solution_global, solution_global_dot, alpha, beta);
          const auto &fe_v = *s.fe_v[0];
          for (const auto &q : fe_v.quadrature_point_indices()) {
            s.cache(0, [&](const uint l) -> const FEValuesBase<dim> & { return *s.fe_v[l]; }, q);
            add_cell_jacobian(s, out, cell_jacobians[k * n_q + q], weight * fe_v.JxW(q));
          }
          for (size_t f = topology.boundary_begin[k]; f < topology.boundary_begin[k + 1]; ++f) {
            const auto &fe_fv = reinit_boundary(s, k, f);
            for (const auto &q : fe_fv.quadrature_point_indices()) {
              s.cache(0, [&](const uint l) -> const FEValuesBase<dim> & { return *s.fe_fv[l]; }, q);
              add_face_jacobian(s, out, 0, 0, boundary_jacobians[f * n_q_face + q], fe_fv.normal_vector(q),
                                weight * fe_fv.JxW(q), true);
            }
          }
          for (size_t ref = topology.face_ref_begin[k]; ref < topology.face_ref_begin[k + 1]; ++ref) {
            const auto [f, side] = topology.face_refs[ref];
            const uint slot = 1 + ref - topology.face_ref_begin[k];
            const auto &fe_iv0 = reinit_interface(s, f);
            for (uint l = 0; l < n_levels; ++l) {
              out.blocks[l][slot].reinit(out.dofs.size(), dof_handler_list[l]->get_fe().n_dofs_per_cell());
              out.columns[l][slot].resize(dof_handler_list[l]->get_fe().n_dofs_per_cell());
              DiFfRG::internal::on(*dof_handler_list[l], topology.faces[f].cell[1 - side])
                  ->get_dof_indices(out.columns[l][slot]);
            }
            const double sign = side == 0 ? 1. : -1.;
            for (const auto &q : fe_iv0.quadrature_point_indices()) {
              for (uint t = 0; t < 2; ++t)
                s.cache(
                    t, [&](const uint l) -> const FEValuesBase<dim> & { return s.fe_iv[l]->get_fe_face_values(t); }, q);
              const auto &Jq = face_jacobians[f * n_q_face + q];
              const NumberType w = sign * weight * fe_iv0.JxW(q);
              add_face_jacobian(s, out, side, side, Jq[side], fe_iv0.normal_vector(q), w, false, 0);
              add_face_jacobian(s, out, side, 1 - side, Jq[1 - side], fe_iv0.normal_vector(q), w, false, slot);
              add_face_extractor_jacobian(s, out, side, Jq[0], fe_iv0.normal_vector(q), w);
            }
          }
        });

        // The chain rule through the levels: d(main)/d(level k) * d(level k)/du.
        for (uint k = 1; k < n_levels; ++k) {
          jacobian_tmp[k] = 0;
          tbb::parallel_for(tbb::blocked_range<uint>(0, n_fe), [&](const tbb::blocked_range<uint> &r) {
            for (uint q = r.begin(); q < r.end(); ++q)
              for (const auto &c : model.get_components().ldg_couplings(k, 0))
                j_ug[k].block(q, c[0]).mmult(jacobian_tmp[k].block(q, c[1]), j_gu[k].block(c[0], c[1]),
                                             Vector<NumberType>(), false);
          });
        }
        tbb::parallel_for(tbb::blocked_range<uint>(0, n_fe * n_fe), [&](const tbb::blocked_range<uint> &r) {
          for (uint b = r.begin(); b < r.end(); ++b)
            for (uint k = 1; k < n_levels; ++k)
              jacobian.block(b / n_fe, b % n_fe).add(NumberType(1.), jacobian_tmp[k].block(b / n_fe, b % n_fe));
        });
        jacobian_times.scatter += phase.wall_time();
        ++jacobian_times.calls;
        timings_jacobian.push_back(timer.wall_time());
      }

      const PhaseTimes &residual_phase_times() const { return residual_times; }
      const PhaseTimes &jacobian_phase_times() const { return jacobian_times; }
      void reset_phase_times() { residual_times = jacobian_times = PhaseTimes{}; }

      SummaryEvent summary() const override
      {
        SummaryEvent result{.component = "LDG"};
        result.timing("reinit", average(timings_reinit) * 1000, timings_reinit.size())
            .timing("residual", average(timings_residual) * 1000, timings_residual.size())
            .timing("jac", average(timings_jacobian) * 1000, timings_jacobian.size());
        if (jacobian_times.calls > 0)
          result.timing("jac eval", jacobian_times.evaluate / jacobian_times.calls * 1000, jacobian_times.calls);
        return result;
      }

      double average_time_reinit() const { return average(timings_reinit); }
      uint num_reinits() const { return timings_reinit.size(); }
      double average_time_residual_assembly() const { return average(timings_residual); }
      uint num_residuals() const { return timings_residual.size(); }
      double average_time_jacobian_assembly() const { return average(timings_jacobian); }
      uint num_jacobians() const { return timings_jacobian.size(); }

    protected:
      using Base::discretization;
      using Base::dof_handler;
      using Base::fe;
      using Base::mapping;
      using Base::model;

      QGauss<dim> quadrature;
      QGauss<dim - 1> quadrature_face;
      using Base::schedule_for;

      std::vector<const DoFHandler<dim> *> dof_handler_list;

      mutable array<BlockVector<NumberType>, Components::count_fe_subsystems()> sol_vector;
      mutable array<BlockVector<NumberType>, Components::count_fe_subsystems()> sol_vector_tmp;
      mutable array<Vector<NumberType>, Components::count_fe_subsystems()> sol_vector_vec_tmp;
      mutable BlockVector<NumberType> sol_block; // FE solution in component blocks, for j_gu[k].vmult

      BlockSparsityPattern sparsity_pattern_jacobian;
      BlockSparsityPattern sparsity_pattern_mass;
      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_ug;
      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_gu;
      array<BlockSparsityPattern, Components::count_fe_subsystems()> sparsity_pattern_wg;

      array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> jacobian_tmp;

      BlockSparseMatrix<NumberType> mass_matrix;
      SparseMatrix<NumberType> component_mass_matrix_inverse;
      array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_ug;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_gu;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_wg;
      mutable array<BlockSparseMatrix<NumberType>, Components::count_fe_subsystems()> j_wg_tmp;

      std::vector<double> timings_reinit;
      std::vector<double> timings_residual;
      std::vector<double> timings_jacobian;

      /// Whether j_gu[k] has been built; it is rebuilt on every jacobian unless the model declares level k's
      /// jacobian constant.
      mutable array<bool, Components::count_fe_subsystems()> ldg_matrix_built{};

      using Base::EoM_config;
      using Base::extractor_dof_indices;

    private:
      static double average(const std::vector<double> &t)
      {
        return t.empty() ? 0. : std::accumulate(t.begin(), t.end(), 0.) / t.size();
      }

      /// About 256 MB of AD inputs and outputs per seed-stacked evaluation.
      static uint default_stacked_points()
      {
        constexpr size_t per_point = sizeof(autodiff::real) * (n_all + n_fe * (1 + dim) + dim);
        return std::max<size_t>(1, (size_t(256) << 20) / per_point);
      }

      template <typename Js> static void require_finite(const Js &J)
      {
        const bool finite = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, J.size()), true,
            [&](const tbb::blocked_range<size_t> &r, bool ok) {
              for (size_t i = r.begin(); ok && i != r.end(); ++i) {
                if constexpr (requires { J[i].is_finite(); })
                  ok = J[i].is_finite();
                else
                  ok = J[i][0].is_finite() && J[i][1].is_finite();
              }
              return ok;
            },
            std::logical_and<bool>());
        if (!finite) throw std::runtime_error("Infinity encountered in jacobian construction");
      }

      /// One cell's contribution to its own rows (of level 0, or of the level being built), before insertion.
      struct LocalData {
        std::vector<types::global_dof_index> dofs;
        /// Whether a dof of the cell is constrained, i.e. insertion has to go through the constraints.
        bool constrained = false;
        Vector<NumberType> residual;
        /// blocks[l][0]: columns of level l of the cell itself, blocks[l][1 + f]: of the neighbor across its
        /// f-th interior face; columns[l][...] the matching dof indices.
        array<std::vector<FullMatrix<NumberType>>, n_levels> blocks;
        array<std::vector<std::vector<types::global_dof_index>>, n_levels> columns;
        FullMatrix<NumberType> extractor_jacobian;
        FullMatrix<NumberType> extractor_dependence;
      };

      /// Per-thread FE data of every level, for the gathers and the scatters.
      struct Scratch {
        Scratch(const Mapping<dim> &mapping, const std::vector<const DoFHandler<dim> *> &dofhs,
                const dealii::Quadrature<dim> &quadrature, const dealii::Quadrature<dim - 1> &quadrature_face)
        {
          const UpdateFlags flags = update_values | update_gradients | update_quadrature_points | update_JxW_values;
          for (uint l = 0; l < n_levels; ++l) {
            const auto &fe = dofhs[l]->get_fe();
            fe_v[l] = std::make_unique<FEValues<dim>>(mapping, fe, quadrature, flags);
            fe_fv[l] = std::make_unique<FEFaceValues<dim>>(mapping, fe, quadrature_face, flags | update_normal_vectors);
            fe_iv[l] =
                std::make_unique<FEInterfaceValues<dim>>(mapping, fe, quadrature_face, flags | update_normal_vectors);
            values[l].resize(quadrature.size(), Vector<NumberType>(fe.n_components()));
            face_values[l].resize(quadrature_face.size(), Vector<NumberType>(fe.n_components()));
            comp[l].resize(fe.n_dofs_per_cell());
            for (uint i = 0; i < comp[l].size(); ++i)
              comp[l][i] = fe.system_to_component_index(i).first;
            for (uint t = 0; t < 2; ++t)
              shape_values[t][l].resize(fe.n_dofs_per_cell());
          }
          values_dot.resize(quadrature.size(), Vector<NumberType>(n_fe));
          test_gradients.resize(dofhs[0]->get_fe().n_dofs_per_cell());
        }

        /**
         * @brief The shape values of every level at point q into shape_values[t], of the FEValues fe_values(l);
         * for t = 0 also the test functions' gradients (level 0).
         */
        template <typename FEVOf> void cache(const uint t, const FEVOf &fe_values, const uint q)
        {
          for (uint l = 0; l < n_levels; ++l) {
            const auto &fev = fe_values(l);
            for (uint j = 0; j < comp[l].size(); ++j)
              shape_values[t][l][j] = fev.shape_value_component(j, q, comp[l][j]);
          }
          if (t == 0)
            for (uint i = 0; i < comp[0].size(); ++i)
              test_gradients[i] = fe_values(0).shape_grad_component(i, q, comp[0][i]);
        }

        array<std::unique_ptr<FEValues<dim>>, n_levels> fe_v;
        array<std::unique_ptr<FEFaceValues<dim>>, n_levels> fe_fv;
        array<std::unique_ptr<FEInterfaceValues<dim>>, n_levels> fe_iv;
        array<std::vector<uint>, n_levels> comp;
        array<std::vector<Vector<NumberType>>, n_levels> values, face_values;
        std::vector<Vector<NumberType>> values_dot;
        /// shape_values[t][l][j]: shape function j of level l at the current point, on trace t.
        array<array<std::vector<double>, n_levels>, 2> shape_values;
        std::vector<Tensor<1, dim>> test_gradients;
        LocalData local, level_local;
      };

      /// Number the owned cells of every level, their faces, and color the cells for the scatters.
      void setup_cells()
      {
        n_q = quadrature.size();
        n_q_face = quadrature_face.size();
        for (uint l = 0; l < n_levels; ++l)
          cells[l].reinit(*dof_handler_list[l], discretization.get_constraints(l));
        topology.reinit(cells[0]);
        scratch = std::make_unique<tbb::enumerable_thread_specific<Scratch>>(
            [this]() { return Scratch(mapping, dof_handler_list, quadrature, quadrature_face); });
      }

      const FEFaceValues<dim> &reinit_boundary(Scratch &s, const size_t k, const size_t f) const
      {
        for (uint l = 0; l < n_levels; ++l)
          s.fe_fv[l]->reinit(cells[l][k], topology.boundary_faces[f].second);
        return *s.fe_fv[0];
      }

      const FEInterfaceValues<dim> &reinit_interface(Scratch &s, const size_t f) const
      {
        for (uint l = 0; l < n_levels; ++l)
          topology.reinit(*s.fe_iv[l], f, *dof_handler_list[l]);
        return *s.fe_iv[0];
      }

      /// The level-l vector: the solution for l = 0, else the LDG level built from it.
      template <uint l> const auto &level_vector(const VectorType &solution) const
      {
        if constexpr (l == 0)
          return solution;
        else
          return sol_vector[l];
      }

      /**
       * @brief Fill batches at the points of the cells, boundary faces and interior faces: store(batch, first,
       * fe_values, fe_values_of_level, values_buffer_of_level, width) is called per cell / face (side) with the
       * FEValues of every level.
       */
      template <typename Batch, typename Store>
      void gather(Batch &cells_b, Batch &boundary_b, std::array<Batch, 2> &faces_b, const Store &store)
      {
        cells_b.reinit(cells[0].size() * n_q, false, false);
        boundary_b.reinit(topology.boundary_faces.size() * n_q_face, false, false);
        for (auto &b : faces_b)
          b.reinit(topology.faces.size() * n_q_face, false, false);
        boundary_normals.resize(boundary_b.size());
        face_normals.resize(faces_b[0].size());

        const auto geometry = [](Batch &batch, const size_t first, const auto &fev, const double width) {
          for (const auto &q : fev.quadrature_point_indices()) {
            for (uint d = 0; d < dim; ++d)
              batch.coordinate(d, first + q) = fev.quadrature_point(q)[d];
            batch.width(first + q) = width;
          }
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cells[0].size()), [&](const tbb::blocked_range<size_t> &r) {
          auto &s = scratch->local();
          for (size_t k = r.begin(); k != r.end(); ++k) {
            for (uint l = 0; l < n_levels; ++l)
              s.fe_v[l]->reinit(cells[l][k]);
            geometry(cells_b, k * n_q, *s.fe_v[0], DiFfRG::internal::cell_width(cells[0][k]));
            store(
                cells_b, k * n_q, [&](const uint l) -> const FEValuesBase<dim> & { return *s.fe_v[l]; },
                [&](const uint l) -> auto & { return s.values[l]; });
          }
        });
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, topology.boundary_faces.size()), [&](const tbb::blocked_range<size_t> &r) {
              auto &s = scratch->local();
              for (size_t f = r.begin(); f != r.end(); ++f) {
                const auto &cell = topology.boundary_faces[f].first;
                for (uint l = 0; l < n_levels; ++l)
                  s.fe_fv[l]->reinit(DiFfRG::internal::on(*dof_handler_list[l], cell),
                                     topology.boundary_faces[f].second);
                geometry(boundary_b, f * n_q_face, *s.fe_fv[0], DiFfRG::internal::cell_width(cell));
                store(
                    boundary_b, f * n_q_face, [&](const uint l) -> const FEValuesBase<dim> & { return *s.fe_fv[l]; },
                    [&](const uint l) -> auto & { return s.face_values[l]; });
                for (uint q = 0; q < n_q_face; ++q)
                  boundary_normals[f * n_q_face + q] = s.fe_fv[0]->normal_vector(q);
              }
            });
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, topology.faces.size()), [&](const tbb::blocked_range<size_t> &r) {
              auto &s = scratch->local();
              for (size_t f = r.begin(); f != r.end(); ++f) {
                const auto &fe_iv0 = reinit_interface(s, f);
                for (uint side = 0; side < 2; ++side) {
                  geometry(faces_b[side], f * n_q_face, fe_iv0.get_fe_face_values(side),
                           DiFfRG::internal::cell_width(topology.faces[f].cell[side]));
                  store(
                      faces_b[side], f * n_q_face,
                      [&](const uint l) -> const FEValuesBase<dim> & { return s.fe_iv[l]->get_fe_face_values(side); },
                      [&](const uint l) -> auto & { return s.face_values[l]; });
                }
                for (uint q = 0; q < n_q_face; ++q)
                  face_normals[f * n_q_face + q] = fe_iv0.normal_vector(q);
              }
            });
      }

      /// Phase 1 of the main level: the values of all levels at every point.
      void gather_main(const VectorType &solution_global, const Extractors &extracted_data, const VectorType &variables)
      {
        gather(cell_batch, boundary_batch, face_batch,
               [&](MainBatch &batch, const size_t first, const auto &fe_values, const auto &buffer) {
                 constexpr_for<0, n_levels, 1>([&](auto l) {
                   auto &vals = buffer(l);
                   fe_values(l).get_function_values(this->template level_vector<l>(solution_global), vals);
                   for (uint q = 0; q < vals.size(); ++q)
                     for (uint c = 0; c < Components::count_fe_functions(l); ++c)
                       batch.ldg_value(l, c, first + q) = vals[q][c];
                 });
               });
        for (auto *b : {&cell_batch, &boundary_batch, &face_batch[0], &face_batch[1]})
          b->set_shared(extracted_data, variables);
      }

      /// Phase 1 of level `from + 1`: the values of level `from` at every point.
      template <uint from, typename Source>
      void gather_level(LevelBatch<from> &cells_b, LevelBatch<from> &boundary_b,
                        std::array<LevelBatch<from>, 2> &faces_b, const Source &source)
      {
        gather(cells_b, boundary_b, faces_b,
               [&](LevelBatch<from> &batch, const size_t first, const auto &fe_values, const auto &buffer) {
                 auto &vals = buffer(from);
                 fe_values(from).get_function_values(source, vals);
                 for (uint q = 0; q < vals.size(); ++q)
                   for (uint c = 0; c < Components::count_fe_functions(from); ++c)
                     batch.value(c, first + q) = vals[q][c];
               });
        for (auto *b : {&cells_b, &boundary_b, &faces_b[0], &faces_b[1]})
          b->set_shared(no_extractors, no_variables);
      }

      /**
       * @brief Phase 3: assemble(k, scratch, local) every owned cell of level `level` (the level of the rows), after
       * reinit of its FEValues on every level and of its dof indices, and insert(local) the result; see
       * internal::ColoredCells::scatter.
       */
      template <typename Global, typename Assemble, typename Insert>
      void scatter(const uint level, LocalData Scratch::*local, Global &global, const Assemble &assemble,
                   const Insert &insert)
      {
        cells[level].scatter(
            global, *scratch, local, local_buffer,
            [&](const size_t k, Scratch &s, LocalData &data) {
              for (uint l = 0; l < n_levels; ++l)
                s.fe_v[l]->reinit(cells[l][k]);
              data.dofs.resize(dof_handler_list[level]->get_fe().n_dofs_per_cell());
              cells[level][k]->get_dof_indices(data.dofs);
              data.constrained = cells[level].is_constrained(k);
              assemble(k, s, data);
            },
            insert);
      }

      template <typename Assemble> void scatter_residual(VectorType &residual, const Assemble &assemble)
      {
        const auto &constraints = discretization.get_constraints(0);
        scatter(
            0, &Scratch::local, residual,
            [&](const size_t k, Scratch &s, LocalData &local) {
              local.residual.reinit(local.dofs.size());
              assemble(k, s, local.residual);
            },
            [&](LocalData &local) {
              if (local.constrained)
                constraints.distribute_local_to_global(local.residual, local.dofs, residual);
              else
                residual.add(local.dofs, local.residual);
            });
      }

      /**
       * @brief The rows of level 0 of the jacobian: columns of level 0 into @p jacobian, through the constraints;
       * with @p with_levels, columns of level l >= 1 into j_ug[l] and the extractor dependence into @p jacobian.
       */
      template <typename Assemble>
      void scatter_jacobian(BlockSparseMatrix<NumberType> &jacobian, const bool with_levels, const Assemble &assemble)
      {
        const auto &constraints = discretization.get_constraints(0);
        const bool any_constraints = constraints.n_constraints() > 0;
        scatter(
            0, &Scratch::local, jacobian,
            [&](const size_t k, Scratch &s, LocalData &local) {
              const size_t n_blocks = 1 + topology.n_interior_faces(k);
              for (uint l = 0; l < (with_levels ? n_levels : 1); ++l) {
                local.blocks[l].resize(n_blocks);
                local.columns[l].resize(n_blocks);
                local.blocks[l][0].reinit(local.dofs.size(), dof_handler_list[l]->get_fe().n_dofs_per_cell());
                local.columns[l][0].resize(dof_handler_list[l]->get_fe().n_dofs_per_cell());
                cells[l][k]->get_dof_indices(local.columns[l][0]);
                for (size_t b = 1; b < n_blocks; ++b)
                  local.blocks[l][b].reinit(0, 0);
              }
              if constexpr (n_extr > 0) local.extractor_jacobian.reinit(local.dofs.size(), n_extr);
              assemble(k, s, local);
            },
            [&](LocalData &local) {
              if (local.constrained)
                constraints.distribute_local_to_global(local.blocks[0][0], local.dofs, jacobian);
              else
                DiFfRG::internal::add_to_blocks(jacobian, local.dofs, local.dofs, local.blocks[0][0]);
              for (size_t b = 1; b < local.blocks[0].size(); ++b) {
                if (local.blocks[0][b].m() == 0) continue;
                if (any_constraints)
                  constraints.distribute_local_to_global(local.blocks[0][b], local.dofs, local.columns[0][b], jacobian);
                else
                  DiFfRG::internal::add_to_blocks(jacobian, local.dofs, local.columns[0][b], local.blocks[0][b]);
              }
              if (!with_levels) return;
              for (uint l = 1; l < n_levels; ++l)
                for (size_t b = 0; b < local.blocks[l].size(); ++b)
                  if (local.blocks[l][b].m() > 0)
                    DiFfRG::internal::add_to_blocks(j_ug[l], local.dofs, local.columns[l][b], local.blocks[l][b]);
              if constexpr (n_extr > 0) {
                local.extractor_dependence.reinit(local.dofs.size(), extractor_dof_indices.size());
                local.extractor_jacobian.mmult(local.extractor_dependence, this->extractor_jacobian);
                constraints.distribute_local_to_global(local.extractor_dependence, local.dofs, extractor_dof_indices,
                                                       jacobian);
              }
            });
      }

      /**
       * @brief r += w phi_i (F . n) over the boundary faces of cell k and +- w phi_i (NF . n) over its interior
       * faces, for the test functions phi of level @p level; F, NF from @p boundary and @p faces.
       */
      template <typename Out>
      void add_face_terms(Scratch &s, const size_t k, const uint level, Vector<NumberType> &r, const Out &boundary,
                          const Out &faces, const NumberType w) const
      {
        const auto &comp = s.comp[level];
        const auto normal_component = [](const Out &result, const size_t c, const size_t p, const Tensor<1, dim> &n) {
          NumberType value = 0.;
          for (uint d = 0; d < dim; ++d)
            value += result.flux(c, d)[p] * n[d];
          return value;
        };
        for (size_t f = topology.boundary_begin[k]; f < topology.boundary_begin[k + 1]; ++f) {
          reinit_boundary(s, k, f);
          const auto &fe_fv = *s.fe_fv[level];
          for (const auto &q : fe_fv.quadrature_point_indices())
            for (uint i = 0; i < r.size(); ++i)
              r(i) += w * fe_fv.JxW(q) * fe_fv.shape_value_component(i, q, comp[i]) *
                      normal_component(boundary, comp[i], f * n_q_face + q, fe_fv.normal_vector(q));
        }
        // [[phi_i]] * numflux * n: the trace of phi_i on its own side, with a minus sign on side 1.
        for (size_t ref = topology.face_ref_begin[k]; ref < topology.face_ref_begin[k + 1]; ++ref) {
          const auto [f, side] = topology.face_refs[ref];
          reinit_interface(s, f);
          const auto &fe_iv = *s.fe_iv[level];
          const auto &fe_fv = fe_iv.get_fe_face_values(side);
          const double sign = side == 0 ? 1. : -1.;
          for (const auto &q : fe_iv.quadrature_point_indices())
            for (uint i = 0; i < r.size(); ++i)
              r(i) += sign * w * fe_iv.JxW(q) * fe_fv.shape_value_component(i, q, comp[i]) *
                      normal_component(faces, comp[i], f * n_q_face + q, fe_iv.normal_vector(q));
        }
      }

      /// J += alpha d(mass)/d(u_dot) + beta d(mass)/du, contracted with the shape values.
      void add_mass_jacobian(Scratch &s, FullMatrix<NumberType> &J, const VectorType &u, const VectorType &u_dot,
                             const NumberType alpha, const NumberType beta) const
      {
        const auto &fe_v = *s.fe_v[0];
        const auto &comp = s.comp[0];
        fe_v.get_function_values(u, s.values[0]);
        fe_v.get_function_values(u_dot, s.values_dot);
        SimpleMatrix<NumberType, n_fe> j_mass, j_mass_dot;
        for (const auto &q : fe_v.quadrature_point_indices()) {
          model.template jacobian_mass<0>(j_mass, fe_v.quadrature_point(q), s.values[0][q], s.values_dot[q]);
          model.template jacobian_mass<1>(j_mass_dot, fe_v.quadrature_point(q), s.values[0][q], s.values_dot[q]);
          for (uint i = 0; i < J.m(); ++i)
            for (uint j = 0; j < J.n(); ++j)
              J(i, j) += fe_v.JxW(q) * fe_v.shape_value_component(i, q, comp[i]) *
                         fe_v.shape_value_component(j, q, comp[j]) *
                         (alpha * j_mass_dot(comp[i], comp[j]) + beta * j_mass(comp[i], comp[j]));
        }
      }

      /// The main-level jacobian blocks Jq at one cell point, contracted with the cached shapes, times w.
      template <typename PJ>
      void add_cell_jacobian(const Scratch &s, LocalData &out, const PJ &Jq, const NumberType w) const
      {
        const auto &comp0 = s.comp[0];
        const auto &sv = s.shape_values[0];
        for (uint l = 0; l < n_levels; ++l) {
          const auto &comp = s.comp[l];
          auto &M = out.blocks[l][0];
          const size_t offset = MainBatch::level_offset(l);
          for (uint i = 0; i < comp0.size(); ++i) {
            const auto ci = comp0[i];
            for (uint j = 0; j < comp.size(); ++j) {
              const auto cj = offset + comp[j];
              M(i, j) += w * sv[l][j] *
                         (-scalar_product(s.test_gradients[i], Jq.j_flux(ci, cj)) + sv[0][i] * Jq.j_source(ci, cj));
            }
          }
        }
        for (uint i = 0; i < comp0.size(); ++i)
          for (uint e = 0; e < n_extr; ++e)
            out.extractor_jacobian(i, e) += w * (-scalar_product(s.test_gradients[i], Jq.j_extr_flux(comp0[i], e)) +
                                                 sv[0][i] * Jq.j_extr_source(comp0[i], e));
      }

      /**
       * @brief w phi_i d(numflux.n)/d(trial_j) at one face point into column block @p block of every level, for test
       * functions of trace `row_side` and trial functions of trace `column_side`, whose jacobian blocks are Jq. With
       * @p with_extractors, also the extractor blocks of Jq (boundary faces; interior faces add their total extractor
       * dependence once, see add_face_extractor_jacobian).
       */
      template <typename PJ>
      void add_face_jacobian(const Scratch &s, LocalData &out, const uint row_side, const uint column_side,
                             const PJ &Jq, const Tensor<1, dim> &normal, const NumberType w, const bool with_extractors,
                             const size_t block = 0) const
      {
        const auto &comp0 = s.comp[0];
        const auto &rows = s.shape_values[row_side][0];
        for (uint l = 0; l < n_levels; ++l) {
          const auto &comp = s.comp[l];
          const auto &cols = s.shape_values[column_side][l];
          auto &M = out.blocks[l][block];
          const size_t offset = MainBatch::level_offset(l);
          for (uint i = 0; i < comp0.size(); ++i)
            for (uint j = 0; j < comp.size(); ++j)
              M(i, j) += w * rows[i] * cols[j] * scalar_product(Jq.j_flux(comp0[i], offset + comp[j]), normal);
        }
        if (with_extractors)
          for (uint i = 0; i < comp0.size(); ++i)
            for (uint e = 0; e < n_extr; ++e)
              out.extractor_jacobian(i, e) += w * rows[i] * scalar_product(Jq.j_extr_flux(comp0[i], e), normal);
      }

      /// The extractor dependence of an interior face's numerical flux, for the test functions of trace `side`.
      template <typename PJ>
      void add_face_extractor_jacobian(const Scratch &s, LocalData &out, const uint side, const PJ &Jq,
                                       const Tensor<1, dim> &normal, const NumberType w) const
      {
        for (uint i = 0; i < s.comp[0].size(); ++i)
          for (uint e = 0; e < n_extr; ++e)
            out.extractor_jacobian(i, e) +=
                w * s.shape_values[side][0][i] * scalar_product(Jq.j_extr_flux(s.comp[0][i], e), normal);
      }

      void rebuild_ldg_vectors(const VectorType &sol) const
      {
        constexpr_for<1, Components::count_fe_subsystems(), 1>([&](auto k) {
          if (!model.get_components().jacobians_constant(k, k - 1)) {
            if constexpr (k == 1)
              build_ldg_vector<k - 1, k>(sol, sol_vector[k], sol_vector_tmp[k]);
            else
              build_ldg_vector<k - 1, k>(sol_vector[k - 1], sol_vector[k], sol_vector_tmp[k]);
          } else {
            if (!ldg_matrix_built[k]) rebuild_ldg_jacobian<k>(sol);

            // sol is a plain Vector; BlockSparseMatrix::vmult(BlockVector, Vector) only uses column block 0,
            // i.e. FE component 0. Copy into the (component-wise) block structure so all couplings act.
            sol_block = sol;
            j_gu[k].vmult(sol_vector[k], sol_block);
          }
        });
      }

      /// j_gu[k] = d(level k)/du at the current levels: level k's own jacobian, chained through j_gu[k - 1].
      template <int k> void rebuild_ldg_jacobian(const VectorType &sol) const
      {
        static_assert(k > 0);
        ldg_matrix_built[k] = true;
        if constexpr (k == 1)
          build_ldg_jacobian<0, 1>(sol, j_gu[1], j_wg_tmp[1]);
        else {
          if (!ldg_matrix_built[k - 1]) rebuild_ldg_jacobian<k - 1>(sol);
          build_ldg_jacobian<k - 1, k>(sol_vector[k - 1], j_wg[k], j_wg_tmp[k]);
          // mmult adds into its target, which therefore has to start from zero.
          j_gu[k] = 0;
          for (const auto &c : model.get_components().ldg_couplings(k, 0))
            for (const auto &b : model.get_components().ldg_couplings(k, k - 1))
              if (b[0] == c[0])
                j_wg[k]
                    .block(b[0], b[1])
                    .mmult(j_gu[k].block(c[0], c[1]), j_gu[k - 1].block(b[1], c[1]), Vector<NumberType>(), false);
        }
      }

      /**
       * @brief Build the LDG vector at level 'to' from level 'from' = to - 1: assemble M l_to = -(grad phi, F) +
       * (phi, s) + the boundary and interior numerical fluxes, then solve with the mass matrix.
       */
      template <int from, int to, typename SourceVector, typename VectorTypeldg>
      void build_ldg_vector(const SourceVector &source, VectorTypeldg &ldg_vector, VectorTypeldg &ldg_vector_tmp) const
      {
        static_assert(to - from == 1, "can only build LDG from last level!");
        constexpr size_t n_to = Components::count_fe_functions(to);
        auto &self = const_cast<Assembler &>(*this);
        LevelBatch<from> cells_b, boundary_b;
        std::array<LevelBatch<from>, 2> faces_b;
        self.template gather_level<from>(cells_b, boundary_b, faces_b, source);

        BatchOutput<dim, NumberType, n_to> cell_out, boundary_out, face_out;
        cell_out.reinit(cells_b.size(), Term::flux | Term::source);
        evaluate_ldg_level<to>(model, cell_out, cells_b);
        boundary_out.reinit(boundary_b.size(), Term::flux);
        evaluate_ldg_boundary_numflux<to>(model, boundary_out, boundary_normals, boundary_b);
        face_out.reinit(faces_b[0].size(), Term::flux);
        evaluate_ldg_numflux<to>(model, face_out, face_normals, faces_b[0], faces_b[1]);

        const auto &constraints = discretization.get_constraints(to);
        ldg_vector_tmp = 0;
        self.scatter(
            to, &Scratch::level_local, ldg_vector_tmp,
            [&](const size_t k, Scratch &s, LocalData &local) {
              auto &r = local.residual;
              r.reinit(local.dofs.size());
              const auto &fe_v = *s.fe_v[to];
              const auto &comp = s.comp[to];
              for (const auto &q : fe_v.quadrature_point_indices()) {
                const size_t p = k * n_q + q;
                for (uint i = 0; i < r.size(); ++i) {
                  const auto c = comp[i];
                  Tensor<1, dim, NumberType> flux;
                  for (uint d = 0; d < dim; ++d)
                    flux[d] = cell_out.flux(c, d)[p];
                  r(i) += fe_v.JxW(q) * (-scalar_product(fe_v.shape_grad_component(i, q, c), flux) +
                                         fe_v.shape_value_component(i, q, c) * cell_out.source(c)[p]);
                }
              }
              add_face_terms(s, k, to, r, boundary_out, face_out, 1.);
            },
            [&](LocalData &local) {
              if (local.constrained)
                constraints.distribute_local_to_global(local.residual, local.dofs, ldg_vector_tmp);
              else
                ldg_vector_tmp.add(local.dofs, local.residual);
            });

        for (uint i = 0; i < n_to; ++i)
          component_mass_matrix_inverse.vmult(ldg_vector.block(i), ldg_vector_tmp.block(i));
      }

      /**
       * @brief Build the LDG jacobian d(level to)/d(level from), from = to - 1: assemble the derivative of the
       * right-hand side of build_ldg_vector, then multiply with the inverse mass matrix.
       */
      template <int from, int to, typename SourceVector>
      void build_ldg_jacobian(const SourceVector &source, BlockSparseMatrix<NumberType> &ldg_jacobian,
                              BlockSparseMatrix<NumberType> &ldg_jacobian_tmp) const
      {
        static_assert(to - from == 1, "can only build LDG from last level!");
        constexpr size_t n_to = Components::count_fe_functions(to), n_from = Components::count_fe_functions(from);
        auto &self = const_cast<Assembler &>(*this);
        LevelBatch<from> cells_b, boundary_b;
        std::array<LevelBatch<from>, 2> faces_b;
        self.template gather_level<from>(cells_b, boundary_b, faces_b, source);

        std::vector<PointJacobian<dim, n_to, n_from, 0>> J_cells, J_boundary;
        std::vector<FaceJacobian<dim, n_to, n_from, 0>> J_faces;
        SeedStackWorkspace<LevelBatch<from>, n_to> workspace;
        SeedStackWorkspace<LevelBatch<from>, n_to, 2> face_workspace;
        evaluate_ldg_level_jacobian<to>(model, J_cells, cells_b, max_stacked_points, workspace);
        evaluate_ldg_boundary_numflux_jacobian<to>(model, J_boundary, boundary_normals, boundary_b, max_stacked_points,
                                                   workspace);
        evaluate_ldg_numflux_jacobian<to>(model, J_faces, face_normals, faces_b[0], faces_b[1], max_stacked_points,
                                          face_workspace);
        require_finite(J_cells);
        require_finite(J_boundary);
        require_finite(J_faces);

        ldg_jacobian_tmp = 0;
        ldg_jacobian = 0;
        self.scatter(
            to, &Scratch::level_local, ldg_jacobian_tmp,
            [&](const size_t k, Scratch &s, LocalData &local) {
              const size_t n_blocks = 1 + topology.n_interior_faces(k);
              const uint n_cols = dof_handler_list[from]->get_fe().n_dofs_per_cell();
              auto &blocks = local.blocks[from];
              auto &columns = local.columns[from];
              blocks.resize(n_blocks);
              columns.resize(n_blocks);
              for (auto &b : blocks)
                b.reinit(local.dofs.size(), n_cols);
              columns[0].resize(n_cols);
              cells[from][k]->get_dof_indices(columns[0]);

              const auto &comp_to = s.comp[to];
              const auto &comp_from = s.comp[from];
              // M(i, j) += w test_i (column block of the trial functions) for the jacobian blocks J of one point.
              const auto add = [&](FullMatrix<NumberType> &M, const auto &test, const auto &trial, const auto &J,
                                   const NumberType w, const auto &grad_test, const Tensor<1, dim> *normal) {
                for (uint i = 0; i < comp_to.size(); ++i)
                  for (uint j = 0; j < comp_from.size(); ++j) {
                    const auto &jF = J.j_flux(comp_to[i], comp_from[j]);
                    const NumberType value =
                        normal ? test[i] * scalar_product(jF, *normal)
                               : -scalar_product(grad_test[i], jF) + test[i] * J.j_source(comp_to[i], comp_from[j]);
                    M(i, j) += w * trial[j] * value;
                  }
              };
              std::vector<double> test(comp_to.size()), trial(comp_from.size()), trial_n(comp_from.size());
              std::vector<Tensor<1, dim>> grad_test(comp_to.size());
              const auto cache = [&](std::vector<double> &v, const auto &fev, const auto &comp, const uint q) {
                for (uint i = 0; i < comp.size(); ++i)
                  v[i] = fev.shape_value_component(i, q, comp[i]);
              };

              const auto &fe_v_to = *s.fe_v[to];
              for (const auto &q : fe_v_to.quadrature_point_indices()) {
                cache(test, fe_v_to, comp_to, q);
                cache(trial, *s.fe_v[from], comp_from, q);
                for (uint i = 0; i < comp_to.size(); ++i)
                  grad_test[i] = fe_v_to.shape_grad_component(i, q, comp_to[i]);
                add(blocks[0], test, trial, J_cells[k * n_q + q], fe_v_to.JxW(q), grad_test, nullptr);
              }
              for (size_t f = topology.boundary_begin[k]; f < topology.boundary_begin[k + 1]; ++f) {
                reinit_boundary(s, k, f);
                const auto &fe_fv = *s.fe_fv[to];
                for (const auto &q : fe_fv.quadrature_point_indices()) {
                  cache(test, fe_fv, comp_to, q);
                  cache(trial, *s.fe_fv[from], comp_from, q);
                  const Tensor<1, dim> normal = fe_fv.normal_vector(q);
                  add(blocks[0], test, trial, J_boundary[f * n_q_face + q], fe_fv.JxW(q), grad_test, &normal);
                }
              }
              for (size_t ref = topology.face_ref_begin[k]; ref < topology.face_ref_begin[k + 1]; ++ref) {
                const auto [f, side] = topology.face_refs[ref];
                const size_t slot = 1 + ref - topology.face_ref_begin[k];
                columns[slot].resize(n_cols);
                DiFfRG::internal::on(*dof_handler_list[from], topology.faces[f].cell[1 - side])
                    ->get_dof_indices(columns[slot]);
                reinit_interface(s, f);
                const auto &fe_iv = *s.fe_iv[to];
                const double sign = side == 0 ? 1. : -1.;
                for (const auto &q : fe_iv.quadrature_point_indices()) {
                  cache(test, fe_iv.get_fe_face_values(side), comp_to, q);
                  cache(trial, s.fe_iv[from]->get_fe_face_values(side), comp_from, q);
                  cache(trial_n, s.fe_iv[from]->get_fe_face_values(1 - side), comp_from, q);
                  const auto &Jq = J_faces[f * n_q_face + q];
                  const Tensor<1, dim> normal = fe_iv.normal_vector(q);
                  add(blocks[0], test, trial, Jq[side], sign * fe_iv.JxW(q), grad_test, &normal);
                  add(blocks[slot], test, trial_n, Jq[1 - side], sign * fe_iv.JxW(q), grad_test, &normal);
                }
              }
            },
            [&](LocalData &local) {
              for (size_t b = 0; b < local.blocks[from].size(); ++b)
                DiFfRG::internal::add_to_blocks(ldg_jacobian_tmp, local.dofs, local.columns[from][b],
                                                local.blocks[from][b]);
            });

        for (const auto &c : model.get_components().ldg_couplings(to, from))
          component_mass_matrix_inverse.mmult(ldg_jacobian.block(c[0], c[1]), ldg_jacobian_tmp.block(c[0], c[1]),
                                              Vector<NumberType>(), false);
      }

      /**
       * @brief Create a sparsity pattern for matrices between the DoFs of two DoFHandlers, with given stencil size.
       *
       * @param sparsity_pattern The pattern to store the result in.
       * @param dofh_to DoFHandler giving the row DoFs.
       * @param dofh_from DoFHandler giving the column DoFs.
       * @param stencil Stencil size of the resulting pattern.
       */
      void build_ldg_sparsity(BlockSparsityPattern &sparsity_pattern, const DoFHandler<dim> &to_dofh,
                              const DoFHandler<dim> &from_dofh, const int stencil = 1,
                              bool add_extractor_dofs = false) const
      {
        const auto &triangulation = discretization.get_triangulation();
        auto to_dofs_per_component = DoFTools::count_dofs_per_fe_component(to_dofh);
        auto from_dofs_per_component = DoFTools::count_dofs_per_fe_component(from_dofh);
        auto to_n_fe = to_dofs_per_component.size();
        auto from_n_fe = from_dofs_per_component.size();
        for (uint j = 1; j < from_dofs_per_component.size(); ++j)
          if (from_dofs_per_component[j] != from_dofs_per_component[0])
            throw std::runtime_error("For LDG the FE basis of all systems must be equal!");
        for (uint j = 1; j < to_dofs_per_component.size(); ++j)
          if (to_dofs_per_component[j] != to_dofs_per_component[0])
            throw std::runtime_error("For LDG the FE basis of all systems must be equal!");

        BlockDynamicSparsityPattern dsp(to_n_fe, from_n_fe);
        for (uint i = 0; i < to_n_fe; ++i)
          for (uint j = 0; j < from_n_fe; ++j)
            dsp.block(i, j).reinit(to_dofs_per_component[i], from_dofs_per_component[j]);
        dsp.collect_sizes();

        const auto to_dofs_per_cell = to_dofh.get_fe().dofs_per_cell;
        const auto from_dofs_per_cell = from_dofh.get_fe().dofs_per_cell;

        for (const auto &t_cell : triangulation.active_cell_iterators()) {
          std::vector<types::global_dof_index> to_dofs(to_dofs_per_cell);
          std::vector<types::global_dof_index> from_dofs(from_dofs_per_cell);
          const auto to_cell = typename DoFHandler<dim>::active_cell_iterator(
              &to_dofh.get_triangulation(), t_cell->level(), t_cell->index(), &to_dofh);
          const auto from_cell = typename DoFHandler<dim>::active_cell_iterator(
              &from_dofh.get_triangulation(), t_cell->level(), t_cell->index(), &from_dofh);
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

        if (add_extractor_dofs)
          for (uint row = 0; row < dsp.n_rows(); ++row)
            for (const auto &col : extractor_dof_indices)
              dsp.add(row, col);

        sparsity_pattern.copy_from(dsp);
      }

      /**
       * @brief Build the inverse of matrix in and save the result to out.
       *
       * @param in The matrix to invert. Must have a valid sparsity pattern.
       * @param out The matrix to store the result. Must have a valid sparsity pattern.
       */
      void build_inverse(const SparseMatrix<NumberType> &in, SparseMatrix<NumberType> &out) const
      {
        GrowingVectorMemory<Vector<NumberType>> mem;
        SparseDirectUMFPACK inverse;
        inverse.initialize(in);
        // out is m x n
        // we go row-wise, i.e. we keep one n fixed and insert a row
        tbb::parallel_for(tbb::blocked_range<int>(0, out.n()), [&](tbb::blocked_range<int> r) {
          typename VectorMemory<Vector<NumberType>>::Pointer tmp(mem);
          tmp->reinit(out.m());
          for (int n = r.begin(); n < r.end(); ++n) {
            *tmp = 0;
            (*tmp)[n] = 1.;
            inverse.solve(*tmp);
            for (auto it = out.begin(n); it != out.end(n); ++it)
              it->value() = (*tmp)[it->column()];
          }
        });
      }

      const uint max_stacked_points;
      uint n_q = 0, n_q_face = 0;
      /// The owned cells, numbered alike on every level; colored by the constraints of their level.
      array<DiFfRG::internal::ColoredCells<dim>, n_levels> cells;
      DiFfRG::internal::FaceTopology<dim> topology;
      std::unique_ptr<tbb::enumerable_thread_specific<Scratch>> scratch;
      std::vector<LocalData> local_buffer;
      const std::array<NumberType, 0> no_extractors{};
      const VectorType no_variables;

      MainBatch cell_batch, boundary_batch;
      std::array<MainBatch, 2> face_batch;
      std::vector<Tensor<1, dim>> boundary_normals, face_normals;
      BatchOutput<dim, NumberType, n_fe> cell_result, boundary_result, face_result;
      std::vector<PointJacobian<dim, n_fe, n_all, n_extr>> cell_jacobians, boundary_jacobians;
      std::vector<FaceJacobian<dim, n_fe, n_all, n_extr>> face_jacobians;
      SeedStackWorkspace<MainBatch, n_fe> cell_workspace, boundary_workspace;
      SeedStackWorkspace<MainBatch, n_fe, 2> face_workspace;

      PhaseTimes residual_times, jacobian_times;

    protected:
      constexpr static int nothing = 0;
      using Base::EoM;
      using Base::EoM_cell;
      using Base::EoM_minimum_guess;
      using Base::extractor_jacobian_u;
      using Base::old_EoM_cell;
      using Base::old_extractor_cell;

      /// The LDG solution across all subsystems, its gradients and hessians, and the raw potential,
      /// all at one point. Built by evaluate_at() and consumed by readouts/extract/jacobian_extractors.
      template <typename PotentialEvaluation = RawPotentialEvaluation<dim, NumberType>> struct PointEvaluation {
        std::vector<Vector<NumberType>> solutions;
        std::vector<std::vector<Tensor<1, dim, NumberType>>> gradients{
            std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
        std::vector<std::vector<Tensor<2, dim, NumberType>>> hessians{
            std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
        PotentialEvaluation potential;
        /// Subsystem 0's FEValues at the point, kept for the shape values the extractor jacobian needs.
        std::shared_ptr<FEValues<dim>> fe_values;
      };

      /**
       * @brief Evaluate every LDG subsystem at @p x, which lies in @p x_cell.
       *
       * Each subsystem carries its own DoFHandler over the same triangulation, so the cell has to be
       * re-addressed per subsystem before its FEValues can be reinit'ed. Only subsystem 0 (the
       * solution proper) has gradients and hessians read off it; the LDG levels are values only.
       */
      template <typename RawPotential>
      auto evaluate_at(const Point<dim> &x, const typename DoFHandler<dim>::cell_iterator &x_cell,
                       const VectorType &solution_global, const RawPotential &raw_potential) const
      {
        using t_Iterator = typename Triangulation<dim>::active_cell_iterator;
        const auto x_unit = mapping.transform_real_to_unit_cell(x_cell, x);

        std::vector<std::shared_ptr<FEValues<dim>>> fe_v;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
          fe_v.emplace_back(std::make_shared<FEValues<dim>>(
              mapping, discretization.get_fe(k), x_unit,
              update_values | update_gradients | update_quadrature_points | update_JxW_values | update_hessians));
          auto cell = dof_handler_list[k]->begin_active();
          cell->copy_from(*t_Iterator(x_cell));
          fe_v[k]->reinit(cell);
        }

        PointEvaluation<decltype(evaluate_raw_potential(raw_potential, mapping, x))> evaluation;
        for (uint k = 0; k < Components::count_fe_subsystems(); ++k) {
          std::vector<Vector<NumberType>> values{Vector<NumberType>(Components::count_fe_functions(k))};
          if (k == 0)
            fe_v[0]->get_function_values(solution_global, values);
          else
            fe_v[k]->get_function_values(sol_vector[k], values);
          evaluation.solutions.push_back(values[0]);
        }
        fe_v[0]->get_function_gradients(solution_global, evaluation.gradients);
        fe_v[0]->get_function_hessians(solution_global, evaluation.hessians);
        evaluation.potential = evaluate_raw_potential(raw_potential, mapping, x);
        evaluation.fe_values = fe_v[0];
        return evaluation;
      }

      void readouts(OutputFrame<dim, VectorType> &data_out, const VectorType &solution_global,
                    const VectorType &variables) const
      {
        auto raw_potential = reconstruct_raw_potential(
            solution_global, dof_handler, mapping,
            [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, EoM_config,
            &this->potential_cache);
        auto helper = [&](auto &&...args) {
          if constexpr (sizeof...(args) == 3) {
            auto &&[id, EoMfun, outputter] = std::forward_as_tuple(std::forward<decltype(args)>(args)...);
            data_out.register_readout(id);
            auto EoM_cell = this->EoM_cell;
            auto EoM_result = get_EoM_point_with_potential(
                EoM_cell, solution_global, this->dof_handler, this->mapping, EoMfun,
                [&](const auto &p, const auto &) { return p; }, this->EoM_config, this->EoM_minimum_guess,
                &this->potential_cache);
            if (EoM_result.potential) this->EoM_minimum_guess = EoM_result.potential->minimum;
            const auto EoM = EoM_result.point;

            // GCC 16 does not find dependent-base members from inside this variadic lambda; keep `this->`.
            const auto readout_solution = this->evaluate_at(EoM, EoM_cell, solution_global, raw_potential);
            const auto &potential = readout_solution.potential;

            // The readout is always at this readout's EoM. The extractors may not be: a model that
            // defines extractor_point reads them elsewhere, and dt_variables must see the same
            // values here as it does during assembly.
            std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
            if constexpr (Components::count_extractors() > 0) {
              const auto [x, cell] = this->resolve_extractor_point(EoM, EoM_cell, solution_global);
              const auto evaluation = this->evaluate_at(x, cell, solution_global, raw_potential);
              auto extractor_tuple =
                  std::tuple_cat(vector_to_tuple<Components::count_fe_subsystems()>(evaluation.solutions),
                                 std::tie(evaluation.gradients[0], evaluation.hessians[0], this->nothing, variables,
                                          evaluation.potential.value, evaluation.potential.gradient,
                                          evaluation.potential.mass_hessian));
              this->model.extract(__extracted_data, x, fe_more_conv(extractor_tuple));
            }
            const auto &extracted_data = __extracted_data;

            auto solution_tuple =
                std::tuple_cat(vector_to_tuple<Components::count_fe_subsystems()>(readout_solution.solutions),
                               std::tie(readout_solution.gradients[0], readout_solution.hessians[0], extracted_data,
                                        variables, potential.value, potential.gradient, potential.mass_hessian));

            outputter(data_out, EoM, fe_more_conv(solution_tuple));
            data_out.attach_eom_potential(std::move(EoM_result));
          } else {
            DiFfRG::internal::validate_readout_helper_arity<decltype(args)...>();
          }
        };
        model.readouts_multiple(helper, data_out);
        data_out.attach_raw_potential(std::move(raw_potential));
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
              EoM_config, EoM_minimum_guess, &this->potential_cache);
          EoM = EoM_result.point;
          if (EoM_result.potential) EoM_minimum_guess = EoM_result.potential->minimum;
        }
        if (set_EoM) {
          this->EoM = EoM;
          this->EoM_cell = EoM_cell;
        }
        rebuild_ldg_vectors(solution_global);

        const auto raw_potential = this->extractor_raw_potential(solution_global);

        const auto [x, cell] = this->resolve_extractor_point(EoM, EoM_cell, solution_global);
        const auto evaluation = evaluate_at(x, cell, solution_global, raw_potential);

        auto solution_tuple = std::tuple_cat(
            vector_to_tuple<Components::count_fe_subsystems()>(evaluation.solutions),
            std::tie(evaluation.gradients[0], evaluation.hessians[0], this->nothing, variables,
                     evaluation.potential.value, evaluation.potential.gradient, evaluation.potential.mass_hessian));

        model.extract(data, x, fe_more_conv(solution_tuple));
      }

      bool jacobian_extractors(FullMatrix<NumberType> &extractor_jacobian, const VectorType &solution_global,
                               const VectorType &variables)
      {
        if (extractor_jacobian_u.m() != Components::count_extractors() ||
            extractor_jacobian_u.n() != Components::count_fe_functions())
          extractor_jacobian_u =
              FullMatrix<NumberType>(Components::count_extractors(), Components::count_fe_functions());

        auto EoM_result = get_EoM_point_with_potential(
            EoM_cell, solution_global, dof_handler, mapping,
            [&](const auto &p, const auto &values) { return model.EoM(p, values); },
            [&](const auto &p, const auto &values) { return model.EoM_postprocess(p, values); }, EoM_config,
            EoM_minimum_guess, &this->potential_cache);
        EoM = EoM_result.point;
        if (EoM_result.potential) EoM_minimum_guess = EoM_result.potential->minimum;
        const auto raw_potential = this->extractor_raw_potential(solution_global);

        // The extractor jacobian couples to the dofs of the cell the extractors are actually
        // evaluated in, which is the extractor point's cell, not the EoM's.
        const auto [x, cell] = this->resolve_extractor_point(EoM, EoM_cell, solution_global);
        bool new_cell = (old_extractor_cell != cell);
        old_EoM_cell = EoM_cell;
        old_extractor_cell = cell;

        const auto evaluation = evaluate_at(x, cell, solution_global, raw_potential);
        const auto &fe_v = *evaluation.fe_values;
        const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();
        if (new_cell) {
          extractor_dof_indices.resize(n_dofs);
          cell->get_dof_indices(extractor_dof_indices);
          rebuild_jacobian_sparsity();
        }

        auto solution_tuple = std::tuple_cat(
            vector_to_tuple<Components::count_fe_subsystems()>(evaluation.solutions),
            std::tie(evaluation.gradients[0], evaluation.hessians[0], this->nothing, variables,
                     evaluation.potential.value, evaluation.potential.gradient, evaluation.potential.mass_hessian));

        extractor_jacobian_u = 0;
        model.template jacobian_extractors<0>(extractor_jacobian_u, x, fe_more_conv(solution_tuple));

        if (extractor_jacobian.m() != Components::count_extractors() || extractor_jacobian.n() != n_dofs)
          extractor_jacobian = FullMatrix<NumberType>(Components::count_extractors(), n_dofs);

        for (uint e = 0; e < Components::count_extractors(); ++e)
          for (uint i = 0; i < n_dofs; ++i) {
            const auto component_i = fe_v.get_fe().system_to_component_index(i).first;
            extractor_jacobian(e, i) =
                extractor_jacobian_u(e, component_i) * fe_v.shape_value_component(i, 0, component_i);
          }

        return new_cell;
      }

      using Base::timings_variable_jacobian;
      using Base::timings_variable_residual;
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

      virtual void residual_variables(VectorType &residual, const VectorType &variables,
                                      const VectorType &spatial_solution) override
      {
        Timer timer;
        std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
        if constexpr (Components::count_extractors() > 0)
          extract(__extracted_data, spatial_solution, variables, true, false, false);
        const auto &extracted_data = __extracted_data;
        model.dt_variables(residual, v_tie(variables, extracted_data));
        Kokkos::fence();
        timings_variable_residual.push_back(timer.wall_time());
      }

      virtual void jacobian_variables(FullMatrix<NumberType> &jacobian, const VectorType &variables,
                                      const VectorType &spatial_solution) override
      {
        Timer timer;
        std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
        if constexpr (Components::count_extractors() > 0)
          extract(__extracted_data, spatial_solution, variables, true, false, false);
        const auto &extracted_data = __extracted_data;
        model.template jacobian_variables<0>(jacobian, v_tie(variables, extracted_data));
        Kokkos::fence();
        timings_variable_jacobian.push_back(timer.wall_time());
      }
    };
  } // namespace LDG
} // namespace DiFfRG
