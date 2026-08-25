#pragma once

// external libraries
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <spdlog/spdlog.h>
#include <tbb/tbb.h>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>
#include <DiFfRG/discretization/common/assembly_schedule.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/common/solution_sample.hh>
#include <DiFfRG/discretization/data/output_session.hh>

namespace DiFfRG
{
  using namespace dealii;
  using std::array;

  /**
   * @brief The basic assembler that can be used for any standard CG scheme with flux and source.
   *
   * @tparam Model The model class which contains the physical equations.
   */
  template <typename Discretization_, typename Model_>
  class FEMAssembler : public AbstractAssembler<typename Discretization_::VectorType,
                                                typename Discretization_::SparseMatrixType, Discretization_::dim>
  {
  protected:
    constexpr static int nothing = 0;

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
    using NumberType = typename Discretization::NumberType;
    using VectorType = typename Discretization::VectorType;
    using SparseMatrixType = typename Discretization::SparseMatrixType;

    using Components = typename Discretization::Components;
    static constexpr uint dim = Discretization::dim;

    [[deprecated("Pass output.log_port() or an intentional LogPort{}")]] FEMAssembler(
        Discretization &discretization, Model &model,
        DiFfRG::internal::LegacyDefaultLogPortArgument<Discretization, ConfigTree> config)
        : FEMAssembler(discretization, model, config.value(),
                       DiFfRG::internal::legacy_default_log_port<Discretization>())
    {
    }

    FEMAssembler(Discretization &discretization, Model &model, const ConfigTree &config, LogPort log_port)
        : discretization(discretization), model(model), log_port(std::move(log_port)), fe(discretization.get_fe()),
          dof_handler(discretization.get_dof_handler()), mapping(discretization.get_mapping()),
          schedule_overrides(AssemblyScheduleOverrides::from_config(config)),
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
      // Restricted to owned rows under the distributed policy: deal.II writes every index of this
      // set into a distributed vector and compresses with VectorOperation::insert, so an
      // unrestricted set has every rank inserting into every entry.
      return restrict_to_owned<VectorType>(DoFTools::extract_dofs(dof_handler, component_mask),
                                           discretization.get_locally_owned_dofs());
    }

    virtual void attach_data_output(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                                    const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                    const VectorType &residual = VectorType()) override
    {
      const auto fe_function_names = Components::FEFunction_Descriptor::get_names_vector();
      std::vector<std::string> fe_function_names_residual;
      fe_function_names_residual.reserve(fe_function_names.size());
      for (const auto &name : fe_function_names)
        fe_function_names_residual.push_back(name + "_residual");
      std::vector<std::string> fe_function_names_dot;
      fe_function_names_dot.reserve(fe_function_names.size());
      for (const auto &name : fe_function_names)
        fe_function_names_dot.push_back(name + "_dot");

      auto fe_out = data_out.fields();
      fe_out.attach(dof_handler, solution, fe_function_names);
      if (dt_solution.size() > 0) fe_out.attach(dof_handler, dt_solution, fe_function_names_dot);
      if (residual.size() > 0) fe_out.attach(dof_handler, residual, fe_function_names_residual);

      readouts(data_out, solution, variables);
    }

    const auto &get_discretization() const { return discretization; }
    auto &get_discretization() { return discretization; }

    virtual void reinit() override
    {
      const auto metadata = internal::build_affine_constraint_metadata<Components, dim>(discretization);
      const AffineConstraintContext<Components, dim> context(metadata);

      auto &constraints = discretization.get_constraints();
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      internal::apply_model_affine_constraints(model, constraints, context);
      constraints.close();

      update_assembly_schedules();
    }

    virtual void rebuild_jacobian_sparsity() = 0;

    virtual void set_time(double t) override { model.set_time(t); }

    virtual void refinement_indicator(Vector<double> & /*indicator*/, const VectorType & /*solution*/) = 0;

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
    };
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
    };

    /// The FE solution and the reconstructed raw potential at one point.
    struct PointEvaluation {
      std::vector<Vector<NumberType>> values{Vector<NumberType>(Components::count_fe_functions())};
      std::vector<std::vector<Tensor<1, dim, NumberType>>> gradients{
          std::vector<Tensor<1, dim, NumberType>>(Components::count_fe_functions())};
      std::vector<std::vector<Tensor<2, dim, NumberType>>> hessians{
          std::vector<Tensor<2, dim, NumberType>>(Components::count_fe_functions())};
      RawPotentialEvaluation<dim, NumberType> potential;
      /// Kept for the shape values the extractor jacobian needs.
      std::shared_ptr<FEValues<dim>> fe_values;
    };

    /// @brief Evaluate the FE solution and the raw potential at @p x, which lies in @p cell.
    template <typename RawPotential>
    PointEvaluation evaluate_at(const Point<dim> &x, const typename DoFHandler<dim>::cell_iterator &cell,
                                const VectorType &solution_global, const RawPotential &raw_potential) const
    {
      PointEvaluation evaluation;
      evaluation.fe_values = std::make_shared<FEValues<dim>>(
          mapping, fe, mapping.transform_real_to_unit_cell(cell, x),
          update_values | update_gradients | update_quadrature_points | update_JxW_values | update_hessians);
      evaluation.fe_values->reinit(cell);
      evaluation.fe_values->get_function_values(solution_global, evaluation.values);
      evaluation.fe_values->get_function_gradients(solution_global, evaluation.gradients);
      evaluation.fe_values->get_function_hessians(solution_global, evaluation.hessians);
      evaluation.potential = evaluate_raw_potential(raw_potential, mapping, x);
      return evaluation;
    }

    void readouts(OutputFrame<dim, VectorType> &data_out, const VectorType &solution_global,
                  const VectorType &variables) const
    {
      auto raw_potential = reconstruct_raw_potential(
          solution_global, dof_handler, mapping,
          [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, EoM_config);
      auto helper = [&](auto &&...args) {
        if constexpr (sizeof...(args) == 3) {
          auto &&[id, EoMfun, outputter] = std::forward_as_tuple(std::forward<decltype(args)>(args)...);
          data_out.register_readout(id);
          auto EoM_cell = this->EoM_cell;
          auto EoM_result = get_EoM_point_with_potential(
              EoM_cell, solution_global, dof_handler, mapping, EoMfun, [&](const auto &p, const auto &) { return p; },
              EoM_config, EoM_minimum_guess);
          if (EoM_result.potential) EoM_minimum_guess = EoM_result.potential->minimum;
          const auto EoM = EoM_result.point;

          // The readout is always at this readout's EoM. The extractors may not be: a model that
          // defines extractor_point reads them elsewhere, and dt_variables must see the same values
          // here as it does during assembly.
          const auto readout_solution = evaluate_at(EoM, EoM_cell, solution_global, raw_potential);
          const auto &potential = readout_solution.potential;

          std::array<NumberType, Components::count_extractors()> __extracted_data{{}};
          if constexpr (Components::count_extractors() > 0) {
            const auto [x, cell] = resolve_extractor_point(EoM, EoM_cell, solution_global);
            const auto extractor_solution = evaluate_at(x, cell, solution_global, raw_potential);
            model.extract(__extracted_data, x,
                          e_tie(extractor_solution.values[0], extractor_solution.gradients[0],
                                extractor_solution.hessians[0], nothing, variables,
                                extractor_solution.potential.value, extractor_solution.potential.gradient,
                                extractor_solution.potential.hessian));
          }
          const auto &extracted_data = __extracted_data;

          outputter(data_out, EoM,
                    e_tie(readout_solution.values[0], readout_solution.gradients[0], readout_solution.hessians[0],
                          extracted_data, variables, potential.value, potential.gradient, potential.hessian));
          data_out.attach_eom_potential(std::move(EoM_result));
        } else {
          internal::validate_readout_helper_arity<decltype(args)...>();
        }
      };
      model.readouts_multiple(helper, data_out);
      data_out.attach_raw_potential(std::move(raw_potential));
    }

    /**
     * @brief Where the model wants its extractors evaluated, and the cell holding that point.
     *
     * The EoM itself when the model does not define `extractor_point` -- and then this costs
     * nothing, because building the SolutionSample is inside the `if constexpr`.
     */
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
            EoM_config, EoM_minimum_guess);
        EoM = EoM_result.point;
        if (EoM_result.potential) EoM_minimum_guess = EoM_result.potential->minimum;
      }
      if (set_EoM) {
        this->EoM = EoM;
        this->EoM_cell = EoM_cell;
      }

      const auto raw_potential = reconstruct_raw_potential(
          solution_global, dof_handler, mapping,
          [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, EoM_config);

      const auto [x, cell] = resolve_extractor_point(EoM, EoM_cell, solution_global);
      const auto e = evaluate_at(x, cell, solution_global, raw_potential);
      model.extract(data, x,
                    e_tie(e.values[0], e.gradients[0], e.hessians[0], nothing, variables, e.potential.value,
                          e.potential.gradient, e.potential.hessian));
    }

    bool jacobian_extractors(FullMatrix<NumberType> &extractor_jacobian, const VectorType &solution_global,
                             const VectorType &variables)
    {
      if (extractor_jacobian_u.m() != Components::count_extractors() ||
          extractor_jacobian_u.n() != Components::count_fe_functions())
        extractor_jacobian_u = FullMatrix<NumberType>(Components::count_extractors(), Components::count_fe_functions());
      if (extractor_jacobian_du.m() != Components::count_extractors() ||
          extractor_jacobian_du.n() != Components::count_fe_functions() * dim)
        extractor_jacobian_du =
            FullMatrix<NumberType>(Components::count_extractors(), Components::count_fe_functions() * dim);
      if (extractor_jacobian_ddu.m() != Components::count_extractors() ||
          extractor_jacobian_ddu.n() != Components::count_fe_functions() * dim * dim)
        extractor_jacobian_ddu =
            FullMatrix<NumberType>(Components::count_extractors(), Components::count_fe_functions() * dim * dim);

      auto EoM_result = get_EoM_point_with_potential(
          EoM_cell, solution_global, dof_handler, mapping,
          [&](const auto &p, const auto &values) { return model.EoM(p, values); },
          [&](const auto &p, const auto &values) { return model.EoM_postprocess(p, values); }, EoM_config,
          EoM_minimum_guess);
      EoM = EoM_result.point;
      if (EoM_result.potential) EoM_minimum_guess = EoM_result.potential->minimum;

      // The extractor jacobian couples to the dofs of the cell the extractors are actually
      // evaluated in, which is the extractor point's cell, not the EoM's.
      const auto [x, cell] = resolve_extractor_point(EoM, EoM_cell, solution_global);
      // Agreed, not rank-local: new_cell gates rebuild_jacobian_sparsity(), which is now
      // collective (SparsityTools::distribute_sparsity_pattern). A rank that skipped the rebuild
      // while another performed it would sit out a collective the others are inside, and the run
      // would hang rather than fail. any_of, not all_of: if ANY rank needs the rebuild, every rank
      // must enter it. On a replicated mesh the ranks agree anyway -- this makes that a guarantee
      // rather than a coincidence, and costs one bool reduction per EoM update.
      bool new_cell = MPI::any_of(discretization.get_communicator(), old_extractor_cell != cell);
      old_EoM_cell = EoM_cell;
      old_extractor_cell = cell;

      const auto raw_potential = reconstruct_raw_potential(
          solution_global, dof_handler, mapping,
          [&](const auto &p, const auto &values) { return model.raw_potential_gradient(p, values); }, EoM_config);
      const auto e = evaluate_at(x, cell, solution_global, raw_potential);
      const auto &fe_v = *e.fe_values;
      const auto &potential = e.potential;

      const uint n_dofs = fe_v.get_fe().n_dofs_per_cell();
      if (new_cell) {
        extractor_dof_indices.resize(n_dofs);
        cell->get_dof_indices(extractor_dof_indices);
        rebuild_jacobian_sparsity();
      }

      extractor_jacobian_u = 0;
      extractor_jacobian_du = 0;
      extractor_jacobian_ddu = 0;
      model.template jacobian_extractors<0>(extractor_jacobian_u, x,
                                            e_tie(e.values[0], e.gradients[0], e.hessians[0], nothing, variables,
                                                  potential.value, potential.gradient, potential.hessian));
      model.template jacobian_extractors<1>(extractor_jacobian_du, x,
                                            e_tie(e.values[0], e.gradients[0], e.hessians[0], nothing, variables,
                                                  potential.value, potential.gradient, potential.hessian));
      model.template jacobian_extractors<2>(extractor_jacobian_ddu, x,
                                            e_tie(e.values[0], e.gradients[0], e.hessians[0], nothing, variables,
                                                  potential.value, potential.gradient, potential.hessian));

      if (extractor_jacobian.m() != Components::count_extractors() || extractor_jacobian.n() != n_dofs)
        extractor_jacobian = FullMatrix<NumberType>(Components::count_extractors(), n_dofs);

      for (uint e = 0; e < Components::count_extractors(); ++e)
        for (uint i = 0; i < n_dofs; ++i) {
          const auto component_i = fe_v.get_fe().system_to_component_index(i).first;
          extractor_jacobian(e, i) =
              extractor_jacobian_u(e, component_i) * fe_v.shape_value_component(i, 0, component_i);
          for (uint d1 = 0; d1 < dim; ++d1) {
            extractor_jacobian(e, i) +=
                extractor_jacobian_du(e, component_i * dim + d1) * fe_v.shape_grad_component(i, 0, component_i)[d1];
            for (uint d2 = 0; d2 < dim; ++d2)
              extractor_jacobian(e, i) += extractor_jacobian_ddu(e, component_i * dim * dim + d1 * dim + d2) *
                                          fe_v.shape_hessian_component(i, 0, component_i)[d1][d2];
          }
        }

      return new_cell;
    }

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
    LogPort log_port;
    const FiniteElement<dim> &fe;
    const DoFHandler<dim> &dof_handler;
    const Mapping<dim> &mapping;

    /**
     * @brief The mesh_loop schedule for a loop whose cell worker costs @p cost_ns nanoseconds.
     *
     * Pure arithmetic on the cached cell count, so it is called per loop rather than stored per
     * cost: the loops do not fall into two classes, and anything between the reference points in
     * namespace assembly_cost gets its own schedule instead of being rounded to one of them.
     */
    AssemblySchedule schedule_for(const double cost_ns) const
    {
      return make_assembly_schedule(n_owned_cells, DiFfRG::n_threads(), cost_ns,
                                    schedule_overrides);
    }

    /**
     * @brief Re-count the cells the schedules are sized from.
     *
     * Called from reinit(), because h-adaptivity moves the cell count. Logs only on a change, so
     * an adaptive run does not narrate every refinement.
     */
    void update_assembly_schedules()
    {
      const uint n_owned = n_locally_owned_cells(discretization);
      const bool unchanged = n_owned == n_owned_cells;
      n_owned_cells = n_owned;
      if (unchanged) return;

      const uint threads = DiFfRG::n_threads();
      const auto cheap = schedule_for(assembly_cost::local_fe);
      const auto integral = schedule_for(assembly_cost::momentum_integral);
      log_port.info("FEM: Assembling {} cells on {} threads -- {}x{} workers/cells for a cheap cell loop, "
                    "{}x{} for an integral one.",
                    n_owned_cells, threads, cheap.queue_length, cheap.chunk_size, integral.queue_length,
                    integral.chunk_size);
    }

    /// Cells this rank assembles. Refreshed in reinit(); the per-loop schedules are sized from it.
    uint n_owned_cells = 0;
    const AssemblyScheduleOverrides schedule_overrides;

    mutable typename DoFHandler<dim>::cell_iterator EoM_cell;
    typename DoFHandler<dim>::cell_iterator old_EoM_cell;
    const Config::EoMConfig EoM_config;
    mutable Point<dim> EoM;
    mutable std::optional<Point<dim>> EoM_minimum_guess;
    /// Where the extractors are evaluated. Equal to EoM unless the model defines extractor_point.
    typename DoFHandler<dim>::cell_iterator old_extractor_cell;
    FullMatrix<NumberType> extractor_jacobian;
    FullMatrix<NumberType> extractor_jacobian_u;
    FullMatrix<NumberType> extractor_jacobian_du;
    FullMatrix<NumberType> extractor_jacobian_ddu;
    std::vector<types::global_dof_index> extractor_dof_indices;

    std::vector<double> timings_variable_residual;
    std::vector<double> timings_variable_jacobian;
  };
} // namespace DiFfRG
