#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/discretization/FEM/assembler/common.hh>
#include <DiFfRG/discretization/common/cell_geometry.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/model/batch.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

// external libraries
#include <tbb/enumerable_thread_specific.h>

// standard library
#include <bit>
#include <memory>
#include <numeric>

namespace DiFfRG
{
  namespace CG
  {
    using namespace dealii;
    using std::array;

    template <typename... T> auto i_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives", "fe_hessians">>(
          std::tie(t...));
    }

    /**
     * @brief The CG assembler for models with flux and source.
     *
     * Residual and jacobian are assembled in three phases:
     *  1. gather: the FE solution at every quadrature point of the locally owned cells and boundary
     *     faces, into a PointBatch;
     *  2. evaluate: the model's flux_source_batch (and boundary numflux) over the whole batch, or their
     *     jacobians, see model/batch.hh. The default flux_source_batch calls the per-point flux and
     *     source in a flat parallel loop; a model can override it to evaluate its momentum integrals
     *     with one map_points() call per batch, on the CPU or the GPU;
     *  3. scatter: the contraction with the shape functions, in a parallel loop over cell colors. Cells
     *     of one color share no dof (nor a constraint master), so they add into deal.II's serial matrix
     *     and vector concurrently; other types (PETSc) are filled serially from per-cell results
     *     computed in parallel.
     *
     * Each MPI rank batches its own cells and phase 2 communicates nothing. The results do not depend on
     * the thread count: contributions to a global entry are summed in color order.
     *
     * Config: /discretization/batched/max_stacked_points bounds the size of one AD evaluation of the
     * jacobian (default: 256 MB of AD inputs), see internal::seed_stacked_jacobian.
     */
    template <typename Discretization_,
              typename Model_ = typename DiFfRG::internal::assembler_model_of<Discretization_>::type>
    class Assembler : public FEMAssembler<Discretization_, Model_>
    {
      using Base = FEMAssembler<Discretization_, Model_>;

    public:
      using Discretization = Discretization_;
      using Model = Model_;
      using NumberType = typename Discretization::NumberType;
      using VectorType = typename Discretization::VectorType;
      using SparseMatrixType = typename Discretization::SparseMatrixType;
      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;

      static constexpr size_t n_fe = Components::count_fe_functions();
      static constexpr size_t n_extr = Components::count_extractors();
      using Extractors = std::array<NumberType, n_extr>;
      using Batch = PointBatch<dim, NumberType, n_fe, Extractors, VectorType>;

      /// Wall time of the three assembly phases, summed over all calls.
      struct PhaseTimes {
        double gather = 0., evaluate = 0., scatter = 0.;
        uint calls = 0;
      };

      Assembler(Discretization &discretization, Model &model, const ConfigTree &config)
          : Base(discretization, model, config),
            quadrature(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
            quadrature_face(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
            max_stacked_points(config.get_uint("/discretization/batched/max_stacked_points", default_stacked_points()))
      {
        static_assert(Components::count_fe_subsystems() == 1, "A CG model cannot have multiple submodels!");
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

      virtual void reinit() override
      {
        Timer timer;
        Base::reinit();

        DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                        /*keep_constrained_dofs = */ true);
        finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern_mass, discretization.get_locally_owned_dofs(),
                                               discretization.get_locally_relevant_dofs(),
                                               discretization.get_communicator());
        reinit_la_matrix(mass_matrix, sparsity_pattern_mass, discretization.get_locally_owned_dofs(),
                         discretization.get_communicator());
        MatrixCreator::create_mass_matrix(dof_handler, quadrature, mass_matrix, (Function<dim, NumberType> *)nullptr,
                                          discretization.get_constraints());
        finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern_jacobian, discretization.get_locally_owned_dofs(),
                                               discretization.get_locally_relevant_dofs(),
                                               discretization.get_communicator());
        timings_reinit.push_back(timer.wall_time());

        const auto metadata = DiFfRG::internal::build_affine_constraint_metadata<Components, dim>(discretization);
        const AffineConstraintContext<Components, dim> context(metadata);
        auto &constraints = discretization.get_constraints();
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        DiFfRG::internal::apply_model_affine_constraints(model, constraints, context);
        constraints.close();

        setup_cells();
      }

      virtual void rebuild_jacobian_sparsity() override
      {
        DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                        /*keep_constrained_dofs = */ true);
        for (const auto &row : discretization.get_locally_relevant_dofs())
          for (const auto &col : extractor_dof_indices)
            dsp.add(row, col);
        finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern_jacobian, discretization.get_locally_owned_dofs(),
                                               discretization.get_locally_relevant_dofs(),
                                               discretization.get_communicator());
      }

      virtual const get_type::SparsityPattern<SparseMatrixType> &get_sparsity_pattern_jacobian() const override
      {
        return sparsity_pattern_jacobian;
      }
      virtual const SparseMatrixType &get_mass_matrix() const override { return mass_matrix; }

      /// The model's cell_indicator integrated over each cell; CG has no face contribution.
      virtual void refinement_indicator(Vector<double> &indicator, const VectorType &solution_global) override
      {
        tbb::enumerable_thread_specific<CellScratch> scratch([this]() {
          return CellScratch(mapping, fe, quadrature, quadrature_face, gather_flags() | update_hessians);
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const tbb::blocked_range<size_t> &r) {
          auto &s = scratch.local();
          for (size_t k = r.begin(); k != r.end(); ++k) {
            auto &fe_v = s.fe_values;
            fe_v.reinit(cells[k]);
            fe_v.get_function_values(solution_global, s.values);
            fe_v.get_function_gradients(solution_global, s.gradients);
            fe_v.get_function_hessians(solution_global, s.hessians);
            double value = 0., local = 0.;
            for (const auto &q : fe_v.quadrature_point_indices()) {
              model.cell_indicator(local, fe_v.quadrature_point(q), i_tie(s.values[q], s.gradients[q], s.hessians[q]));
              value += fe_v.JxW(q) * local;
            }
            indicator[cells[k]->active_cell_index()] += value;
          }
        });
      }

      virtual void mass(VectorType &mass, const VectorType &solution_global, const VectorType &solution_global_dot,
                        NumberType weight) override
      {
        scatter_residual(mass, [&](const size_t, CellScratch &s, Vector<NumberType> &r) {
          const auto &fe_v = s.fe_values;
          fe_v.get_function_values(solution_global, s.values);
          fe_v.get_function_values(solution_global_dot, s.values_dot);
          array<NumberType, n_fe> m{};
          for (const auto &q : fe_v.quadrature_point_indices()) {
            model.mass(m, fe_v.quadrature_point(q), s.values[q], s.values_dot[q]);
            for (uint i = 0; i < r.size(); ++i)
              r(i) += weight * fe_v.JxW(q) * fe_v.shape_value_component(i, q, s.comp[i]) * m[s.comp[i]];
          }
        });
      }

      virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                            const VectorType &solution_global_dot, NumberType weight_mass,
                            const VectorType &variables = VectorType()) override
      {
        Timer timer, phase;
        Extractors extracted_data{{}};
        if constexpr (n_extr > 0) this->extract(extracted_data, solution_global, variables, true, false, true);
        gather(solution_global, extracted_data, variables);
        residual_times.gather += phase.wall_time();

        phase.restart();
        cell_result.reinit(cell_batch.size());
        model.flux_source_batch(cell_result, cell_batch);
        face_result.reinit(face_batch.size());
        evaluate_boundary_numflux(model, face_result, face_normals, face_batch);
        residual_times.evaluate += phase.wall_time();

        phase.restart();
        scatter_residual(residual, [&](const size_t k, CellScratch &s, Vector<NumberType> &r) {
          const auto &fe_v = s.fe_values;
          fe_v.get_function_values(solution_global, s.values);
          fe_v.get_function_values(solution_global_dot, s.values_dot);
          array<Tensor<1, dim, NumberType>, n_fe> flux;
          array<NumberType, n_fe> source, mass{};
          for (const auto &q : fe_v.quadrature_point_indices()) {
            const size_t p = k * n_q + q;
            for (size_t c = 0; c < n_fe; ++c) {
              for (uint d = 0; d < dim; ++d)
                flux[c][d] = cell_result.flux(c, d)[p];
              source[c] = cell_result.source(c)[p];
            }
            model.mass(mass, fe_v.quadrature_point(q), s.values[q], s.values_dot[q]);
            for (uint i = 0; i < r.size(); ++i) {
              const auto c = s.comp[i];
              r(i) += fe_v.JxW(q) * (weight * (-scalar_product(fe_v.shape_grad_component(i, q, c), flux[c]) +
                                               fe_v.shape_value_component(i, q, c) * source[c]) +
                                     weight_mass * fe_v.shape_value_component(i, q, c) * mass[c]);
            }
          }
          for (size_t f = face_begin[k]; f < face_begin[k + 1]; ++f) {
            auto &fe_fv = s.fe_face_values;
            fe_fv.reinit(cells[k], boundary_faces[f].second);
            for (const auto &q : fe_fv.quadrature_point_indices())
              for (uint i = 0; i < r.size(); ++i) {
                const auto c = s.comp[i];
                Tensor<1, dim, NumberType> numflux;
                for (uint d = 0; d < dim; ++d)
                  numflux[d] = face_result.flux(c, d)[f * n_q_face + q];
                r(i) += weight * fe_fv.JxW(q) * fe_fv.shape_value_component(i, q, c) *
                        scalar_product(numflux, fe_fv.normal_vector(q));
              }
          }
        });
        residual_times.scatter += phase.wall_time();
        ++residual_times.calls;
        timings_residual.push_back(timer.wall_time());
      }

      virtual void jacobian_mass(SparseMatrixType &jacobian, const VectorType &solution_global,
                                 const VectorType &solution_global_dot, NumberType alpha, NumberType beta) override
      {
        Timer timer;
        scatter_jacobian(jacobian, [&](const size_t, CellScratch &s, LocalData &out) {
          add_mass_jacobian(s, out.jacobian, solution_global, solution_global_dot, alpha, beta);
        });
        timings_jacobian.push_back(timer.wall_time());
      }

      virtual void jacobian(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType weight,
                            const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                            const VectorType &variables = VectorType()) override
      {
        Timer timer, phase;
        Extractors extracted_data{{}};
        if constexpr (n_extr > 0) {
          this->extract(extracted_data, solution_global, variables, true, true, true);
          if (this->jacobian_extractors(this->extractor_jacobian, solution_global, variables))
            reinit_la_matrix(jacobian, sparsity_pattern_jacobian, discretization.get_locally_owned_dofs(),
                             discretization.get_communicator());
        }
        gather(solution_global, extracted_data, variables);
        jacobian_times.gather += phase.wall_time();

        phase.restart();
        evaluate_flux_source_jacobian(model, cell_jacobians, cell_batch, max_stacked_points, cell_workspace);
        evaluate_boundary_numflux_jacobian(model, face_jacobians, face_normals, face_batch, max_stacked_points,
                                           face_workspace);
        jacobian_times.evaluate += phase.wall_time();

        phase.restart();
        scatter_jacobian(jacobian, [&](const size_t k, CellScratch &s, LocalData &out) {
          add_mass_jacobian(s, out.jacobian, solution_global, solution_global_dot, alpha, beta);
          const auto &fe_v = s.fe_values;
          for (const auto &q : fe_v.quadrature_point_indices()) {
            s.cache_shapes(fe_v, q);
            add_flux_source_jacobian(s, out, cell_jacobians[k * n_q + q], weight * fe_v.JxW(q));
          }
          for (size_t f = face_begin[k]; f < face_begin[k + 1]; ++f) {
            auto &fe_fv = s.fe_face_values;
            fe_fv.reinit(cells[k], boundary_faces[f].second);
            for (const auto &q : fe_fv.quadrature_point_indices()) {
              s.cache_shapes(fe_fv, q);
              add_boundary_jacobian(s, out, face_jacobians[f * n_q_face + q], fe_fv.normal_vector(q),
                                    weight * fe_fv.JxW(q));
            }
          }
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
        SummaryEvent result{.component = "CG"};
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
      using Base::extractor_dof_indices;
      using Base::fe;
      using Base::mapping;
      using Base::model;

      QGauss<dim> quadrature;
      QGauss<dim - 1> quadrature_face;

      get_type::SparsityPattern<SparseMatrixType> sparsity_pattern_mass;
      get_type::SparsityPattern<SparseMatrixType> sparsity_pattern_jacobian;
      SparseMatrixType mass_matrix;

      std::vector<double> timings_reinit;
      std::vector<double> timings_residual;
      std::vector<double> timings_jacobian;

    private:
      static constexpr bool reads_hessians = batch_reads_hessians<Model>();

      static double average(const std::vector<double> &t)
      {
        return t.empty() ? 0. : std::accumulate(t.begin(), t.end(), 0.) / t.size();
      }

      /// About 256 MB of AD inputs and outputs per seed-stacked evaluation.
      static uint default_stacked_points()
      {
        constexpr size_t per_point = sizeof(autodiff::real) * n_fe * (2 + 2 * dim + (reads_hessians ? dim * dim : 0));
        return std::max<size_t>(1, (size_t(256) << 20) / per_point);
      }

      static UpdateFlags gather_flags()
      {
        return update_values | update_gradients | update_quadrature_points | update_JxW_values |
               (reads_hessians ? update_hessians : update_default);
      }

      /// One cell's contribution to the global system, before insertion.
      struct LocalData {
        std::vector<types::global_dof_index> dofs;
        /// Whether a dof of the cell is constrained, i.e. insertion has to go through the constraints.
        bool constrained = false;
        Vector<NumberType> residual;
        FullMatrix<NumberType> jacobian;
        FullMatrix<NumberType> extractor_jacobian;
        FullMatrix<NumberType> extractor_dependence;
      };

      /// Per-thread FE data, for the gather and the scatter.
      struct CellScratch {
        CellScratch(const Mapping<dim> &mapping, const FiniteElement<dim> &fe,
                    const dealii::Quadrature<dim> &quadrature, const dealii::Quadrature<dim - 1> &quadrature_face,
                    const UpdateFlags flags)
            : fe_values(mapping, fe, quadrature, flags),
              fe_face_values(mapping, fe, quadrature_face, flags | update_normal_vectors),
              values(std::max(quadrature.size(), quadrature_face.size()), Vector<NumberType>(fe.n_components())),
              values_dot(values), gradients(values.size(), std::vector<Tensor<1, dim, NumberType>>(fe.n_components())),
              hessians(values.size(), std::vector<Tensor<2, dim, NumberType>>(fe.n_components())),
              comp(fe.n_dofs_per_cell()), sv(fe.n_dofs_per_cell()), sg(fe.n_dofs_per_cell()), sh(fe.n_dofs_per_cell())
        {
          for (uint i = 0; i < comp.size(); ++i)
            comp[i] = fe.system_to_component_index(i).first;
        }

        /// Shape values, gradients and (if read) hessians of every dof at point q.
        template <typename FEV> void cache_shapes(const FEV &fe_v, const uint q)
        {
          for (uint i = 0; i < comp.size(); ++i) {
            sv[i] = fe_v.shape_value_component(i, q, comp[i]);
            sg[i] = fe_v.shape_grad_component(i, q, comp[i]);
            if constexpr (reads_hessians) sh[i] = fe_v.shape_hessian_component(i, q, comp[i]);
          }
        }

        FEValues<dim> fe_values;
        FEFaceValues<dim> fe_face_values;
        std::vector<Vector<NumberType>> values, values_dot;
        std::vector<std::vector<Tensor<1, dim, NumberType>>> gradients;
        std::vector<std::vector<Tensor<2, dim, NumberType>>> hessians;
        std::vector<uint> comp;
        std::vector<double> sv;
        std::vector<Tensor<1, dim>> sg;
        std::vector<Tensor<2, dim>> sh;
        LocalData local;
      };

      /// Number the owned cells and their boundary faces, and color the cells for the scatter.
      void setup_cells()
      {
        n_q = quadrature.size();
        n_q_face = quadrature_face.size();
        cells.clear();
        boundary_faces.clear();
        face_begin.assign(1, 0);
        // The cells and faces mesh_loop visits with assemble_own_cells | assemble_boundary_faces.
        for (const auto &cell : locally_owned_cells(dof_handler)) {
          cells.push_back(cell);
          for (const uint f : cell->face_indices())
            if (cell->at_boundary(f) && !cell->has_periodic_neighbor(f)) boundary_faces.emplace_back(cell, f);
          face_begin.push_back(boundary_faces.size());
        }

        // Greedy coloring, one bit per color and row. A cell writes the rows of its dofs and, through the
        // constraints, those of their masters; no two cells of a color may share one.
        const auto &constraints = discretization.get_constraints();
        std::vector<std::uint64_t> row_colors(dof_handler.n_dofs(), 0);
        std::vector<types::global_dof_index> rows;
        colors.clear();
        constrained_cell.assign(cells.size(), false);
        for (size_t k = 0; k < cells.size(); ++k) {
          rows.resize(fe.n_dofs_per_cell());
          cells[k]->get_dof_indices(rows);
          for (uint i = 0; i < fe.n_dofs_per_cell(); ++i)
            if (constraints.is_constrained(rows[i])) {
              constrained_cell[k] = true;
              if (const auto *entries = constraints.get_constraint_entries(rows[i]))
                for (const auto &entry : *entries)
                  rows.push_back(entry.first);
            }
          std::uint64_t taken = 0;
          for (const auto row : rows)
            taken |= row_colors[row];
          if (~taken == 0) throw std::runtime_error("CG::Assembler: more than 64 cell colors needed.");
          const uint color = std::countr_one(taken);
          for (const auto row : rows)
            row_colors[row] |= std::uint64_t(1) << color;
          if (color >= colors.size()) colors.resize(color + 1);
          colors[color].push_back(k);
        }

        scratch = std::make_unique<tbb::enumerable_thread_specific<CellScratch>>(
            [this]() { return CellScratch(mapping, fe, quadrature, quadrature_face, gather_flags()); });
      }

      /// Phase 1: the solution at every quadrature point of the owned cells and boundary faces.
      void gather(const VectorType &solution_global, const Extractors &extracted_data, const VectorType &variables)
      {
        cell_batch.reinit(cells.size() * n_q, reads_hessians);
        cell_batch.set_shared(extracted_data, variables);
        face_batch.reinit(boundary_faces.size() * n_q_face, reads_hessians);
        face_batch.set_shared(extracted_data, variables);
        face_normals.resize(face_batch.size());

        const auto store = [&](Batch &batch, const size_t first, const auto &fe_v, CellScratch &s, const double width) {
          fe_v.get_function_values(solution_global, s.values);
          fe_v.get_function_gradients(solution_global, s.gradients);
          if constexpr (reads_hessians) fe_v.get_function_hessians(solution_global, s.hessians);
          for (const auto &q : fe_v.quadrature_point_indices()) {
            const size_t i = first + q;
            for (size_t c = 0; c < n_fe; ++c) {
              batch.value(c, i) = s.values[q][c];
              for (uint d1 = 0; d1 < dim; ++d1) {
                batch.derivative(c, d1, i) = s.gradients[q][c][d1];
                if constexpr (reads_hessians)
                  for (uint d2 = 0; d2 < dim; ++d2)
                    batch.hessian(c, d1, d2, i) = s.hessians[q][c][d1][d2];
              }
            }
            for (uint d = 0; d < dim; ++d)
              batch.coordinate(d, i) = fe_v.quadrature_point(q)[d];
            batch.width(i) = width;
          }
        };

        tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const tbb::blocked_range<size_t> &r) {
          auto &s = scratch->local();
          for (size_t k = r.begin(); k != r.end(); ++k) {
            s.fe_values.reinit(cells[k]);
            store(cell_batch, k * n_q, s.fe_values, s, DiFfRG::internal::cell_width(cells[k]));
          }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, boundary_faces.size()),
                          [&](const tbb::blocked_range<size_t> &r) {
                            auto &s = scratch->local();
                            for (size_t f = r.begin(); f != r.end(); ++f) {
                              const auto &[cell, face_no] = boundary_faces[f];
                              s.fe_face_values.reinit(cell, face_no);
                              store(face_batch, f * n_q_face, s.fe_face_values, s, DiFfRG::internal::cell_width(cell));
                              for (uint q = 0; q < n_q_face; ++q)
                                face_normals[f * n_q_face + q] = s.fe_face_values.normal_vector(q);
                            }
                          });
      }

      /**
       * @brief Phase 3: assemble(k, scratch, local) every owned cell, after reinit of the cell's FEValues and
       * dof indices, and insert(local) the result.
       *
       * deal.II's serial vector and matrix are written concurrently, one parallel loop per color. Other types
       * (PETSc) are assembled in parallel into per-cell buffers and inserted serially in the same order.
       */
      template <typename Global, typename Assemble, typename Insert>
      void scatter(Global &global, const Assemble &assemble, const Insert &insert)
      {
        constexpr bool concurrent = std::is_same_v<Global, dealii::Vector<NumberType>> ||
                                    std::is_same_v<Global, dealii::SparseMatrix<NumberType>>;
        const auto run = [&](const size_t k, CellScratch &s, LocalData &local) {
          s.fe_values.reinit(cells[k]);
          local.dofs.resize(fe.n_dofs_per_cell());
          cells[k]->get_dof_indices(local.dofs);
          local.constrained = constrained_cell[k];
          assemble(k, s, local);
        };
        // map() is collective and each rank visits only its own cells; see NoMapsHere.
        const NoMapsHere no_maps_during_assembly;
        if constexpr (concurrent) {
          for (const auto &color : colors)
            tbb::parallel_for(tbb::blocked_range<size_t>(0, color.size()), [&](const tbb::blocked_range<size_t> &r) {
              auto &s = scratch->local();
              for (size_t i = r.begin(); i != r.end(); ++i) {
                run(color[i], s, s.local);
                insert(s.local);
              }
            });
        } else {
          local_buffer.resize(cells.size());
          tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const tbb::blocked_range<size_t> &r) {
            auto &s = scratch->local();
            for (size_t k = r.begin(); k != r.end(); ++k)
              run(k, s, local_buffer[k]);
          });
          for (const auto &color : colors)
            for (const size_t k : color)
              insert(local_buffer[k]);
        }
        global.compress(dealii::VectorOperation::add);
      }

      template <typename Assemble> void scatter_residual(VectorType &residual, const Assemble &assemble)
      {
        const auto &constraints = discretization.get_constraints();
        scatter(
            residual,
            [&](const size_t k, CellScratch &s, LocalData &local) {
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

      template <typename Assemble> void scatter_jacobian(SparseMatrixType &jacobian, const Assemble &assemble)
      {
        const auto &constraints = discretization.get_constraints();
        scatter(
            jacobian,
            [&](const size_t k, CellScratch &s, LocalData &local) {
              local.jacobian.reinit(local.dofs.size(), local.dofs.size());
              if constexpr (n_extr > 0) local.extractor_jacobian.reinit(local.dofs.size(), n_extr);
              assemble(k, s, local);
            },
            [&](LocalData &local) {
              if (local.constrained)
                constraints.distribute_local_to_global(local.jacobian, local.dofs, jacobian);
              else
                jacobian.add(local.dofs, local.jacobian, false);
              if constexpr (n_extr > 0) {
                local.extractor_dependence.reinit(local.dofs.size(), extractor_dof_indices.size());
                local.extractor_jacobian.mmult(local.extractor_dependence, this->extractor_jacobian);
                constraints.distribute_local_to_global(local.extractor_dependence, local.dofs, extractor_dof_indices,
                                                       jacobian);
              }
            });
      }

      /// J += alpha d(mass)/d(u_dot) + beta d(mass)/du, contracted with the shape values.
      void add_mass_jacobian(CellScratch &s, FullMatrix<NumberType> &J, const VectorType &u, const VectorType &u_dot,
                             const NumberType alpha, const NumberType beta) const
      {
        const auto &fe_v = s.fe_values;
        fe_v.get_function_values(u, s.values);
        fe_v.get_function_values(u_dot, s.values_dot);
        SimpleMatrix<NumberType, n_fe> j_mass, j_mass_dot;
        for (const auto &q : fe_v.quadrature_point_indices()) {
          model.template jacobian_mass<0>(j_mass, fe_v.quadrature_point(q), s.values[q], s.values_dot[q]);
          model.template jacobian_mass<1>(j_mass_dot, fe_v.quadrature_point(q), s.values[q], s.values_dot[q]);
          for (uint i = 0; i < J.m(); ++i)
            for (uint j = 0; j < J.n(); ++j)
              J(i, j) += fe_v.JxW(q) * fe_v.shape_value_component(i, q, s.comp[i]) *
                         fe_v.shape_value_component(j, q, s.comp[j]) *
                         (alpha * j_mass_dot(s.comp[i], s.comp[j]) + beta * j_mass(s.comp[i], s.comp[j]));
        }
      }

      /// The flux/source jacobian blocks Jq at one cell point, contracted with the cached shapes, times w.
      template <typename PJ>
      void add_flux_source_jacobian(const CellScratch &s, LocalData &out, const PJ &Jq, const NumberType w) const
      {
        const auto &[comp, sv, sg, sh] = std::tie(s.comp, s.sv, s.sg, s.sh);
        for (uint i = 0; i < comp.size(); ++i) {
          const auto ci = comp[i];
          for (uint j = 0; j < comp.size(); ++j) {
            const auto cj = comp[j];
            NumberType c = sv[j] * (-scalar_product(sg[i], Jq.j_flux(ci, cj)) + sv[i] * Jq.j_source(ci, cj)) +
                           scalar_product(sg[j], -scalar_product(sg[i], Jq.j_grad_flux(ci, cj)) +
                                                     sv[i] * Jq.j_grad_source(ci, cj));
            if constexpr (reads_hessians)
              c += scalar_product(sh[j],
                                  -scalar_product(sg[i], Jq.j_hess_flux(ci, cj)) + sv[i] * Jq.j_hess_source(ci, cj));
            out.jacobian(i, j) += w * c;
          }
          for (uint e = 0; e < n_extr; ++e)
            out.extractor_jacobian(i, e) +=
                w * (-scalar_product(sg[i], Jq.j_extr_flux(ci, e)) + sv[i] * Jq.j_extr_source(ci, e));
        }
      }

      /// The boundary numflux jacobian blocks Jq at one face point, contracted with the normal and the shapes.
      template <typename PJ>
      void add_boundary_jacobian(const CellScratch &s, LocalData &out, const PJ &Jq, const Tensor<1, dim> &normal,
                                 const NumberType w) const
      {
        const auto &[comp, sv, sg, sh] = std::tie(s.comp, s.sv, s.sg, s.sh);
        for (uint i = 0; i < comp.size(); ++i) {
          const auto ci = comp[i];
          for (uint j = 0; j < comp.size(); ++j) {
            const auto cj = comp[j];
            NumberType c = sv[j] * scalar_product(Jq.j_flux(ci, cj), normal) +
                           scalar_product(sg[j], scalar_product(Jq.j_grad_flux(ci, cj), normal));
            if constexpr (reads_hessians) c += scalar_product(sh[j], scalar_product(Jq.j_hess_flux(ci, cj), normal));
            out.jacobian(i, j) += w * sv[i] * c;
          }
          for (uint e = 0; e < n_extr; ++e)
            out.extractor_jacobian(i, e) += w * sv[i] * scalar_product(Jq.j_extr_flux(ci, e), normal);
        }
      }

      const uint max_stacked_points;

      uint n_q = 0, n_q_face = 0;
      std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
      /// Boundary faces in cell order; those of cell k are [face_begin[k], face_begin[k + 1]).
      std::vector<std::pair<typename DoFHandler<dim>::active_cell_iterator, uint>> boundary_faces;
      std::vector<size_t> face_begin;
      /// Cell indices k by color, and whether cell k has a constrained dof.
      std::vector<std::vector<size_t>> colors;
      std::vector<bool> constrained_cell;
      std::unique_ptr<tbb::enumerable_thread_specific<CellScratch>> scratch;
      std::vector<LocalData> local_buffer;

      Batch cell_batch, face_batch;
      std::vector<Tensor<1, dim>> face_normals;
      FluxSourceBatch<dim, NumberType, n_fe> cell_result, face_result;
      std::vector<PointJacobian<dim, n_fe, n_extr>> cell_jacobians, face_jacobians;
      SeedStackWorkspace<dim, n_fe, n_extr, VectorType> cell_workspace, face_workspace;

      PhaseTimes residual_times, jacobian_times;
    };
  } // namespace CG
} // namespace DiFfRG
