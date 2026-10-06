#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/discretization/FEM/assembler/common.hh>
#include <DiFfRG/discretization/common/batched_scatter.hh>
#include <DiFfRG/discretization/common/cell_geometry.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/model/batch.hh>

// external libraries
#include <tbb/enumerable_thread_specific.h>

// standard library
#include <memory>
#include <numeric>

namespace DiFfRG
{
  namespace DG
  {
    using namespace dealii;
    using std::array;

    template <typename... T> auto i_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives", "fe_hessians">>(
          std::tie(t...));
    }

    namespace internal
    {
      /**
       * @brief The DG assembler for models with flux, source and numerical flux; DG::Assembler and
       * dDG::Assembler are its two configurations.
       *
       * @tparam with_derivatives false for DG: the model sees only the FE values. true for dDG: it also sees
       * derivatives and hessians, unless it declares batch_reads_derivatives / batch_reads_hessians = false.
       *
       * Residual and jacobian are assembled in three phases, as in CG::Assembler:
       *  1. gather: the FE solution at every quadrature point of the locally owned cells, their boundary faces,
       *     and both traces at every interior face adjacent to them, into PointBatches;
       *  2. evaluate: the model's evaluate_batch at the cell points, its boundary numflux and its numerical
       *     flux (numflux_batch, see def::LLFFlux) over the whole batches, or their jacobians, see
       *     model/batch.hh. Each interior face is evaluated once;
       *  3. scatter: every owned cell contracts its cell term and its share of each of its faces into its own
       *     rows, so no two cells write the same row (internal::ColoredCells).
       *
       * Each MPI rank batches its own cells; a face between two ranks is evaluated on both, and no rank writes
       * rows it does not own. The results do not depend on the thread count.
       *
       * Config: /discretization/batched/max_stacked_points bounds the size of one AD evaluation of the
       * jacobian (default: 256 MB of AD inputs), see DiFfRG::internal::seed_stacked_jacobian.
       */
      template <typename Discretization_, typename Model_, bool with_derivatives>
      class BatchedAssembler : public FEMAssembler<Discretization_, Model_>
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

        BatchedAssembler(Discretization &discretization, Model &model, const ConfigTree &config)
            : Base(discretization, model, config),
              quadrature(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
              quadrature_face(fe.degree + 1 + config.get_uint("/discretization/overintegration", 0)),
              max_stacked_points(
                  config.get_uint("/discretization/batched/max_stacked_points", default_stacked_points()))
        {
          static_assert(Components::count_fe_subsystems() == 1, "A DG model cannot have multiple submodels!");
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

          {
            DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
            DoFTools::make_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                            /*keep_constrained_dofs = */ true);
            finalize_la_sparsity<SparseMatrixType>(dsp, sparsity_pattern_mass, discretization.get_locally_owned_dofs(),
                                                   discretization.get_locally_relevant_dofs(),
                                                   discretization.get_communicator());
            reinit_la_matrix(mass_matrix, sparsity_pattern_mass, discretization.get_locally_owned_dofs(),
                             discretization.get_communicator());
            MatrixCreator::create_mass_matrix(dof_handler, quadrature, mass_matrix,
                                              (Function<dim, NumberType> *)nullptr, discretization.get_constraints());
          }
          {
            DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
            DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                                 /*keep_constrained_dofs = */ true);
            finalize_la_sparsity<SparseMatrixType>(
                dsp, sparsity_pattern_jacobian, discretization.get_locally_owned_dofs(),
                discretization.get_locally_relevant_dofs(), discretization.get_communicator());
          }
          setup_faces();
          timings_reinit.push_back(timer.wall_time());
        }

        virtual void rebuild_jacobian_sparsity() override
        {
          DynamicSparsityPattern dsp(discretization.get_locally_relevant_dofs());
          DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, discretization.get_constraints(),
                                               /*keep_constrained_dofs = */ true);
          for (const auto &row : discretization.get_locally_relevant_dofs())
            for (const auto &col : extractor_dof_indices)
              dsp.add(row, col);
          finalize_la_sparsity<SparseMatrixType>(
              dsp, sparsity_pattern_jacobian, discretization.get_locally_owned_dofs(),
              discretization.get_locally_relevant_dofs(), discretization.get_communicator());
        }

        virtual const get_type::SparsityPattern<SparseMatrixType> &get_sparsity_pattern_jacobian() const override
        {
          return sparsity_pattern_jacobian;
        }
        virtual const SparseMatrixType &get_mass_matrix() const override { return mass_matrix; }

        /**
         * @brief The model's cell_indicator integrated over each cell, plus its face_indicator integrated
         * over each interior face, added to the two cells of the face.
         */
        virtual void refinement_indicator(Vector<double> &indicator, const VectorType &solution_global) override
        {
          using Iterator = typename DoFHandler<dim>::active_cell_iterator;
          using Scratch = IndicatorScratch;
          struct CopyData {
            struct Face {
              std::array<uint, 2> cell_indices;
              std::array<double, 2> values;
            };
            std::vector<Face> face_data;
            double value = 0.;
            uint cell_index = 0;
          };
          const auto evaluate = [&](const auto &fe_v, const uint n_q) {
            std::vector<Vector<NumberType>> u(n_q, Vector<NumberType>(n_fe));
            std::vector<std::vector<Tensor<1, dim, NumberType>>> du(n_q, std::vector<Tensor<1, dim, NumberType>>(n_fe));
            std::vector<std::vector<Tensor<2, dim, NumberType>>> ddu(n_q,
                                                                     std::vector<Tensor<2, dim, NumberType>>(n_fe));
            fe_v.get_function_values(solution_global, u);
            fe_v.get_function_gradients(solution_global, du);
            fe_v.get_function_hessians(solution_global, ddu);
            return std::make_tuple(u, du, ddu);
          };

          const auto cell_worker = [&](const Iterator &cell, Scratch &scratch, CopyData &copy_data) {
            auto &fe_v = scratch.fe_values;
            fe_v.reinit(cell);
            copy_data.cell_index = cell->active_cell_index();
            copy_data.value = 0;
            const auto [u, du, ddu] = evaluate(fe_v, fe_v.n_quadrature_points);
            double local = 0.;
            for (const auto &q : fe_v.quadrature_point_indices()) {
              model.cell_indicator(local, fe_v.quadrature_point(q), i_tie(u[q], du[q], ddu[q]));
              copy_data.value += fe_v.JxW(q) * local;
            }
          };
          const auto face_worker = [&](const Iterator &cell, const uint &f, const uint &sf, const Iterator &ncell,
                                       const uint &nf, const uint &nsf, Scratch &scratch, CopyData &copy_data) {
            auto &fe_iv = scratch.fe_interface_values;
            fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
            auto &face = copy_data.face_data.emplace_back();
            face.cell_indices = {cell->active_cell_index(), ncell->active_cell_index()};
            face.values = {0., 0.};
            const uint n_q = fe_iv.n_quadrature_points;
            const auto [u_s, du_s, ddu_s] = evaluate(fe_iv.get_fe_face_values(0), n_q);
            const auto [u_n, du_n, ddu_n] = evaluate(fe_iv.get_fe_face_values(1), n_q);
            array<double, 2> local{};
            for (const auto &q : fe_iv.quadrature_point_indices()) {
              model.face_indicator(local, fe_iv.normal_vector(q), fe_iv.quadrature_point(q),
                                   i_tie(u_s[q], du_s[q], ddu_s[q]), i_tie(u_n[q], du_n[q], ddu_n[q]));
              face.values[0] += fe_iv.JxW(q) * local[0] * (1. + cell->at_boundary());
              face.values[1] += fe_iv.JxW(q) * local[1] * (1. + ncell->at_boundary());
            }
          };
          const auto copier = [&](const CopyData &c) {
            for (const auto &face : c.face_data)
              for (uint j = 0; j < 2; ++j)
                indicator[face.cell_indices[j]] += face.values[j];
            indicator[c.cell_index] += c.value;
          };

          Scratch scratch(mapping, fe, quadrature, quadrature_face);
          CopyData copy_data;
          // assemble_ghost_faces_once gives each partition-boundary face to exactly one rank, which is what the
          // sum_reduce of the indicator under the distributed policy expects.
          const auto flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_own_interior_faces_once |
                             MeshWorker::assemble_ghost_faces_once;
          // map() is collective and each rank visits only its own cells; see NoMapsHere.
          const NoMapsHere no_maps_during_assembly;
          const auto schedule = schedule_for(assembly_cost::local_fe);
          MeshWorker::mesh_loop(locally_owned_cells(dof_handler), cell_worker, copier, scratch, copy_data, flags,
                                nullptr, face_worker, schedule.queue_length, schedule.chunk_size);
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
          cell_result.reinit(cell_batch.size(), Term::flux | Term::source);
          model.evaluate_batch(cell_result, cell_batch);
          boundary_result.reinit(boundary_batch.size(), Term::flux);
          evaluate_boundary_numflux(model, boundary_result, boundary_normals, boundary_batch);
          face_result.reinit(face_batch[0].size(), Term::flux);
          evaluate_numflux(model, face_result, face_normals, face_batch[0], face_batch[1]);
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
            for (size_t f = topology.boundary_begin[k]; f < topology.boundary_begin[k + 1]; ++f) {
              auto &fe_fv = s.fe_face_values;
              fe_fv.reinit(cells[k], topology.boundary_faces[f].second);
              for (const auto &q : fe_fv.quadrature_point_indices())
                for (uint i = 0; i < r.size(); ++i) {
                  const auto c = s.comp[i];
                  r(i) += weight * fe_fv.JxW(q) * fe_fv.shape_value_component(i, q, c) *
                          normal_component(boundary_result, c, f * n_q_face + q, fe_fv.normal_vector(q));
                }
            }
            // [[phi_i]] * numflux * n: the trace of phi_i on its own side, with a minus sign on side 1.
            for (size_t ref = topology.face_ref_begin[k]; ref < topology.face_ref_begin[k + 1]; ++ref) {
              const auto [f, side] = topology.face_refs[ref];
              const auto &fe_iv = reinit_interface(s, f);
              const auto &fe_fv = fe_iv.get_fe_face_values(side);
              const double sign = side == 0 ? 1. : -1.;
              for (const auto &q : fe_iv.quadrature_point_indices())
                for (uint i = 0; i < r.size(); ++i) {
                  const auto c = s.comp[i];
                  r(i) += sign * weight * fe_iv.JxW(q) * fe_fv.shape_value_component(i, q, c) *
                          normal_component(face_result, c, f * n_q_face + q, fe_iv.normal_vector(q));
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
          evaluate_boundary_numflux_jacobian(model, boundary_jacobians, boundary_normals, boundary_batch,
                                             max_stacked_points, boundary_workspace);
          evaluate_numflux_jacobian(model, face_jacobians, face_normals, face_batch[0], face_batch[1],
                                    max_stacked_points, face_workspace);
          jacobian_times.evaluate += phase.wall_time();

          phase.restart();
          scatter_jacobian(jacobian, [&](const size_t k, CellScratch &s, LocalData &out) {
            add_mass_jacobian(s, out.jacobian, solution_global, solution_global_dot, alpha, beta);
            const auto &fe_v = s.fe_values;
            for (const auto &q : fe_v.quadrature_point_indices()) {
              s.trace[0].cache(fe_v, q, s.comp);
              add_flux_source_jacobian(s, out, cell_jacobians[k * n_q + q], weight * fe_v.JxW(q));
            }
            for (size_t f = topology.boundary_begin[k]; f < topology.boundary_begin[k + 1]; ++f) {
              auto &fe_fv = s.fe_face_values;
              fe_fv.reinit(cells[k], topology.boundary_faces[f].second);
              for (const auto &q : fe_fv.quadrature_point_indices()) {
                s.trace[0].cache(fe_fv, q, s.comp);
                add_face_jacobian(s, out, out.jacobian, 0, 0, boundary_jacobians[f * n_q_face + q],
                                  fe_fv.normal_vector(q), weight * fe_fv.JxW(q), true);
              }
            }
            for (size_t ref = topology.face_ref_begin[k]; ref < topology.face_ref_begin[k + 1]; ++ref) {
              const auto [f, side] = topology.face_refs[ref];
              const auto &fe_iv = reinit_interface(s, f);
              const uint slot = ref - topology.face_ref_begin[k];
              auto &neighbor = out.neighbor_jacobian[slot];
              neighbor.reinit(out.dofs.size(), fe.n_dofs_per_cell());
              out.neighbor_dofs[slot].resize(fe.n_dofs_per_cell());
              topology.faces[f].cell[1 - side]->get_dof_indices(out.neighbor_dofs[slot]);
              const double sign = side == 0 ? 1. : -1.;
              for (const auto &q : fe_iv.quadrature_point_indices()) {
                s.trace[0].cache(fe_iv.get_fe_face_values(0), q, s.comp);
                s.trace[1].cache(fe_iv.get_fe_face_values(1), q, s.comp);
                const auto &Jq = face_jacobians[f * n_q_face + q];
                const NumberType w = sign * weight * fe_iv.JxW(q);
                add_face_jacobian(s, out, out.jacobian, side, side, Jq[side], fe_iv.normal_vector(q), w, false);
                add_face_jacobian(s, out, neighbor, side, 1 - side, Jq[1 - side], fe_iv.normal_vector(q), w, false);
                add_face_extractor_jacobian(s, out, side, Jq[0], fe_iv.normal_vector(q), w);
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
          SummaryEvent result{.component = with_derivatives ? "dDG" : "DG"};
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
        using Base::schedule_for;

        QGauss<dim> quadrature;
        QGauss<dim - 1> quadrature_face;

        get_type::SparsityPattern<SparseMatrixType> sparsity_pattern_mass;
        get_type::SparsityPattern<SparseMatrixType> sparsity_pattern_jacobian;
        SparseMatrixType mass_matrix;

        std::vector<double> timings_reinit;
        std::vector<double> timings_residual;
        std::vector<double> timings_jacobian;

      private:
        static constexpr bool reads_derivatives = with_derivatives && batch_reads_derivatives<Model>();
        static constexpr bool reads_hessians = with_derivatives && batch_reads_hessians<Model>();

        static double average(const std::vector<double> &t)
        {
          return t.empty() ? 0. : std::accumulate(t.begin(), t.end(), 0.) / t.size();
        }

        /// About 256 MB of AD inputs and outputs per seed-stacked evaluation.
        static uint default_stacked_points()
        {
          constexpr size_t per_point = sizeof(autodiff::real) * n_fe *
                                       (2 + dim + (reads_derivatives ? dim : 0) + (reads_hessians ? dim * dim : 0));
          return std::max<size_t>(1, (size_t(256) << 20) / per_point);
        }

        static UpdateFlags gather_flags()
        {
          return update_values | update_gradients | update_quadrature_points | update_JxW_values |
                 (reads_hessians ? update_hessians : update_default);
        }

        /// FE data of refinement_indicator's mesh_loop.
        struct IndicatorScratch {
          static UpdateFlags flags()
          {
            return update_values | update_gradients | update_hessians | update_quadrature_points | update_JxW_values;
          }
          IndicatorScratch(const Mapping<dim> &mapping, const FiniteElement<dim> &fe, const Quadrature<dim> &q,
                           const Quadrature<dim - 1> &q_face)
              : fe_values(mapping, fe, q, flags()),
                fe_interface_values(mapping, fe, q_face, flags() | update_normal_vectors)
          {
          }
          IndicatorScratch(const IndicatorScratch &s)
              : IndicatorScratch(s.fe_values.get_mapping(), s.fe_values.get_fe(), s.fe_values.get_quadrature(),
                                 s.fe_interface_values.get_quadrature())
          {
          }
          FEValues<dim> fe_values;
          FEInterfaceValues<dim> fe_interface_values;
        };

        /// One cell's contribution to its own rows of the global system, before insertion.
        struct LocalData {
          std::vector<types::global_dof_index> dofs;
          /// Whether a dof of the cell is constrained, i.e. insertion has to go through the constraints.
          bool constrained = false;
          Vector<NumberType> residual;
          FullMatrix<NumberType> jacobian;
          /// Columns of the neighbors across the cell's interior faces, one block per face.
          std::vector<std::vector<types::global_dof_index>> neighbor_dofs;
          std::vector<FullMatrix<NumberType>> neighbor_jacobian;
          FullMatrix<NumberType> extractor_jacobian;
          FullMatrix<NumberType> extractor_dependence;
        };

        /// Shape values, gradients and (if read) hessians of every dof of one cell at one point. The gradients
        /// are needed even when the model reads no derivatives: the flux is tested with them.
        struct Shapes {
          explicit Shapes(const uint n_dofs) : v(n_dofs), g(n_dofs), h(n_dofs) {}
          template <typename FEV> void cache(const FEV &fe_v, const uint q, const std::vector<uint> &comp)
          {
            for (uint i = 0; i < comp.size(); ++i) {
              v[i] = fe_v.shape_value_component(i, q, comp[i]);
              g[i] = fe_v.shape_grad_component(i, q, comp[i]);
              if constexpr (reads_hessians) h[i] = fe_v.shape_hessian_component(i, q, comp[i]);
            }
          }
          std::vector<double> v;
          std::vector<Tensor<1, dim>> g;
          std::vector<Tensor<2, dim>> h;
        };

        /// Per-thread FE data, for the gather and the scatter.
        struct CellScratch {
          CellScratch(const Mapping<dim> &mapping, const FiniteElement<dim> &fe,
                      const dealii::Quadrature<dim> &quadrature, const dealii::Quadrature<dim - 1> &quadrature_face,
                      const UpdateFlags flags)
              : fe_values(mapping, fe, quadrature, flags),
                fe_face_values(mapping, fe, quadrature_face, flags | update_normal_vectors),
                fe_interface_values(mapping, fe, quadrature_face, flags | update_normal_vectors),
                values(quadrature.size(), Vector<NumberType>(n_fe)), values_dot(values),
                gradients(quadrature.size(), std::vector<Tensor<1, dim, NumberType>>(n_fe)),
                hessians(quadrature.size(), std::vector<Tensor<2, dim, NumberType>>(n_fe)),
                face_values(quadrature_face.size(), Vector<NumberType>(n_fe)),
                face_gradients(quadrature_face.size(), std::vector<Tensor<1, dim, NumberType>>(n_fe)),
                face_hessians(quadrature_face.size(), std::vector<Tensor<2, dim, NumberType>>(n_fe)),
                comp(fe.n_dofs_per_cell()), trace{Shapes(fe.n_dofs_per_cell()), Shapes(fe.n_dofs_per_cell())}
          {
            for (uint i = 0; i < comp.size(); ++i)
              comp[i] = fe.system_to_component_index(i).first;
          }

          FEValues<dim> fe_values;
          FEFaceValues<dim> fe_face_values;
          FEInterfaceValues<dim> fe_interface_values;
          std::vector<Vector<NumberType>> values, values_dot;
          std::vector<std::vector<Tensor<1, dim, NumberType>>> gradients;
          std::vector<std::vector<Tensor<2, dim, NumberType>>> hessians;
          std::vector<Vector<NumberType>> face_values;
          std::vector<std::vector<Tensor<1, dim, NumberType>>> face_gradients;
          std::vector<std::vector<Tensor<2, dim, NumberType>>> face_hessians;
          std::vector<uint> comp;
          /// Shapes at the current point, of the cell (trace[0]) or of both traces of an interior face.
          std::array<Shapes, 2> trace;
          LocalData local;
        };

        /// Number the owned cells, their boundary and interior faces, and color the cells for the scatter.
        void setup_faces()
        {
          n_q = quadrature.size();
          n_q_face = quadrature_face.size();
          cells.reinit(dof_handler, discretization.get_constraints());
          topology.reinit(cells);
          scratch = std::make_unique<tbb::enumerable_thread_specific<CellScratch>>(
              [this]() { return CellScratch(mapping, fe, quadrature, quadrature_face, gather_flags()); });
        }

        const FEInterfaceValues<dim> &reinit_interface(CellScratch &s, const size_t f) const
        {
          return topology.reinit(s.fe_interface_values, f, dof_handler);
        }

        /// The flux of component c at point p of @p result, dotted with @p normal.
        static NumberType normal_component(const BatchOutput<dim, NumberType, n_fe> &result, const size_t c,
                                           const size_t p, const Tensor<1, dim> &normal)
        {
          NumberType value = 0.;
          for (uint d = 0; d < dim; ++d)
            value += result.flux(c, d)[p] * normal[d];
          return value;
        }

        /// Phase 1: the solution at every quadrature point of the owned cells and their faces.
        void gather(const VectorType &solution_global, const Extractors &extracted_data, const VectorType &variables)
        {
          cell_batch.reinit(cells.size() * n_q, reads_derivatives, reads_hessians);
          boundary_batch.reinit(topology.boundary_faces.size() * n_q_face, reads_derivatives, reads_hessians);
          for (auto &b : face_batch)
            b.reinit(topology.faces.size() * n_q_face, reads_derivatives, reads_hessians);
          for (auto *b : {&cell_batch, &boundary_batch, &face_batch[0], &face_batch[1]})
            b->set_shared(extracted_data, variables);
          boundary_normals.resize(boundary_batch.size());
          face_normals.resize(face_batch[0].size());

          const auto store = [&](Batch &batch, const size_t first, const auto &fe_v, auto &values, auto &gradients,
                                 auto &hessians, const double width) {
            fe_v.get_function_values(solution_global, values);
            if constexpr (reads_derivatives) fe_v.get_function_gradients(solution_global, gradients);
            if constexpr (reads_hessians) fe_v.get_function_hessians(solution_global, hessians);
            for (const auto &q : fe_v.quadrature_point_indices()) {
              const size_t i = first + q;
              for (size_t c = 0; c < n_fe; ++c) {
                batch.value(c, i) = values[q][c];
                for (uint d1 = 0; d1 < dim; ++d1) {
                  if constexpr (reads_derivatives) batch.derivative(c, d1, i) = gradients[q][c][d1];
                  if constexpr (reads_hessians)
                    for (uint d2 = 0; d2 < dim; ++d2)
                      batch.hessian(c, d1, d2, i) = hessians[q][c][d1][d2];
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
              store(cell_batch, k * n_q, s.fe_values, s.values, s.gradients, s.hessians,
                    DiFfRG::internal::cell_width(cells[k]));
            }
          });
          tbb::parallel_for(tbb::blocked_range<size_t>(0, topology.boundary_faces.size()),
                            [&](const tbb::blocked_range<size_t> &r) {
                              auto &s = scratch->local();
                              for (size_t f = r.begin(); f != r.end(); ++f) {
                                const auto &[cell, face_no] = topology.boundary_faces[f];
                                s.fe_face_values.reinit(cell, face_no);
                                store(boundary_batch, f * n_q_face, s.fe_face_values, s.face_values, s.face_gradients,
                                      s.face_hessians, DiFfRG::internal::cell_width(cell));
                                for (uint q = 0; q < n_q_face; ++q)
                                  boundary_normals[f * n_q_face + q] = s.fe_face_values.normal_vector(q);
                              }
                            });
          tbb::parallel_for(tbb::blocked_range<size_t>(0, topology.faces.size()),
                            [&](const tbb::blocked_range<size_t> &r) {
                              auto &s = scratch->local();
                              for (size_t f = r.begin(); f != r.end(); ++f) {
                                const auto &fe_iv = reinit_interface(s, f);
                                for (uint side = 0; side < 2; ++side)
                                  store(face_batch[side], f * n_q_face, fe_iv.get_fe_face_values(side), s.face_values,
                                        s.face_gradients, s.face_hessians,
                                        DiFfRG::internal::cell_width(topology.faces[f].cell[side]));
                                for (uint q = 0; q < n_q_face; ++q)
                                  face_normals[f * n_q_face + q] = fe_iv.normal_vector(q);
                              }
                            });
        }

        /// Phase 3: assemble(k, scratch, local) every owned cell, after reinit of its FEValues and dof indices.
        template <typename Global, typename Assemble, typename Insert>
        void scatter(Global &global, const Assemble &assemble, const Insert &insert)
        {
          cells.scatter(
              global, *scratch, &CellScratch::local, local_buffer,
              [&](const size_t k, CellScratch &s, LocalData &local) {
                s.fe_values.reinit(cells[k]);
                local.dofs.resize(fe.n_dofs_per_cell());
                cells[k]->get_dof_indices(local.dofs);
                local.constrained = cells.is_constrained(k);
                assemble(k, s, local);
              },
              insert);
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
          const bool any_constraints = constraints.n_constraints() > 0;
          scatter(
              jacobian,
              [&](const size_t k, CellScratch &s, LocalData &local) {
                local.jacobian.reinit(local.dofs.size(), local.dofs.size());
                const size_t n_neighbors = topology.face_ref_begin[k + 1] - topology.face_ref_begin[k];
                local.neighbor_jacobian.resize(n_neighbors);
                local.neighbor_dofs.resize(n_neighbors);
                for (auto &block : local.neighbor_jacobian)
                  block.reinit(0, 0);
                if constexpr (n_extr > 0) local.extractor_jacobian.reinit(local.dofs.size(), n_extr);
                assemble(k, s, local);
              },
              [&](LocalData &local) {
                if (local.constrained)
                  constraints.distribute_local_to_global(local.jacobian, local.dofs, jacobian);
                else
                  jacobian.add(local.dofs, local.jacobian, false);
                for (size_t n = 0; n < local.neighbor_jacobian.size(); ++n) {
                  if (local.neighbor_jacobian[n].m() == 0) continue;
                  if (any_constraints)
                    constraints.distribute_local_to_global(local.neighbor_jacobian[n], local.dofs,
                                                           local.neighbor_dofs[n], jacobian);
                  else
                    jacobian.add(local.dofs, local.neighbor_dofs[n], local.neighbor_jacobian[n], false);
                }
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
          const auto &comp = s.comp;
          const auto &[sv, sg, sh] = std::tie(s.trace[0].v, s.trace[0].g, s.trace[0].h);
          for (uint i = 0; i < comp.size(); ++i) {
            const auto ci = comp[i];
            const auto &grad_i = sg[i];
            for (uint j = 0; j < comp.size(); ++j) {
              const auto cj = comp[j];
              NumberType c = sv[j] * (-scalar_product(grad_i, Jq.j_flux(ci, cj)) + sv[i] * Jq.j_source(ci, cj));
              if constexpr (reads_derivatives)
                c += scalar_product(sg[j],
                                    -scalar_product(grad_i, Jq.j_grad_flux(ci, cj)) + sv[i] * Jq.j_grad_source(ci, cj));
              if constexpr (reads_hessians)
                c += scalar_product(sh[j],
                                    -scalar_product(grad_i, Jq.j_hess_flux(ci, cj)) + sv[i] * Jq.j_hess_source(ci, cj));
              out.jacobian(i, j) += w * c;
            }
            for (uint e = 0; e < n_extr; ++e)
              out.extractor_jacobian(i, e) +=
                  w * (-scalar_product(grad_i, Jq.j_extr_flux(ci, e)) + sv[i] * Jq.j_extr_source(ci, e));
          }
        }

        /**
         * @brief M(i, j) += w phi_i d(numflux.n)/d(trial_j) at one face point, for test functions of trace
         * `row_side` and trial functions of trace `column_side`, whose jacobian blocks are Jq. With
         * @p with_extractors, also the extractor blocks of Jq (at boundary faces; interior faces add their
         * total extractor dependence once, see add_face_extractor_jacobian).
         */
        template <typename PJ>
        void add_face_jacobian(const CellScratch &s, LocalData &out, FullMatrix<NumberType> &M, const uint row_side,
                               const uint column_side, const PJ &Jq, const Tensor<1, dim> &normal, const NumberType w,
                               const bool with_extractors) const
        {
          const auto &comp = s.comp;
          const auto &rows = s.trace[row_side];
          const auto &cols = s.trace[column_side];
          for (uint i = 0; i < comp.size(); ++i) {
            const auto ci = comp[i];
            for (uint j = 0; j < comp.size(); ++j) {
              const auto cj = comp[j];
              NumberType c = cols.v[j] * scalar_product(Jq.j_flux(ci, cj), normal);
              if constexpr (reads_derivatives)
                c += scalar_product(cols.g[j], scalar_product(Jq.j_grad_flux(ci, cj), normal));
              if constexpr (reads_hessians)
                c += scalar_product(cols.h[j], scalar_product(Jq.j_hess_flux(ci, cj), normal));
              M(i, j) += w * rows.v[i] * c;
            }
            if (with_extractors)
              for (uint e = 0; e < n_extr; ++e)
                out.extractor_jacobian(i, e) += w * rows.v[i] * scalar_product(Jq.j_extr_flux(ci, e), normal);
          }
        }

        /// The extractor dependence of an interior face's numerical flux, for the test functions of trace `side`.
        template <typename PJ>
        void add_face_extractor_jacobian(const CellScratch &s, LocalData &out, const uint side, const PJ &Jq,
                                         const Tensor<1, dim> &normal, const NumberType w) const
        {
          for (uint i = 0; i < s.comp.size(); ++i)
            for (uint e = 0; e < n_extr; ++e)
              out.extractor_jacobian(i, e) +=
                  w * s.trace[side].v[i] * scalar_product(Jq.j_extr_flux(s.comp[i], e), normal);
        }

        const uint max_stacked_points;

        uint n_q = 0, n_q_face = 0;
        DiFfRG::internal::ColoredCells<dim> cells;
        DiFfRG::internal::FaceTopology<dim> topology;
        std::unique_ptr<tbb::enumerable_thread_specific<CellScratch>> scratch;
        std::vector<LocalData> local_buffer;

        Batch cell_batch, boundary_batch;
        std::array<Batch, 2> face_batch;
        std::vector<Tensor<1, dim>> boundary_normals, face_normals;
        BatchOutput<dim, NumberType, n_fe> cell_result, boundary_result, face_result;
        std::vector<PointJacobian<dim, n_fe, n_fe, n_extr>> cell_jacobians, boundary_jacobians;
        std::vector<FaceJacobian<dim, n_fe, n_fe, n_extr>> face_jacobians;
        SeedStackWorkspace<Batch, n_fe> cell_workspace, boundary_workspace;
        SeedStackWorkspace<Batch, n_fe, 2> face_workspace;

        PhaseTimes residual_times, jacobian_times;
      };
    } // namespace internal

    /**
     * @brief The DG assembler: the model's flux, source and numerical flux see the FE values only. See
     * internal::BatchedAssembler.
     */
    template <typename Discretization,
              typename Model = typename DiFfRG::internal::assembler_model_of<Discretization>::type>
    class Assembler : public internal::BatchedAssembler<Discretization, Model, false>
    {
    public:
      using internal::BatchedAssembler<Discretization, Model, false>::BatchedAssembler;
    };
  } // namespace DG
} // namespace DiFfRG
