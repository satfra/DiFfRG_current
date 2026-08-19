// external libraries
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/sundials/ida.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/PETScDirect.hh>
#include <DiFfRG/timestepping/linear_solver/PETScKrylov.hh>
#include <DiFfRG/timestepping/linear_solver/ScaledGMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/sundials_diagnostics.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/timestepping/sundials_ida.hh>

namespace DiFfRG
{
  using namespace dealii;

  namespace
  {
    constexpr int recoverable_ida_callback_failure = 1;

    template <typename VectorType> double l1_norm_or_nan(const VectorType *vector)
    {
      if (vector == nullptr) return std::numeric_limits<double>::quiet_NaN();
      try {
        return vector->l1_norm();
      } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    template <typename VectorType> bool is_finite_vector(const VectorType &vector)
    {
      return std::isfinite(l1_norm_or_nan(&vector));
    }

    struct IDACallbackTrace {
      bool enabled = false;
      double min_t = 0.;
      uint max_lines = 200;
      bool trace_successes = false;
      double Lambda = -1.;

      size_t call_index = 0;
      size_t printed_lines = 0;
      double last_t = std::numeric_limits<double>::quiet_NaN();
      size_t same_t_index = 0;

      IDACallbackTrace(const bool enabled, const double min_t, const uint max_lines, const bool trace_successes,
                       const double Lambda)
          : enabled(enabled), min_t(min_t), max_lines(max_lines), trace_successes(trace_successes), Lambda(Lambda)
      {
      }

      template <typename IDAType, typename VectorType, typename ResidualType = VectorType>
      void record(const char *phase, const double t, const uint failure_counter, const IDAType &time_stepper,
                  const IDACallbackDiagnostics &callback_diagnostics, const char *outcome, const VectorType *y,
                  const VectorType *y_dot, const ResidualType *residual, const std::string &detail = {})
      {
        ++call_index;
        if (std::isfinite(last_t) && std::abs(t - last_t) <= 1e-14)
          ++same_t_index;
        else {
          same_t_index = 0;
          last_t = t;
        }

        if (!enabled || t < min_t) return;
        const bool failure = std::string(outcome) != "success";
        if (!failure && printed_lines >= max_lines) return;
        if (!trace_successes && !failure) return;

        const auto diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        std::ostringstream stream;
        stream << std::scientific << std::setprecision(16);
        stream << "[IDA TRACE] phase=" << phase << " outcome=" << outcome << " call=" << call_index
               << " same_t=" << same_t_index
               << " hint=" << (same_t_index == 0 ? "first-trial-candidate" : "repeat-corrector-candidate")
               << " t=" << t;
        if (Lambda > 0.) stream << " k=" << std::exp(-t) * Lambda;
        stream << " failure_counter=" << failure_counter;
        if (diagnostics.has_ida) {
          stream << " ida_steps=" << diagnostics.ida_steps << " residual_evals=" << diagnostics.ida_residual_evaluations
                 << " precision_rejects=" << diagnostics.ida_error_test_failures
                 << " nonlinear_iterations=" << diagnostics.ida_nonlinear_iterations
                 << " nonlinear_failures=" << diagnostics.ida_nonlinear_convergence_failures
                 << " step_solve_failures=" << diagnostics.ida_step_solve_failures
                 << " last_h=" << diagnostics.ida_last_step_size << " current_h=" << diagnostics.ida_current_step_size;
        }
        stream << " y_l1=" << l1_norm_or_nan(y) << " ydot_l1=" << l1_norm_or_nan(y_dot)
               << " residual_l1=" << l1_norm_or_nan(residual);
        if (!detail.empty()) stream << " detail=\"" << detail << "\"";
        std::clog << stream.str() << '\n';
        ++printed_lines;
      }

      template <typename IDAType, typename VectorType>
      void record(const char *phase, const double t, const uint failure_counter, const IDAType &time_stepper,
                  const IDACallbackDiagnostics &callback_diagnostics, const char *outcome, const VectorType *y,
                  const VectorType *y_dot, std::nullptr_t, const std::string &detail = {})
      {
        record(phase, t, failure_counter, time_stepper, callback_diagnostics, outcome, y, y_dot,
               static_cast<const VectorType *>(nullptr), detail);
      }
    };

    template <typename VectorType> class IDAErrorDofMonitor
    {
    public:
      IDAErrorDofMonitor(const bool enabled, const uint top_n, const double Lambda)
          : enabled(enabled), top_n(top_n), Lambda(Lambda)
      {
      }

      template <typename IDAType, typename Callback>
      void observe(const double t, const IDAType &time_stepper, const TimesteppingDiagnostics &diagnostics,
                   const VectorType &solution, const Callback &callback)
      {
        if (!diagnostics.has_ida) return;

        const long int previous_error_test_failures = initialized ? last_error_test_failures : 0;
        initialized = true;
        const long int reject_delta = diagnostics.ida_error_test_failures - previous_error_test_failures;
        last_error_test_failures = diagnostics.ida_error_test_failures;
        if (!enabled || reject_delta <= 0) return;

        if constexpr (requires(const IDAType &ida, VectorType &errors, VectorType &weights) {
                        ida.get_error_test_vectors(errors, weights);
                      }) {
          estimated_local_errors.reinit(solution);
          error_weights.reinit(solution);

          if (!time_stepper.get_error_test_vectors(estimated_local_errors, error_weights)) {
            std::clog << "[IDA ERROR DOF DIAG] precision rejects +" << reject_delta << " at t=" << t
                      << ": failed to query IDA local-error vectors\n";
            return;
          }

          IDAErrorDofDiagnostics report;
          report.t = t;
          report.k = Lambda > 0. ? std::exp(-t) * Lambda : std::numeric_limits<double>::quiet_NaN();
          report.reject_delta = reject_delta;
          report.total_rejects = diagnostics.ida_error_test_failures;
          report.ida_steps = diagnostics.ida_steps;
          report.ida_last_step_size = diagnostics.ida_last_step_size;
          report.ida_current_step_size = diagnostics.ida_current_step_size;
          report.ida_current_time = diagnostics.ida_current_time;

          const std::size_t n_dofs = solution.size();
          report.top_dofs.reserve(std::min<std::size_t>(top_n, n_dofs));
          double weighted_error_norm_squared = 0.;
          std::vector<IDAErrorDofRecord> records;
          records.reserve(n_dofs);
          for (std::size_t i = 0; i < n_dofs; ++i) {
            const double estimated_local_error = static_cast<double>(estimated_local_errors[i]);
            const double error_weight = static_cast<double>(error_weights[i]);
            const double contribution = std::abs(estimated_local_error * error_weight);
            weighted_error_norm_squared += contribution * contribution;
            records.push_back({static_cast<types::global_dof_index>(i), static_cast<double>(solution[i]),
                               estimated_local_error, error_weight, contribution});
          }

          if (n_dofs > 0) report.wrms = std::sqrt(weighted_error_norm_squared / static_cast<double>(n_dofs));
          const std::size_t report_count = std::min<std::size_t>(top_n, records.size());
          std::partial_sort(records.begin(), records.begin() + report_count, records.end(),
                            [](const auto &a, const auto &b) { return a.contribution > b.contribution; });
          report.top_dofs.insert(report.top_dofs.end(), records.begin(), records.begin() + report_count);

          if (callback) {
            callback(report);
          } else {
            log_generic_report(report);
          }
        } else {
          std::clog << "[IDA ERROR DOF DIAG] precision rejects +" << reject_delta << " at t=" << t
                    << ": IDA local-error vectors are not available in this build\n";
        }
      }

    private:
      bool enabled = false;
      uint top_n = 0;
      double Lambda = -1.;
      bool initialized = false;
      long int last_error_test_failures = 0;
      VectorType estimated_local_errors;
      VectorType error_weights;

      static void log_generic_report(const IDAErrorDofDiagnostics &report)
      {
        std::clog << std::setprecision(17) << "[IDA ERROR DOF DIAG] precision rejects +" << report.reject_delta
                  << " at t=" << report.t << ", k=" << report.k << ", total_rejects=" << report.total_rejects
                  << ", ida_steps=" << report.ida_steps << ", last_h=" << report.ida_last_step_size
                  << ", current_h=" << report.ida_current_step_size << ", wrms=" << report.wrms << '\n';
        for (std::size_t rank = 0; rank < report.top_dofs.size(); ++rank) {
          const auto &record = report.top_dofs[rank];
          std::clog << std::setprecision(17) << "[IDA ERROR DOF DIAG] rank=" << rank << " dof=" << record.dof
                    << " y=" << record.value << " ele=" << record.estimated_local_error
                    << " ewt=" << record.error_weight << " abs_ele_ewt=" << record.contribution << '\n';
        }
      }
    };
  } // namespace

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      AbstractFlowingVariables<NumberType, VectorType> *initial_condition, const double t_start, const double t_stop)
  {
    this->data_out = this->get_data_out();
    this->adaptor = this->get_adaptor();

    auto &full_data = initial_condition->data();
    if constexpr (dim == 0)
      run_vars(full_data.block(1), t_start, t_stop);
    else {
      if (full_data.n_blocks() != 1)
        run(full_data, t_start, t_stop);
      else
        run(full_data.block(0), t_start, t_stop);
    }
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, LinearSolver>::run(VectorType &initial_data,
                                                                                     const double t_start,
                                                                                     const double t_stop)
  {
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    SparseMatrixType jacobian;
    assembler->reinit_matrix(jacobian);
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    const DiagnosticPort jacobian_diagnostic_port = data_out ? data_out->diagnostic_port() : DiagnosticPort{};

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                impl.max_non_linear_iterations, 0, impl.abs_tol,
                                                                impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = 0.;
    uint failure_counter = 0;
    IDACallbackDiagnostics callback_diagnostics;
    IDACallbackTrace callback_trace(impl.ida_callback_trace, impl.ida_callback_trace_min_t,
                                    impl.ida_callback_trace_max_lines, impl.ida_callback_trace_successes, this->Lambda);
    IDAErrorDofMonitor<VectorType> error_dof_monitor(impl.ida_error_dof_diagnostics,
                                                     impl.ida_error_dof_diagnostics_top_n, this->Lambda);
    bool output_after_failure = false;

    // Initialize initial condition
    VectorType y = initial_data;
    VectorType y_dot = initial_data;
    y_dot *= 0.;

    // Pointer to current residual for monitoring. Must be initialised before
    // SUNDIALS::IDA calls `output_step` at the initial `t = 0` (the IDA wrapper
    // may emit an initial-condition output before the first residual lambda
    // has run and set this pointer). Without this placeholder the first
    // `output_step` dereferences uninitialised garbage and the process
    // segfaults inside `time_stepper.solve_dae`.
    VectorType residual_placeholder = initial_data;
    residual_placeholder *= 0.;
    VectorType *residual = &residual_placeholder;

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement
    time_stepper.solver_should_restart = [&](const double t, VectorType &sol, VectorType &sol_dot) -> bool {
      if ((*adaptor)(t, sol)) {
        assembler->reinit_vector(sol_dot);
        assembler->reinit_matrix(jacobian);
        return true;
      }
      return false;
    };

    time_stepper.differential_components = [&]() { return assembler->get_differential_indices(); };

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](VectorType &v) { assembler->reinit_vector(v); };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    time_stepper.output_step = [&](const double t, const VectorType &sol, const VectorType &sol_dot,
                                   unsigned int /*step_number*/) {
      if (!is_close(last_save, t, 1e-10)) {
        assembler->set_time(t);
        data_out->write_frame(
            t, [&](auto &frame) { assembler->attach_data_output(frame, sol, Vector<double>(), sol_dot, (*residual)); });

        last_save = t;
      }
      if (!output_after_failure && !is_close(t, 0.)) {
        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        error_dof_monitor.observe(t, time_stepper, current_diagnostics, sol, ida_error_dof_callback);
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (failure_counter == 0 && !is_close(t, 0.) && stuck > 100) {
        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        console_out(t, "implicit residual", 1, &current_diagnostics);
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter == 0 && is_close(t, 0.) && stuck > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "failure-limit", &y,
                              &y_dot, &res);
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
        callback_diagnostics.nonfinite_solution_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-y", &y,
                              &y_dot, &res);
        return recoverable_ida_callback_failure;
      }

      assembler->set_time(t);

      res = 0;
      assembler->residual(res, y, 1., y_dot, 1.);
      residual = &res;

      if (!std::isfinite(res.l1_norm())) {
        callback_diagnostics.nonfinite_residual_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-residual",
                              &y, &y_dot, &res);
        return recoverable_ida_callback_failure;
      }
      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      console_out(t, "implicit residual", 1, &current_diagnostics, calc_timer.lap());
      callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            &res);

      failure_counter = 0;
      return 0;
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const VectorType &y, const VectorType &y_dot,
                                      const double alpha) -> int {
      CalcDtTimer calc_timer;
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      const auto build_diagnostics =
          make_ida_jacobian_build_diagnostics(time_stepper, this->next_jacobian_build_id++, alpha);
      JacobianMatrixDiagnostics matrix_diagnostics;
      JacobianFactorizationDiagnostics factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "jacobian_diagnostics.csv", t, build_diagnostics,
                                    matrix_diagnostics, factorization_diagnostics);
      };

      assembler->set_time(t);

      try {
        if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-state",
                                &y, &y_dot, nullptr);
          return recoverable_ida_callback_failure;
        }

        jacobian = 0;
        assembler->jacobian(jacobian, y, 1., y_dot, alpha, 1.);
        matrix_diagnostics = analyze_jacobian_matrix(jacobian);
        if (!std::isfinite(jacobian.frobenius_norm())) {
          factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-jacobian", &y, &y_dot, nullptr);
          return recoverable_ida_callback_failure;
        }
        linSolver.init(jacobian);

        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        console_out(t, "jacobian construction", 2, &current_diagnostics, calc_timer.lap());

        factorize_with_diagnostics(linSolver, jacobian, factorization_diagnostics);
        if (factorization_diagnostics.factorization_success == 1.) {
          const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
          console_out(t, "jacobian inversion", 3, &current_diagnostics, calc_timer.lap());
        }
        record_diagnostics();
      } catch (std::exception &e) {
        if (matrix_diagnostics.n_rows == 0.) matrix_diagnostics = analyze_jacobian_matrix(jacobian);
        if constexpr (decltype(linSolver)::performs_factorization)
          if (std::isnan(factorization_diagnostics.factorization_success))
            factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        callback_diagnostics.jacobian_failures++;
        ++failure_counter;
        callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, nullptr, e.what());
        return recoverable_ida_callback_failure;
      }

      callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            nullptr);
      failure_counter = 0;
      return 0;
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double tol) -> int {
      CalcDtTimer calc_timer;
      try {
        if (!is_finite_vector(src)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-source", &src, &dst, nullptr);
          return recoverable_ida_callback_failure;
        }

        const auto sol_iterations = linSolver.solve(src, dst, tol);
        if (!is_finite_vector(dst)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-solution", &src, &dst, nullptr);
          return recoverable_ida_callback_failure;
        }
        if (sol_iterations >= 0) {
          const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
          console_out(stuck_t, "linear solver (" + std::to_string(sol_iterations) + " it)", 2, &current_diagnostics,
                      calc_timer.lap());
        }
      } catch (std::exception &) {
        callback_diagnostics.linear_solver_failures++;
        ++failure_counter;
        callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "exception",
                              &src, &dst, nullptr);
        return recoverable_ida_callback_failure;
      }
      callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "success",
                            &src, &dst, nullptr);
      return 0;
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      this->log.error("Timestepping failed: {}", e.what());
      output_after_failure = true;
      this->finalize_output_after_failure([&]() { time_stepper.output_step(stuck_t, y, y_dot, 0); });
      throw;
    }

    initial_data = y;
    this->drain_output();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, LinearSolver>::run(BlockVectorType &initial_data,
                                                                                     const double t_start,
                                                                                     const double t_stop)
  {
    if (initial_data.n_blocks() != 2)
      throw std::runtime_error("TimeStepperSUNDIALS_ARKode_vars::run: y must have two blocks!");
    if (initial_data.block(1).size() == 0)
      throw std::runtime_error(
          "TimeStepperSUNDIALS_ARKode_vars::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    SparseMatrixType spatial_jacobian;
    assembler->reinit_matrix(spatial_jacobian);
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    const uint n_FE_dofs = initial_data.block(0).size();
    const uint n_vars = initial_data.block(1).size();
    FullMatrix<NumberType> variable_jacobian(n_vars);
    FullMatrix<NumberType> variable_jacobian_inverse(n_vars);
    const DiagnosticPort jacobian_diagnostic_port = data_out ? data_out->diagnostic_port() : DiagnosticPort{};

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<BlockVectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt,
                                                                     impl.minimal_dt, 5, impl.max_non_linear_iterations,
                                                                     0, impl.abs_tol, impl.rel_tol);
    typename SUNDIALS::IDA<BlockVectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = 0.;
    uint failure_counter = 0;
    IDACallbackDiagnostics callback_diagnostics;
    IDACallbackTrace callback_trace(impl.ida_callback_trace, impl.ida_callback_trace_min_t,
                                    impl.ida_callback_trace_max_lines, impl.ida_callback_trace_successes, this->Lambda);

    // Initialize initial condition
    BlockVectorType y = initial_data;
    BlockVectorType y_dot = initial_data;
    y_dot *= 0.;

    // Pointer to current residual for monitoring. See the VectorType overload
    // for why this needs a placeholder before SUNDIALS::IDA's first
    // `output_step` call at `t = 0`.
    BlockVectorType residual_placeholder = initial_data;
    residual_placeholder *= 0.;
    BlockVectorType *residual = &residual_placeholder;

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement
    time_stepper.solver_should_restart = [&](const double t, BlockVectorType &sol, BlockVectorType &sol_dot) -> bool {
      if ((*adaptor)(t, sol.block(0))) {
        assembler->reinit_vector(sol_dot.block(0));
        assembler->reinit_matrix(spatial_jacobian);
        return true;
      }
      return false;
    };

    time_stepper.differential_components = [&]() {
      IndexSet dof_indices = assembler->get_differential_indices();
      IndexSet differential_indices(n_FE_dofs + n_vars);
      differential_indices.add_indices(dof_indices.begin(), dof_indices.end());
      differential_indices.add_range(n_FE_dofs, n_FE_dofs + n_vars);
      return differential_indices;
    };

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](BlockVectorType &v) {
      v.reinit(2);
      assembler->reinit_vector(v.block(0));
      reinit_la_variables_vector(v.block(1), n_vars, assembler->get_communicator());
      v.collect_sizes();
    };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    time_stepper.output_step = [&](const double t, const BlockVectorType &sol, const BlockVectorType &sol_dot,
                                   unsigned int /*step_number*/) {
      if (!is_close(last_save, t, 1e-10)) {
        assembler->set_time(t);
        data_out->write_frame(t, [&](auto &frame) {
          assembler->attach_data_output(frame, sol.block(0), sol.block(1), sol_dot.block(0), (*residual).block(0));
        });

        last_save = t;
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const BlockVectorType &y, const BlockVectorType &y_dot,
                                BlockVectorType &res) -> int {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t, impl.minimal_dt))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (failure_counter == 0 && !is_close(t, 0.) && stuck > 100) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter == 0 && is_close(t, 0.) && stuck > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter > 200) {
        std::cerr << "timestep failure, at t = " << t << std::endl;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "failure-limit", &y,
                              &y_dot, &res);
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
        callback_diagnostics.nonfinite_solution_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-y", &y,
                              &y_dot, &res);
        return recoverable_ida_callback_failure;
      }

      try {
        res = 0;
        assembler->set_time(t);
        assembler->residual_variables(res.block(1), y.block(1), y.block(0));
        assembler->residual(res.block(0), y.block(0), 1., y_dot.block(0), 1., y.block(1));
        res.block(1) += y_dot.block(1);
        residual = &res;
      } catch (std::exception &e) {
        callback_diagnostics.residual_exceptions++;
        std::cerr << e.what() << std::endl;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, &res, e.what());
        return recoverable_ida_callback_failure;
      }

      if (!std::isfinite(res.l1_norm())) {
        callback_diagnostics.nonfinite_residual_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-residual",
                              &y, &y_dot, &res);
        return recoverable_ida_callback_failure;
      }

      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      console_out(t, "implicit residual", 1, &current_diagnostics, calc_timer.lap());
      callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            &res);

      failure_counter = 0;
      return 0;
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const BlockVectorType &y, const BlockVectorType &y_dot,
                                      const double alpha) -> int {
      CalcDtTimer calc_timer;
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      const auto build_diagnostics =
          make_ida_jacobian_build_diagnostics(time_stepper, this->next_jacobian_build_id++, alpha);
      JacobianMatrixDiagnostics spatial_matrix_diagnostics;
      JacobianMatrixDiagnostics variable_matrix_diagnostics;
      spatial_matrix_diagnostics.n_rows = spatial_jacobian.m();
      variable_matrix_diagnostics.n_rows = variable_jacobian.m();
      JacobianFactorizationDiagnostics spatial_factorization_diagnostics;
      JacobianFactorizationDiagnostics variable_factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "jacobian_diagnostics.csv", t, build_diagnostics,
                                    spatial_matrix_diagnostics, spatial_factorization_diagnostics);
        record_jacobian_diagnostics(jacobian_diagnostic_port, "variable_jacobian_diagnostics.csv", t, build_diagnostics,
                                    variable_matrix_diagnostics, variable_factorization_diagnostics);
      };

      try {
        if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
          spatial_factorization_diagnostics.factorization_success = 0.;
          variable_factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-state",
                                &y, &y_dot, nullptr);
          return recoverable_ida_callback_failure;
        }

        spatial_jacobian = 0;
        variable_jacobian = 0;
        assembler->set_time(t);
        assembler->jacobian(spatial_jacobian, y.block(0), 1., y_dot.block(0), alpha, 1., y.block(1));
        assembler->jacobian_variables(variable_jacobian, y.block(1), y.block(0));
        variable_jacobian *= -1.;
        variable_jacobian.diagadd(alpha);
        spatial_matrix_diagnostics = analyze_jacobian_matrix(spatial_jacobian);
        variable_matrix_diagnostics = analyze_jacobian_matrix(variable_jacobian);

        if (!std::isfinite(spatial_matrix_diagnostics.frobenius_norm) ||
            !std::isfinite(variable_matrix_diagnostics.frobenius_norm)) {
          spatial_factorization_diagnostics.factorization_success = 0.;
          variable_factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-jacobian", &y, &y_dot, nullptr);
          return recoverable_ida_callback_failure;
        }

        linSolver.init(spatial_jacobian);

        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        console_out(t, "jacobian construction", 2, &current_diagnostics, calc_timer.lap());

        factorize_with_diagnostics(linSolver, spatial_jacobian, spatial_factorization_diagnostics);
        const auto variable_factorization_start = std::chrono::steady_clock::now();
        try {
          variable_jacobian_inverse.invert(variable_jacobian);
          variable_factorization_diagnostics.factorization_success = 1.;
        } catch (...) {
          variable_factorization_diagnostics.factorization_success = 0.;
          variable_factorization_diagnostics.factorization_ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - variable_factorization_start)
                  .count();
          throw;
        }
        variable_factorization_diagnostics.factorization_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - variable_factorization_start)
                .count();
        const auto current_diagnostics_after_inversion =
            make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        console_out(t, "jacobian inversion", 3, &current_diagnostics_after_inversion, calc_timer.lap());
        record_diagnostics();
      } catch (std::exception &e) {
        if constexpr (decltype(linSolver)::performs_factorization)
          if (std::isnan(spatial_factorization_diagnostics.factorization_success))
            spatial_factorization_diagnostics.factorization_success = 0.;
        if (std::isnan(variable_factorization_diagnostics.factorization_success))
          variable_factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        callback_diagnostics.jacobian_failures++;
        std::cerr << e.what() << std::endl;
        ++failure_counter;
        callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, nullptr, e.what());
        return recoverable_ida_callback_failure;
      }

      callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            nullptr);
      failure_counter = 0;
      return 0;
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const BlockVectorType &src, BlockVectorType &dst, const double tol) -> int {
      CalcDtTimer calc_timer;
      try {
        if (!is_finite_vector(src)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-source", &src, &dst, nullptr);
          return recoverable_ida_callback_failure;
        }

        const auto sol_iterations = linSolver.solve(src.block(0), dst.block(0), tol);
        variable_jacobian_inverse.vmult(dst.block(1), src.block(1));
        if (!is_finite_vector(dst)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-solution", &src, &dst, nullptr);
          return recoverable_ida_callback_failure;
        }

        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        if (sol_iterations >= 0)
          console_out(stuck_t, "linear solver (" + std::to_string(sol_iterations) + " it)", 2, &current_diagnostics,
                      calc_timer.lap());
        else
          console_out(stuck_t, "linear solver", 2, &current_diagnostics, calc_timer.lap());
      } catch (std::exception &) {
        callback_diagnostics.linear_solver_failures++;
        ++failure_counter;
        callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "exception",
                              &src, &dst, nullptr);
        return recoverable_ida_callback_failure;
      }
      callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "success",
                            &src, &dst, nullptr);
      return 0;
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      this->log.error("Timestepping failed: {}", e.what());
      this->finalize_output_after_failure([&]() { time_stepper.output_step(stuck_t, y, y_dot, 0); });
      throw;
    }

    initial_data = y;
    this->drain_output();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, LinearSolver>::run_vars(VectorType &initial_data,
                                                                                          const double t_start,
                                                                                          const double t_stop)
  {
    if (initial_data.size() == 0)
      throw std::runtime_error("TimeStepperSUNDIALS_IDA::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    const uint n_vars = initial_data.size();
    FullMatrix<NumberType> variable_jacobian(n_vars);
    FullMatrix<NumberType> variable_jacobian_inverse(n_vars);
    const DiagnosticPort jacobian_diagnostic_port = data_out ? data_out->diagnostic_port() : DiagnosticPort{};

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                impl.max_non_linear_iterations, 0, impl.abs_tol,
                                                                impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = 0.;
    uint failure_counter = 0;
    IDACallbackDiagnostics callback_diagnostics;
    IDACallbackTrace callback_trace(impl.ida_callback_trace, impl.ida_callback_trace_min_t,
                                    impl.ida_callback_trace_max_lines, impl.ida_callback_trace_successes, this->Lambda);
    IDAErrorDofMonitor<VectorType> error_dof_monitor(impl.ida_error_dof_diagnostics,
                                                     impl.ida_error_dof_diagnostics_top_n, this->Lambda);
    bool output_after_failure = false;

    // Initialize initial condition
    VectorType y = initial_data;
    VectorType y_dot = initial_data;
    y_dot *= 0.;

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](VectorType &v) {
      reinit_la_variables_vector(v, n_vars, assembler->get_communicator());
    };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    time_stepper.output_step = [&](const double t, const VectorType &sol, const VectorType & /*sol_dot*/,
                                   uint /*step_number*/) {
      if (!is_close(last_save, t, 1e-10)) {
        assembler->set_time(t);
        data_out->write_frame(t, [&](auto &frame) { assembler->attach_data_output(frame, Vector<double>(), sol); });

        last_save = t;
      }
      if (!output_after_failure && !is_close(t, 0.)) {
        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        error_dof_monitor.observe(t, time_stepper, current_diagnostics, sol, ida_error_dof_callback);
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t, impl.minimal_dt))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (failure_counter == 0 && !is_close(t, 0.) && stuck > 100) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter == 0 && is_close(t, 0.) && stuck > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter > 200) {
        std::cerr << "timestep failure, at t = " << t << std::endl;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "failure-limit", &y,
                              &y_dot, &res);
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
        callback_diagnostics.nonfinite_solution_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-y", &y,
                              &y_dot, &res);
        return recoverable_ida_callback_failure;
      }

      try {
        res = 0;
        assembler->set_time(t);
        assembler->residual_variables(res, y, VectorType());
        res += y_dot;
      } catch (std::exception &e) {
        callback_diagnostics.residual_exceptions++;
        std::cerr << e.what() << std::endl;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, &res, e.what());
        return recoverable_ida_callback_failure;
      }

      if (!std::isfinite(res.l1_norm())) {
        callback_diagnostics.nonfinite_residual_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-residual",
                              &y, &y_dot, &res);
        return recoverable_ida_callback_failure;
      }

      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      console_out(t, "implicit residual", 1, &current_diagnostics, calc_timer.lap());
      callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            &res);

      failure_counter = 0;
      return 0;
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const VectorType &y, const VectorType &y_dot,
                                      const double alpha) -> int {
      CalcDtTimer calc_timer;
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      const auto build_diagnostics =
          make_ida_jacobian_build_diagnostics(time_stepper, this->next_jacobian_build_id++, alpha);
      JacobianMatrixDiagnostics matrix_diagnostics;
      matrix_diagnostics.n_rows = variable_jacobian.m();
      JacobianFactorizationDiagnostics factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "variable_jacobian_diagnostics.csv", t, build_diagnostics,
                                    matrix_diagnostics, factorization_diagnostics);
      };

      try {
        if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
          factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-state",
                                &y, &y_dot, nullptr);
          return recoverable_ida_callback_failure;
        }

        variable_jacobian = 0;
        assembler->set_time(t);
        assembler->jacobian_variables(variable_jacobian, y, VectorType());
        variable_jacobian *= -1.;
        variable_jacobian.diagadd(alpha);
        matrix_diagnostics = analyze_jacobian_matrix(variable_jacobian);

        if (!std::isfinite(matrix_diagnostics.frobenius_norm)) {
          factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-variable-jacobian", &y, &y_dot, nullptr);
          return recoverable_ida_callback_failure;
        }

        const auto factorization_start = std::chrono::steady_clock::now();
        try {
          variable_jacobian_inverse.invert(variable_jacobian);
          factorization_diagnostics.factorization_success = 1.;
        } catch (...) {
          factorization_diagnostics.factorization_success = 0.;
          factorization_diagnostics.factorization_ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - factorization_start).count();
          throw;
        }
        factorization_diagnostics.factorization_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - factorization_start).count();
        record_diagnostics();
      } catch (std::exception &e) {
        if (std::isnan(factorization_diagnostics.factorization_success))
          factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        callback_diagnostics.jacobian_failures++;
        std::cerr << e.what() << std::endl;
        ++failure_counter;
        callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, nullptr, e.what());
        return recoverable_ida_callback_failure;
      }

      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      console_out(t, "jacobian", 2, &current_diagnostics, calc_timer.lap());
      callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            nullptr);

      failure_counter = 0;
      return 0;
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double) -> int {
      try {
        if (!is_finite_vector(src)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-source", &src, &dst, nullptr);
          return recoverable_ida_callback_failure;
        }

        variable_jacobian_inverse.vmult(dst, src);
        if (!is_finite_vector(dst)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-solution", &src, &dst, nullptr);
          return recoverable_ida_callback_failure;
        }
      } catch (std::exception &) {
        callback_diagnostics.linear_solver_failures++;
        ++failure_counter;
        callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "exception",
                              &src, &dst, nullptr);
        return recoverable_ida_callback_failure;
      }
      callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "success",
                            &src, &dst, nullptr);
      return 0;
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      this->log.error("Timestepping failed: {}", e.what());
      output_after_failure = true;
      this->finalize_output_after_failure([&]() { time_stepper.output_step(stuck_t, y, y_dot, 0); });
      throw;
    }

    initial_data = y;
    this->drain_output();
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                               DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                               DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 0, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, DiFfRG::GMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                               DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                               DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                               DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                               DiFfRG::ScaledUMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                               DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                               DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                               DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                               DiFfRG::ScaledUMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                               DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                               DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                               DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                               DiFfRG::ScaledGMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                               DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                               DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                               DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                               DiFfRG::ScaledGMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                               DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                               DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                               DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                               DiFfRG::GMRES>;

// ##############################################################################
// Distributed (PETSc-backed) instantiations -- one root cause left
// ##############################################################################
//
// Measured by instantiating TimeStepperSUNDIALS_IDA for
// PETScWrappers::MPI::{Vector,SparseMatrix} x dim{1,2,3} and counting compiler errors:
//
//     before the Stage B fixes:  12.3 errors per instantiation
//     after them:                10.0
//     after Stage D2:             2.0
//
// Cleared in D2, all by routing construction through the assembler instead of doing it here:
//
//   matrix construction   -> assembler->reinit_matrix(), which knows the owned row set and the
//                            communicator. A PETSc matrix has no constructor from a bare pattern.
//   variables vectors     -> reinit_la_variables_vector(), which shares its layout with
//                            reinit_la_block_vector() so block 1 cannot be sized two ways.
//   `Vector<double>()`    -> `VectorType()` at the two residual_variables/jacobian_variables calls.
//
// What is left is ONE cause at two sites, :765 and :1019:
//
//     variable_jacobian_inverse.vmult(dst.block(1), src.block(1));
//
// A dense FullMatrix applied to the distributed variables block. This is not a typing problem and
// should not be papered over as one. Block 1 is owned outright by rank 0 (see
// la_policy.hh::variables_owner_set), so on every other rank src.block(1) is locally empty --
// a correct implementation has to gather it, apply the dense inverse redundantly, and scatter the
// owned part back. That is a collective on IDA's critical path, so it belongs with Stage E+F,
// where the IDA control-flow predicates become collective too. Adding a collective here while
// those predicates are still rank-local would introduce a deadlock rather than fix anything.
