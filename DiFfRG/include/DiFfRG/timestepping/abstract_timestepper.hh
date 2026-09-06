#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/no_adaptivity.hh>
#include <DiFfRG/timestepping/jacobian_diagnostics.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace DiFfRG
{
  struct IDAErrorDofRecord {
    types::global_dof_index dof = dealii::numbers::invalid_dof_index;
    double value = 0.;
    double estimated_local_error = 0.;
    double error_weight = 0.;
    double contribution = 0.;
  };

  /**
   * @brief Stopwatch feeding structured progress durations.
   *
   * `lap()` returns the milliseconds elapsed since construction or the previous
   * `lap()` and re-arms, so consecutive operations inside one callback
   * (residual, jacobian construction, jacobian inversion, linear solve) each
   * report their own duration rather than a cumulative total.
   */
  struct CalcDtTimer {
    std::chrono::high_resolution_clock::time_point mark = std::chrono::high_resolution_clock::now();

    double lap()
    {
      const auto now = std::chrono::high_resolution_clock::now();
      const double ms = double(std::chrono::duration_cast<std::chrono::milliseconds>(now - mark).count());
      mark = now;
      return ms;
    }
  };

  struct IDAErrorDofDiagnostics {
    double t = 0.;
    double k = std::numeric_limits<double>::quiet_NaN();
    long int reject_delta = 0;
    long int total_rejects = 0;
    long int ida_steps = 0;
    double ida_last_step_size = 0.;
    double ida_current_step_size = 0.;
    double ida_current_time = 0.;
    double wrms = 0.;
    std::vector<IDAErrorDofRecord> top_dofs;
  };

  struct IDAProgressDiagnostics {
    long int ida_steps = 0;
    long int ida_error_test_failures = 0;
    long int ida_nonlinear_convergence_failures = 0;
    long int ida_step_solve_failures = 0;
    long int ida_residual_evaluations = 0;
    long int ida_nonlinear_iterations = 0;
    double ida_last_step_size = 0.;
    double ida_current_step_size = 0.;
    double ida_current_time = 0.;
    void append_to(ProgressEvent &event) const
    {
      event.field("h", ida_current_step_size, 2)
          .field("last_h", ida_last_step_size, 2)
          .field("steps", ida_steps, 2)
          .field("rejects", ida_error_test_failures, 2)
          .field("nl_fail", ida_nonlinear_convergence_failures, 2);
    }
  };

  struct SolverCallbackDiagnostics {
    size_t nonfinite_solution_failures = 0;
    size_t nonfinite_residual_failures = 0;
    size_t residual_exceptions = 0;
    size_t jacobian_failures = 0;
    size_t linear_solver_failures = 0;

    size_t nonfinite_failures() const { return nonfinite_solution_failures + nonfinite_residual_failures; }
    bool has_failures() const
    {
      return nonfinite_failures() > 0 || residual_exceptions > 0 || jacobian_failures > 0 || linear_solver_failures > 0;
    }
    void append_to(ProgressEvent &event) const
    {
      if (!has_failures()) return;
      event.field("nan", nonfinite_failures(), 2)
          .field("res_exc", residual_exceptions, 2)
          .field("jac_fail", jacobian_failures, 2)
          .field("lin_fail", linear_solver_failures, 2);
    }
  };

  struct TimesteppingDiagnostics {
    std::optional<IDAProgressDiagnostics> ida;
    SolverCallbackDiagnostics callbacks;

    void append_to(ProgressEvent &event) const
    {
      if (ida) ida->append_to(event);
      callbacks.append_to(event);
    }
    bool empty() const { return !ida && !callbacks.has_failures(); }
  };

  /**
   * @brief The abstract base class for all timestepping algorithms.
   * It provides a standard constructor which populates typical timestepping parameters from a given ConfigTree object,
   * such as the timestep sizes, tolerances, verbosity, etc. that are used in the timestepping algorithms.
   *
   * In the ConfigTree object, a /timestepping/ section must be present with the following parameters:
   * - /timestepping/output_dt: The output timestep size.
   * - /timestepping/implicit/dt: The timestep size for an implicit timestepping algorithm.
   * - /timestepping/implicit/minimal_dt: The minimal timestep size for an implicit timestepping algorithm.
   * - /timestepping/implicit/maximal_dt: The maximal timestep size for an implicit timestepping algorithm.
   * - /timestepping/implicit/abs_tol: The absolute tolerance for an implicit timestepping algorithm.
   * - /timestepping/implicit/rel_tol: The relative tolerance for an implicit timestepping algorithm.
   * - /timestepping/implicit/max_steps: The maximal number of internal SUNDIALS steps between outputs.
   * - /timestepping/implicit/max_non_linear_iterations: The maximal number of nonlinear IDA iterations.
   * - /timestepping/implicit/jacobian_diagnostics: Whether the Jacobian diagnostics tables are written (default false).
   * - /timestepping/explicit/dt: The timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/minimal_dt: The minimal timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/maximal_dt: The maximal timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/abs_tol: The absolute tolerance for an explicit timestepping algorithm.
   * - /timestepping/explicit/rel_tol: The relative tolerance for an explicit timestepping algorithm.
   * - /timestepping/explicit/detect_stuck: Whether repeated-time callback detection is enabled.
   *
   * Additionally, the following parameters are being used:
   * - /output/verbosity: At 0 no progress is printed; 1 reports residual work; 2 adds Jacobian, linear-solver, and
   * solver diagnostics; 3 adds factorization and output work. Levels 1--4 are aggregated by the run reporter.
   * Level 5 prints every progress event and is intended only for debugging.
   *
   * Output settings, including the optional RG scale, are obtained from the typed ReportPort owned by the
   * OutputSession. Timesteppers submit ProgressEvents directly; they never write to a stream.
   *
   * @tparam VectorType_ The type of the vector used in the timestepping algorithm. Must satisfy
   * SupportedVectorType: dealii::Vector<double> and dealii::BlockVector<double>, plus the PETSc MPI
   * vectors in an MPI build.
   * @tparam SparseMatrixType_ The type of the sparse matrix used in the timestepping algorithm. This depends on the
   * assembler used in the computation.
   * @tparam dim_ The dimensionality of the spatial discretization.
   */
  template <typename VectorType_, typename SparseMatrixType_, uint dim_> class AbstractTimestepper
  {
  protected:
    static constexpr uint dim = dim_;
    using VectorType = VectorType_;
    using NumberType = typename get_type::NumberType<VectorType>;
    using SparseMatrixType = SparseMatrixType_;
    // NOTE: InverseSparseMatrixType is deliberately NOT declared here. A member alias of a
    // class template is instantiated with the class, so declaring it in the common base forces
    // every timestepper to have a mass-matrix inverse type -- including the implicit ones,
    // which never invert a mass matrix. That made the whole hierarchy unusable with any matrix
    // type lacking a SparseDirectUMFPACK-shaped inverse (i.e. every distributed matrix). It now
    // lives only in the explicit steppers that actually use it.
    static_assert(SupportedVectorType<VectorType>,
                  "VectorType is not a vector type DiFfRG supports; see get_type in common/types.hh.");
    using BlockVectorType = typename get_type::BlockVectorType<VectorType>;

  public:
    /**
     * @brief Construct a new Abstract Timestepper object
     *
     * @param config The ConfigTree object must contain a /timestepping/ section with all necessary parameters.
     * @param assembler The assembler object is used to assemble the system matrices and vectors for the timestepping
     * algorithm.
     * @param data_out The data output object is used to write the output data to disk.
     * @param adaptor The adaptor object is used to adapt the mesh and the solution vector to the new mesh. The
     * overload without it uses NoAdaptivity, i.e. no mesh adaptation.
     * @param implicit_stepper, explicit_stepper which /timestepping/ sections this stepper reads.
     */
    AbstractTimestepper(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                        OutputSession_impl<dim, VectorType> &data_out, const bool implicit_stepper,
                        const bool explicit_stepper)
        : config(config), adaptor_default(), assembler(assembler), data_out(data_out), adaptor(adaptor_default),
          log(data_out.report_port()), m_is_implicit(implicit_stepper), m_is_explicit(explicit_stepper)
    {
      read_parameters();
    }

    AbstractTimestepper(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                        OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor,
                        const bool implicit_stepper, const bool explicit_stepper)
        : config(config), adaptor_default(), assembler(assembler), data_out(data_out), adaptor(adaptor),
          log(data_out.report_port()), m_is_implicit(implicit_stepper), m_is_explicit(explicit_stepper)
    {
      read_parameters();
    }

  private:
    void read_parameters()
    {
      output_dt = config.get_double("/timestepping/output_dt", 1e-1);

      if (m_is_implicit) {
        // Stuff you should really set
        impl.abs_tol = config.get_double_or_warn("/timestepping/implicit/abs_tol", 1e-13);
        impl.rel_tol = config.get_double_or_warn("/timestepping/implicit/rel_tol", 1e-7);

        // Stuff you can set, but defaults are reasonable
        impl.dt = config.get_double("/timestepping/implicit/dt", 1e-4);
        impl.minimal_dt = config.get_double("/timestepping/implicit/minimal_dt", 1e-8);
        impl.maximal_dt = config.get_double("/timestepping/implicit/maximal_dt", 1.);
        impl.max_steps = config.get_uint("/timestepping/implicit/max_steps", 1e6);
        impl.max_non_linear_iterations = config.get_uint("/timestepping/implicit/max_non_linear_iterations", 10);
        impl.jacobian_diagnostics = config.get_bool("/timestepping/implicit/jacobian_diagnostics", false);
        impl.ida_callback_trace = config.get_bool("/timestepping/implicit/ida_callback_trace", false);
        impl.ida_callback_trace_min_t = config.get_double("/timestepping/implicit/ida_callback_trace_min_t", 0.0);
        impl.ida_callback_trace_max_lines = config.get_uint("/timestepping/implicit/ida_callback_trace_max_lines", 200);
        impl.ida_callback_trace_successes =
            config.get_bool("/timestepping/implicit/ida_callback_trace_successes", false);
        impl.ida_error_dof_diagnostics = config.get_bool("/timestepping/implicit/ida_error_dof_diagnostics", false);
        impl.ida_error_dof_diagnostics_top_n =
            config.get_uint("/timestepping/implicit/ida_error_dof_diagnostics_top_n", 8);

        // Sanity checks:
        if (impl.minimal_dt <= 0.0) throw std::invalid_argument("Minimal timestep size must be positive.");
        if (impl.maximal_dt <= 0.0) throw std::invalid_argument("Maximal timestep size must be positive.");
        if (impl.minimal_dt > impl.maximal_dt)
          throw std::invalid_argument("Minimal timestep size must be smaller than maximal timestep size.");
        if (impl.dt < impl.minimal_dt || impl.dt > impl.maximal_dt)
          throw std::invalid_argument("Initial timestep size must be within the minimal and maximal timestep size.");
        if (impl.abs_tol <= 0.0) throw std::invalid_argument("Absolute tolerance must be > 0.");
        if (impl.rel_tol <= 0.0) throw std::invalid_argument("Relative tolerance must be > 0.");
      }

      if (m_is_explicit) {
        expl.dt = config.get_double_or_warn("/timestepping/explicit/dt", 1e-2);
        expl.minimal_dt = config.get_double("/timestepping/explicit/minimal_dt", 1e-16);
        expl.maximal_dt = config.get_double("/timestepping/explicit/maximal_dt", 1e16);
        expl.abs_tol = config.get_double_or_warn("/timestepping/explicit/abs_tol", 1e-3);
        expl.rel_tol = config.get_double_or_warn("/timestepping/explicit/rel_tol", 1e-3);
        expl.detect_stuck = config.get_bool("/timestepping/explicit/detect_stuck", true);

        // Sanity checks:
        if (expl.minimal_dt <= 0.0) throw std::invalid_argument("Minimal timestep size must be positive.");
        if (expl.maximal_dt <= 0.0) throw std::invalid_argument("Maximal timestep size must be positive.");
        if (expl.minimal_dt > expl.maximal_dt)
          throw std::invalid_argument("Minimal timestep size must be smaller than maximal timestep size.");
        if (expl.dt < expl.minimal_dt || expl.dt > expl.maximal_dt)
          throw std::invalid_argument("Initial timestep size must be within the minimal and maximal timestep size.");
        if (expl.abs_tol <= 0.0) throw std::invalid_argument("Absolute tolerance must be > 0.");
        if (expl.rel_tol <= 0.0) throw std::invalid_argument("Relative tolerance must be > 0.");
      }
    }

  protected:
    void drain_output()
    {
      log.summary(assembler.summary());
      data_out.drain();
    }

    /**
     * @brief Best-effort output cleanup while a timestepping failure is being reported.
     *
     * Writes one final frame for the state that failed, then pushes everything pending to
     * disk. Both steps are swallowed on error: a readout that cannot cope with a diverged
     * solution must not replace the failure the caller is actually trying to report, and a
     * drain error is latched anyway and resurfaces at finish().
     *
     * Callers rethrow the original exception afterwards.
     */
    template <typename EmitFinalFrame> void finalize_output_after_failure(EmitFinalFrame &&emit_final_frame)
    {
      try {
        std::forward<EmitFinalFrame>(emit_final_frame)();
      } catch (const std::exception &e) {
        log.error("Output of the final frame after the failure failed: {}", e.what());
      } catch (...) {
        log.error("Output of the final frame after the failure failed.");
      }
      try {
        drain_output();
      } catch (const std::exception &e) {
        log.error("Draining pending output after the failure failed: {}", e.what());
      } catch (...) {
        log.error("Draining pending output after the failure failed.");
      }
    }

    /**
     * @brief Any derived class must implement this method to run the timestepping algorithm.
     *
     * @param initial_condition The flowing variables object that contains the initial condition.
     * @param t_start The start time of the simulation.
     * @param t_stop The run method will evolve the system from t_start to t_stop.
     */
    virtual void run(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start,
                     const double t_stop) = 0;

    bool is_implicit() const { return m_is_implicit; }
    bool is_explicit() const { return m_is_explicit; }

  protected:
    const ConfigTree config;
    NoAdaptivity<VectorType> adaptor_default;
    AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler;
    OutputSession_impl<dim, VectorType> &data_out;
    AbstractAdaptor<VectorType> &adaptor;
    ReportPort log;

    const bool m_is_implicit;
    const bool m_is_explicit;


    double output_dt;
    struct ImplicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
      uint max_steps;
      uint max_non_linear_iterations;
      /** Enables the `<run>_jacobian_diagnostics.csv` tables. Off by default because the records
       * are not free: each one costs a full sweep over the assembled Jacobian and, for factorizing
       * solvers, a condition estimate worth several extra triangular solves. */
      bool jacobian_diagnostics;
      bool ida_callback_trace;
      double ida_callback_trace_min_t;
      uint ida_callback_trace_max_lines;
      bool ida_callback_trace_successes;
      bool ida_error_dof_diagnostics;
      uint ida_error_dof_diagnostics_top_n;
    } impl;

    struct ExplicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
      bool detect_stuck;
    } expl;

    std::size_t next_jacobian_build_id = 0;
    TimestepperJacobianDiagnosticsState jacobian_diagnostics_state;
  };
} // namespace DiFfRG
