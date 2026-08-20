#pragma once

// DiFfRG
#include <DiFfRG/common/types.hh>
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
#include <iomanip>
#include <limits>
#include <optional>
#include <unordered_map>
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
   * @brief Stopwatch feeding the `calc_dt` console column.
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

  struct TimesteppingDiagnostics {
    bool has_ida = false;
    long int ida_steps = 0;
    long int ida_error_test_failures = 0;
    long int ida_nonlinear_convergence_failures = 0;
    long int ida_step_solve_failures = 0;
    long int ida_residual_evaluations = 0;
    long int ida_nonlinear_iterations = 0;
    double ida_last_step_size = 0.;
    double ida_current_step_size = 0.;
    double ida_current_time = 0.;
    bool has_ida_step_iteration_stats = false;
    double ida_min_step_size = 0.;
    double ida_average_step_size = 0.;
    long int ida_step_size_samples = 0;
    double ida_average_nonlinear_iterations_per_step = 0.;
    long int ida_max_nonlinear_iterations_per_step = 0;

    bool has_callback = false;
    size_t nonfinite_solution_failures = 0;
    size_t nonfinite_residual_failures = 0;
    size_t residual_exceptions = 0;
    size_t jacobian_failures = 0;
    size_t linear_solver_failures = 0;

    size_t nonfinite_failures() const { return nonfinite_solution_failures + nonfinite_residual_failures; }
    bool empty() const { return !has_ida && !has_callback; }
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
   * - /timestepping/explicit/dt: The timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/minimal_dt: The minimal timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/maximal_dt: The maximal timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/abs_tol: The absolute tolerance for an explicit timestepping algorithm.
   * - /timestepping/explicit/rel_tol: The relative tolerance for an explicit timestepping algorithm.
   * - /timestepping/explicit/detect_stuck: Whether repeated-time callback detection is enabled.
   *
   * Additionally, the following parameters are being used:
   * - /output/verbosity: The verbosity level of the output. At 0 nothing is printed, at 1 the progress lines for
   * residual evaluations, at 2 additionally the jacobian/linear solver lines and the IDA step/failure diagnostics
   * line, at 3 additionally jacobian inversion and output frames. From 4 upwards the per-second aggregation is
   * disabled and every call is printed.
   * - /physical/Lambda: The RG scale parameter Lambda. If not present, no RG scale is given when console_out is called.
   *
   * The console_out method is used to print information about the current time, the calculation time and the current
   * RG scale k to the console in a standardized way.
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
     * @param adaptor The adaptor object is used to adapt the mesh and the solution vector to the new mesh. It can be
     * nullptr if no adaptation is needed.
     * @param implicit_stepper, explicit_stepper which /timestepping/ sections this stepper reads.
     */
    AbstractTimestepper(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler,
                        OutputSession<dim, VectorType> *data_out, AbstractAdaptor<VectorType> *adaptor,
                        const bool implicit_stepper, const bool explicit_stepper)
        : config(config), assembler(assembler), data_out(data_out), adaptor(adaptor),
          log(data_out ? data_out->log_port() : LogPort{}), start_time(std::chrono::high_resolution_clock::now()),
          m_is_implicit(implicit_stepper), m_is_explicit(explicit_stepper)
    {
      verbosity = config.get_int("/output/verbosity", 0);
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

      // -1 marks "no cutoff scale given"; the timesteppers use it only to label output, so
      // unlike def::fRG this must not warn - plenty of models have no /physical/Lambda at all.
      Lambda = config.get_double("/physical/Lambda", -1.0);
    }

    /**
     * @brief Obtain the run-owned output session supplied by the application.
     *
     * @return OutputSession<dim, VectorType>* The active output session.
     */
    OutputSession<dim, VectorType> *get_data_out()
    {
      if (data_out == nullptr)
        throw std::invalid_argument("AbstractTimestepper: an OutputSession must be supplied explicitly.");
      return data_out;
    }

    /**
     * @brief Utility function to obtain an Adaptor object. If no Adaptor object is provided, a default one is created,
     * which is the NoAdaptivity object, i.e. no mesh adaptivity is used.
     *
     * @return AbstractAdaptor<VectorType>* A pointer to the Adaptor object.
     */
    AbstractAdaptor<VectorType> *get_adaptor()
    {
      if (adaptor == nullptr) {
        adaptor_default = std::make_shared<NoAdaptivity<VectorType>>();
        return adaptor_default.get();
      }
      return adaptor;
    }

    void drain_output()
    {
      if (data_out) data_out->drain();
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
     * @param initial_condition A pointer to a flowing variables object that contains the initial condition.
     * @param t_start The start time of the simulation.
     * @param t_stop The run method will evolve the system from t_start to t_stop.
     */
    virtual void run(AbstractFlowingVariables<NumberType, VectorType> *initial_condition, const double t_start,
                     const double t_stop) = 0;

    bool is_implicit() const { return m_is_implicit; }
    bool is_explicit() const { return m_is_explicit; }

  protected:
    const ConfigTree config;
    AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler;
    OutputSession<dim, VectorType> *data_out;
    AbstractAdaptor<VectorType> *adaptor;
    LogPort log;

    const std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

    // Declared after start_time so the declaration order matches the member-initializer order.
    const bool m_is_implicit;
    const bool m_is_explicit;

    std::shared_ptr<NoAdaptivity<VectorType>> adaptor_default;
    double Lambda;
    int verbosity;
    double output_dt;
    struct ImplicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
      uint max_steps;
      uint max_non_linear_iterations;
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

    static constexpr size_t console_name_width = 21;

    // Minimum /output/verbosity at which the IDA step/failure diagnostics line is printed below the
    // progress line. Level 1 stays a plain one-line-per-second progress report.
    static constexpr int diagnostics_verbosity_level = 2;

    struct IDAAcceptedStepStats {
      long int last_steps = 0;
      long int last_nonlinear_iterations = 0;
      double last_time = 0.;
      long int observed_steps = 0;
      long int observed_nonlinear_iterations = 0;
      long int max_nonlinear_iterations_per_step = 0;
      bool initialized = false;

      void add(TimesteppingDiagnostics &diagnostics)
      {
        if (!diagnostics.has_ida) return;

        if (!initialized) {
          initialized = true;
        }

        const long int step_delta = diagnostics.ida_steps - last_steps;
        const long int nonlinear_delta = diagnostics.ida_nonlinear_iterations - last_nonlinear_iterations;

        if (step_delta < 0 || nonlinear_delta < 0) {
          last_steps = diagnostics.ida_steps;
          last_nonlinear_iterations = diagnostics.ida_nonlinear_iterations;
          last_time = diagnostics.ida_current_time;
          return;
        }

        const bool has_new_accepted_steps = step_delta > 0;
        if (has_new_accepted_steps) {
          observed_steps += step_delta;
          observed_nonlinear_iterations += nonlinear_delta;

          const long int per_step_delta =
              step_delta == 1 ? nonlinear_delta : (nonlinear_delta + step_delta - 1) / step_delta;
          max_nonlinear_iterations_per_step = std::max(max_nonlinear_iterations_per_step, per_step_delta);

          if (diagnostics.ida_last_step_size > 0. && std::isfinite(diagnostics.ida_last_step_size)) {
            diagnostics.ida_min_step_size = diagnostics.ida_last_step_size;
          }
          if (diagnostics.ida_current_time >= last_time && std::isfinite(diagnostics.ida_current_time)) {
            diagnostics.ida_average_step_size =
                (diagnostics.ida_current_time - last_time) / static_cast<double>(step_delta);
            diagnostics.ida_step_size_samples = step_delta;
          }
        }

        last_steps = diagnostics.ida_steps;
        last_nonlinear_iterations = diagnostics.ida_nonlinear_iterations;
        last_time = diagnostics.ida_current_time;

        if (observed_steps > 0) {
          diagnostics.has_ida_step_iteration_stats = true;
          diagnostics.ida_average_nonlinear_iterations_per_step =
              static_cast<double>(observed_nonlinear_iterations) / static_cast<double>(observed_steps);
          diagnostics.ida_max_nonlinear_iterations_per_step = max_nonlinear_iterations_per_step;
        }
        if (!has_new_accepted_steps) {
          diagnostics.ida_min_step_size = 0.;
          diagnostics.ida_average_step_size = 0.;
          diagnostics.ida_step_size_samples = 0;
        }
      }
    };

    struct ConsoleOutStats {
      size_t first_ms = 0;
      size_t last_ms = 0;
      double latest_t = 0.;
      bool has_entry = false;
      TimesteppingDiagnostics diagnostics;
      bool has_diagnostics = false;
      double step_size_sum = 0.;
      long int step_size_samples = 0;
      // Number of console_out calls aggregated into the current window, reported
      // alongside the calc_dt timing.
      size_t calls = 0;

      // Wall time spent in the measured operation (residual evaluation, jacobian
      // construction/inversion, linear solve), aggregated over the window and
      // reported as mean(uncertainty).
      size_t calc_dt_calls = 0;
      double calc_dt_sum_ms = 0.;
      double calc_dt_min_ms = std::numeric_limits<double>::max();
      double calc_dt_max_ms = 0.;
      bool has_calc_dt = false;

      void add(const double t, const size_t milliseconds, const TimesteppingDiagnostics *latest_diagnostics,
               const double calc_dt_ms = -1.0)
      {
        if (!has_entry) first_ms = milliseconds;
        has_entry = true;
        last_ms = milliseconds;
        latest_t = t;
        ++calls;

        if (calc_dt_ms >= 0.) {
          has_calc_dt = true;
          ++calc_dt_calls;
          calc_dt_sum_ms += calc_dt_ms;
          calc_dt_min_ms = std::min(calc_dt_min_ms, calc_dt_ms);
          calc_dt_max_ms = std::max(calc_dt_max_ms, calc_dt_ms);
        }

        if (latest_diagnostics != nullptr && !latest_diagnostics->empty()) {
          const bool had_step_stats = has_diagnostics && diagnostics.has_ida_step_iteration_stats;
          const double previous_min_step_size = diagnostics.ida_min_step_size;
          const double previous_step_size_sum = step_size_sum;
          const long int previous_step_size_samples = step_size_samples;
          diagnostics = *latest_diagnostics;
          if (had_step_stats && previous_min_step_size > 0. && diagnostics.has_ida_step_iteration_stats) {
            if (diagnostics.ida_min_step_size > 0.)
              diagnostics.ida_min_step_size = std::min(previous_min_step_size, diagnostics.ida_min_step_size);
            else
              diagnostics.ida_min_step_size = previous_min_step_size;
          }
          if (previous_step_size_samples > 0) {
            step_size_sum = previous_step_size_sum;
            step_size_samples = previous_step_size_samples;
          }
          if (latest_diagnostics->ida_average_step_size > 0. && latest_diagnostics->ida_step_size_samples > 0) {
            step_size_sum += latest_diagnostics->ida_average_step_size *
                             static_cast<double>(latest_diagnostics->ida_step_size_samples);
            step_size_samples += latest_diagnostics->ida_step_size_samples;
          }
          if (step_size_samples > 0)
            diagnostics.ida_average_step_size = step_size_sum / static_cast<double>(step_size_samples);
          has_diagnostics = true;
        }
      }

      double mean_calc_dt_ms() const { return calc_dt_sum_ms / double(calc_dt_calls); }
      double calc_dt_uncertainty_ms() const { return 0.5 * (calc_dt_max_ms - calc_dt_min_ms); }
    };

    mutable std::unordered_map<std::string, ConsoleOutStats> console_out_stats_by_category;
    mutable std::unordered_map<std::string, TimesteppingDiagnostics> last_printed_diagnostics_by_category;
    mutable IDAAcceptedStepStats ida_accepted_step_stats;
    std::size_t next_jacobian_build_id = 0;
    TimestepperJacobianDiagnosticsState jacobian_diagnostics_state;

    static std::string console_out_category(const std::string &name) { return name.substr(0, name.find(" (")); }

    /**
     * @brief Render a mean with its uncertainty as e.g. "12.3(4)ms".
     *
     * The uncertainty is rounded to one significant digit and the mean is
     * printed with matching precision.
     */
    static std::string format_uncertain_ms(const double mean_ms, const double uncertainty_ms)
    {
      const double abs_uncertainty = std::abs(uncertainty_ms);
      const bool has_uncertainty = abs_uncertainty > 0.;
      int decimals = 0;
      long rounded_uncertainty = 0;

      if (has_uncertainty) {
        double exponent = std::floor(std::log10(abs_uncertainty));
        double scale = std::pow(10., -exponent);
        rounded_uncertainty = long(std::round(abs_uncertainty * scale));
        if (rounded_uncertainty >= 10) {
          exponent += 1.;
          scale = std::pow(10., -exponent);
          rounded_uncertainty = long(std::round(abs_uncertainty * scale));
        }
        decimals = std::max(0, int(-exponent));
      } else if (std::abs(mean_ms) < 1.) {
        decimals = 2;
      } else if (std::abs(mean_ms) < 100.) {
        decimals = 1;
      }

      std::stringstream stream;
      stream << std::fixed << std::setprecision(decimals) << mean_ms << "(" << rounded_uncertainty << ")ms";
      return stream.str();
    }

    static std::string format_diagnostics_delta(const size_t latest, const size_t previous)
    {
      std::stringstream stream;
      stream << latest;
      if (latest >= previous) stream << "(+" << latest - previous << ")";
      return stream.str();
    }

    static std::string format_diagnostics_delta(const long int latest, const long int previous)
    {
      std::stringstream stream;
      stream << latest;
      if (latest >= previous) stream << "(+" << latest - previous << ")";
      return stream.str();
    }

    static std::string format_scientific(const double value)
    {
      std::stringstream stream;
      stream << std::scientific << value;
      return stream.str();
    }

    static std::string format_diagnostics(const TimesteppingDiagnostics &latest,
                                          const TimesteppingDiagnostics *previous)
    {
      const TimesteppingDiagnostics zero;
      const auto &prev = previous != nullptr ? *previous : zero;

      std::stringstream stream;
      stream << "accepted steps " << format_diagnostics_delta(latest.ida_steps, prev.ida_steps);
      stream << ", precision rejects "
             << format_diagnostics_delta(latest.ida_error_test_failures, prev.ida_error_test_failures);
      stream << ", nonlinear failures "
             << format_diagnostics_delta(latest.ida_nonlinear_convergence_failures,
                                         prev.ida_nonlinear_convergence_failures);
      if (latest.has_ida_step_iteration_stats) {
        if (latest.ida_min_step_size > 0.) stream << ", min step width " << format_scientific(latest.ida_min_step_size);
        if (latest.ida_average_step_size > 0.)
          stream << ", avg step width " << format_scientific(latest.ida_average_step_size);
        stream << ", nonlinear it/step avg " << format_scientific(latest.ida_average_nonlinear_iterations_per_step);
        stream << ", max " << latest.ida_max_nonlinear_iterations_per_step;
      }
      stream << ", accumulated step failures "
             << format_diagnostics_delta(latest.ida_step_solve_failures, prev.ida_step_solve_failures);
      stream << ", NaN/Inf callbacks "
             << format_diagnostics_delta(latest.nonfinite_failures(), prev.nonfinite_failures());
      return stream.str();
    }

    void print_console_out(const std::string &name, const double t, const size_t milliseconds,
                           const std::optional<std::string> diagnostics = std::nullopt, const size_t calls = 0,
                           const std::optional<std::string> calc_dt = std::nullopt) const
    {
      if (name.size() > console_name_width) throw std::runtime_error("console_out: log label is too long: " + name);

      const std::ios_base::fmtflags oldflags = std::cout.flags();
      const std::streamsize oldprecision = std::cout.precision();
      std::cout << "[" << std::setw(console_name_width) << std::left << name << "]";
      constexpr auto console_time_precision = std::numeric_limits<double>::max_digits10;
      std::cout << " t: " << std::setw(24) << std::left << std::setprecision(console_time_precision) << std::scientific
                << t;
      if (Lambda > 0.0) {
        std::cout << " | k: " << std::setw(24) << std::left << std::setprecision(console_time_precision)
                  << std::scientific << exp(-t) * Lambda;
      }
      if (calc_dt.has_value()) {
        std::cout << " | calc_dt: " << std::setw(11) << calc_dt.value();
      }
      if (calls > 0) {
        std::cout << " | calls: " << std::setw(4) << calls;
      }
      std::cout << " | calc_t: " << time_format_ms(milliseconds);
      std::cout << std::endl;
      if (diagnostics.has_value()) std::cout << "  " << diagnostics.value() << std::endl;
      std::cout.flags(oldflags);
      std::cout.precision(oldprecision);
    }

    /**
     * @brief Pretty-print the status of the timestepping algorithm to the console.
     *
     * @param t Current time.
     * @param name A tag prepended to the output.
     * @param verbosity_level The verbosity level of the output.
     * @param diagnostics Optional timestepper diagnostics printed as a second line. Only printed from verbosity
     * diagnostics_verbosity_level (2) upwards; below that the pointer is ignored and the progress line stays alone.
     * @param calc_dt_ms Wall time in milliseconds spent in the operation being reported. Negative
     * values mean "not measured" and suppress the calc_dt column. At verbosity < 4 the value is
     * aggregated over the reporting window and shown as mean(uncertainty).
     */
    void console_out(const double t, const std::string name, const int verbosity_level,
                     const TimesteppingDiagnostics *diagnostics = nullptr, const double calc_dt_ms = -1.0) const
    {
      if (verbosity < verbosity_level) return;
      if (verbosity < diagnostics_verbosity_level) diagnostics = nullptr;

      const size_t milliseconds =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time)
              .count();
      const std::string category = console_out_category(name);
      std::optional<TimesteppingDiagnostics> diagnostics_with_stats;
      if (diagnostics != nullptr) {
        diagnostics_with_stats = *diagnostics;
        ida_accepted_step_stats.add(*diagnostics_with_stats);
        diagnostics = &*diagnostics_with_stats;
      }
      if (verbosity >= 4) {
        const auto previous_it = last_printed_diagnostics_by_category.find(category);
        const auto formatted_diagnostics =
            diagnostics != nullptr && !diagnostics->empty()
                ? std::optional<std::string>(format_diagnostics(
                      *diagnostics,
                      previous_it != last_printed_diagnostics_by_category.end() ? &previous_it->second : nullptr))
                : std::nullopt;
        const auto calc_dt =
            calc_dt_ms >= 0.0 ? std::optional<std::string>(time_format_ms(size_t(calc_dt_ms))) : std::nullopt;
        print_console_out(name, t, milliseconds, formatted_diagnostics, 0, calc_dt);
        if (diagnostics != nullptr && !diagnostics->empty())
          last_printed_diagnostics_by_category[category] = *diagnostics;
        return;
      }

      auto &stats = console_out_stats_by_category[category];
      stats.add(t, milliseconds, diagnostics, calc_dt_ms);
      if (milliseconds - stats.first_ms < 1000) return;

      const auto previous_it = last_printed_diagnostics_by_category.find(category);
      const auto formatted_diagnostics =
          stats.has_diagnostics
              ? std::optional<std::string>(format_diagnostics(
                    stats.diagnostics,
                    previous_it != last_printed_diagnostics_by_category.end() ? &previous_it->second : nullptr))
              : std::nullopt;
      const auto calc_dt =
          stats.has_calc_dt
              ? std::optional<std::string>(format_uncertain_ms(stats.mean_calc_dt_ms(), stats.calc_dt_uncertainty_ms()))
              : std::nullopt;
      print_console_out(category, stats.latest_t, stats.last_ms, formatted_diagnostics, stats.calls, calc_dt);
      if (stats.has_diagnostics) last_printed_diagnostics_by_category[category] = stats.diagnostics;
      stats = {};
    }
  };
} // namespace DiFfRG
