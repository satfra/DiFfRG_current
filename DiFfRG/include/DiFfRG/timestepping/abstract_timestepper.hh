#pragma once

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/discretization/mesh/no_adaptivity.hh>
#include <algorithm>
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
   * It provides a standard constructor which populates typical timestepping parameters from a given JSONValue object,
   * such as the timestep sizes, tolerances, verbosity, etc. that are used in the timestepping algorithms.
   *
   * In the JSONValue object, a /timestepping/ section must be present with the following parameters:
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
   * - /output/verbosity: The verbosity level of the output.
   * - /physical/Lambda: The RG scale parameter Lambda. If not present, no RG scale is given when console_out is called.
   *
   * The console_out method is used to print information about the current time, the calculation time and the current
   * RG scale k to the console in a standardized way.
   *
   * @tparam VectorType_ The type of the vector used in the timestepping algorithm.. Currently only Vector<double> is
   * supported.
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
    using InverseSparseMatrixType = typename get_type::InverseSparseMatrixType<SparseMatrixType>;
    static_assert(std::is_same_v<VectorType, Vector<NumberType>>, "VectorType must not be Vector<double>!");
    using BlockVectorType = dealii::BlockVector<NumberType>;

  public:
    /**
     * @brief Construct a new Abstract Timestepper object
     *
     * @param json The JSONValue object must contain a /timestepping/ section with all necessary parameters.
     * @param assembler
     * @param data_out
     * @param adaptor
     */
    AbstractTimestepper(const JSONValue &json, AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler,
                        DataOutput<dim, VectorType> *data_out = nullptr, AbstractAdaptor<VectorType> *adaptor = nullptr)
        : json(json), assembler(assembler), data_out(data_out), adaptor(adaptor),
          start_time(std::chrono::high_resolution_clock::now())
    {
      verbosity = json.get_int("/output/verbosity");
      output_dt = json.get_double("/timestepping/output_dt");

      impl.dt = json.get_double("/timestepping/implicit/dt");
      impl.minimal_dt = json.get_double("/timestepping/implicit/minimal_dt");
      impl.maximal_dt = json.get_double("/timestepping/implicit/maximal_dt");
      impl.abs_tol = json.get_double("/timestepping/implicit/abs_tol");
      impl.rel_tol = json.get_double("/timestepping/implicit/rel_tol");
      impl.max_steps = json.get_uint("/timestepping/implicit/max_steps", 1000000);
      impl.max_non_linear_iterations = json.get_uint("/timestepping/implicit/max_non_linear_iterations", 10);
      impl.ida_callback_trace = json.get_bool("/timestepping/implicit/ida_callback_trace", false);
      impl.ida_callback_trace_min_t = json.get_double("/timestepping/implicit/ida_callback_trace_min_t", 0.0);
      impl.ida_callback_trace_max_lines = json.get_uint("/timestepping/implicit/ida_callback_trace_max_lines", 200);
      impl.ida_callback_trace_successes = json.get_bool("/timestepping/implicit/ida_callback_trace_successes", false);
      impl.ida_error_dof_diagnostics = json.get_bool("/timestepping/implicit/ida_error_dof_diagnostics", false);
      impl.ida_error_dof_diagnostics_top_n = json.get_uint("/timestepping/implicit/ida_error_dof_diagnostics_top_n", 8);

      expl.dt = json.get_double("/timestepping/explicit/dt");
      expl.minimal_dt = json.get_double("/timestepping/explicit/minimal_dt");
      expl.maximal_dt = json.get_double("/timestepping/explicit/maximal_dt");
      expl.abs_tol = json.get_double("/timestepping/explicit/abs_tol");
      expl.rel_tol = json.get_double("/timestepping/explicit/rel_tol");
      expl.detect_stuck = json.get_bool("/timestepping/explicit/detect_stuck", true);

      try {
        Lambda = json.get_double("/physical/Lambda");
      } catch (std::exception &e) {
        Lambda = -1.0;
      }
    }

    /**
     * @brief Utility function to obtain a DataOutput object. If no DataOutput object is provided, a default one is
     * created.
     *
     * @return DataOutput<dim, VectorType>* A pointer to the DataOutput object.
     */
    DataOutput<dim, VectorType> *get_data_out()
    {
      if (data_out == nullptr) {
        data_out_default = std::make_shared<DataOutput<dim, VectorType>>(json);
        return data_out_default.get();
      }
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

    /**
     * @brief Any derived class must implement this method to run the timestepping algorithm.
     *
     * @param initial_condition A pointer to a flowing variables object that contains the initial condition.
     * @param t_start The start time of the simulation.
     * @param t_stop The run method will evolve the system from t_start to t_stop.
     */
    virtual void run(AbstractFlowingVariables<NumberType> *initial_condition, const double t_start,
                     const double t_stop) = 0;

  protected:
    const JSONValue json;
    AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler;
    DataOutput<dim, VectorType> *data_out;
    AbstractAdaptor<VectorType> *adaptor;

    const std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

    std::shared_ptr<NoAdaptivity<VectorType>> adaptor_default;
    std::shared_ptr<DataOutput<dim, VectorType>> data_out_default;

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

      void add(const double t, const size_t milliseconds, const TimesteppingDiagnostics *latest_diagnostics)
      {
        if (!has_entry) first_ms = milliseconds;
        has_entry = true;
        last_ms = milliseconds;
        latest_t = t;

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
    };

    mutable std::unordered_map<std::string, ConsoleOutStats> console_out_stats_by_category;
    mutable std::unordered_map<std::string, TimesteppingDiagnostics> last_printed_diagnostics_by_category;
    mutable IDAAcceptedStepStats ida_accepted_step_stats;

    static std::string console_out_category(const std::string &name) { return name.substr(0, name.find(" (")); }

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
                           const std::optional<std::string> diagnostics = std::nullopt) const
    {
      if (name.size() > console_name_width) throw std::runtime_error("console_out: log label is too long: " + name);

      const std::ios_base::fmtflags oldflags = std::cout.flags();
      const std::streamsize oldprecision = std::cout.precision();
      std::cout << "[" << std::setw(console_name_width) << std::left << name << "]";
      constexpr auto console_time_precision = std::numeric_limits<double>::max_digits10;
      std::cout << " t: " << std::setw(24) << std::left << std::setprecision(console_time_precision)
                << std::scientific << t;
      if (Lambda > 0.0) {
        std::cout << " | k: " << std::setw(24) << std::left << std::setprecision(console_time_precision)
                  << std::scientific
                  << exp(-t) * Lambda;
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
     * @param diagnostics Optional timestepper diagnostics printed as a second line.
     */
    void console_out(const double t, const std::string name, const int verbosity_level,
                     const TimesteppingDiagnostics *diagnostics = nullptr) const
    {
      if (verbosity < verbosity_level) return;

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
        print_console_out(name, t, milliseconds, formatted_diagnostics);
        if (diagnostics != nullptr && !diagnostics->empty())
          last_printed_diagnostics_by_category[category] = *diagnostics;
        return;
      }

      auto &stats = console_out_stats_by_category[category];
      stats.add(t, milliseconds, diagnostics);
      if (milliseconds - stats.first_ms < 1000) return;

      const auto previous_it = last_printed_diagnostics_by_category.find(category);
      const auto formatted_diagnostics =
          stats.has_diagnostics
              ? std::optional<std::string>(format_diagnostics(
                    stats.diagnostics,
                    previous_it != last_printed_diagnostics_by_category.end() ? &previous_it->second : nullptr))
              : std::nullopt;
      print_console_out(category, stats.latest_t, stats.last_ms, formatted_diagnostics);
      if (stats.has_diagnostics) last_printed_diagnostics_by_category[category] = stats.diagnostics;
      stats = {};
    }
  };
} // namespace DiFfRG
