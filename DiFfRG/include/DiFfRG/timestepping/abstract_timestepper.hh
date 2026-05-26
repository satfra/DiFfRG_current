#pragma once

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/discretization/mesh/no_adaptivity.hh>
#include <cmath>
#include <optional>
#include <unordered_map>

namespace DiFfRG
{
  struct TimesteppingDiagnostics {
    bool has_ida = false;
    long int ida_steps = 0;
    long int ida_error_test_failures = 0;
    long int ida_nonlinear_convergence_failures = 0;
    long int ida_step_solve_failures = 0;
    long int ida_residual_evaluations = 0;
    double ida_last_step_size = 0.;
    double ida_current_step_size = 0.;

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
   * - /timestepping/explicit/dt: The timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/minimal_dt: The minimal timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/maximal_dt: The maximal timestep size for an explicit timestepping algorithm.
   * - /timestepping/explicit/abs_tol: The absolute tolerance for an explicit timestepping algorithm.
   * - /timestepping/explicit/rel_tol: The relative tolerance for an explicit timestepping algorithm.
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

      expl.dt = json.get_double("/timestepping/explicit/dt");
      expl.minimal_dt = json.get_double("/timestepping/explicit/minimal_dt");
      expl.maximal_dt = json.get_double("/timestepping/explicit/maximal_dt");
      expl.abs_tol = json.get_double("/timestepping/explicit/abs_tol");
      expl.rel_tol = json.get_double("/timestepping/explicit/rel_tol");

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
    } impl;

    struct ExplicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
    } expl;

    static constexpr size_t console_name_width = 21;

    struct ConsoleOutStats {
      size_t first_ms = 0;
      size_t last_ms = 0;
      double latest_t = 0.;
      bool has_entry = false;
      TimesteppingDiagnostics diagnostics;
      bool has_diagnostics = false;

      void add(const double t, const size_t milliseconds, const TimesteppingDiagnostics *latest_diagnostics)
      {
        if (!has_entry) first_ms = milliseconds;
        has_entry = true;
        last_ms = milliseconds;
        latest_t = t;

        if (latest_diagnostics != nullptr && !latest_diagnostics->empty()) {
          diagnostics = *latest_diagnostics;
          has_diagnostics = true;
        }
      }
    };

    mutable std::unordered_map<std::string, ConsoleOutStats> console_out_stats_by_category;
    mutable std::unordered_map<std::string, TimesteppingDiagnostics> last_printed_diagnostics_by_category;

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

    static std::string format_diagnostics(const TimesteppingDiagnostics &latest,
                                          const TimesteppingDiagnostics *previous)
    {
      const TimesteppingDiagnostics zero;
      const auto &prev = previous != nullptr ? *previous : zero;

      std::stringstream stream;
      stream << "accepted steps " << format_diagnostics_delta(latest.ida_steps, prev.ida_steps);
      stream << ", precision rejects "
             << format_diagnostics_delta(latest.ida_error_test_failures, prev.ida_error_test_failures);
      stream << ", nonlinear failures " << format_diagnostics_delta(latest.ida_nonlinear_convergence_failures,
                                                                     prev.ida_nonlinear_convergence_failures);
      stream << ", accumulated step failures "
             << format_diagnostics_delta(latest.ida_step_solve_failures, prev.ida_step_solve_failures);
      stream << ", NaN/Inf callbacks " << format_diagnostics_delta(latest.nonfinite_failures(), prev.nonfinite_failures());
      return stream.str();
    }

    void print_console_out(const std::string &name, const double t, const size_t milliseconds,
                           const std::optional<std::string> diagnostics = std::nullopt) const
    {
      if (name.size() > console_name_width) throw std::runtime_error("console_out: log label is too long: " + name);

      const std::ios_base::fmtflags oldflags = std::cout.flags();
      const std::streamsize oldprecision = std::cout.precision();
      std::cout << "[" << std::setw(console_name_width) << std::left << name << "]";
      std::cout << " t: " << std::setw(10) << std::left << std::setprecision(4) << std::scientific << t;
      if (Lambda > 0.0) {
        std::cout << " | k: " << std::setw(10) << std::left << std::setprecision(4) << std::scientific
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
      if (verbosity >= 4) {
        const auto previous_it = last_printed_diagnostics_by_category.find(category);
        const auto formatted_diagnostics =
            diagnostics != nullptr && !diagnostics->empty()
                ? std::optional<std::string>(
                      format_diagnostics(*diagnostics, previous_it != last_printed_diagnostics_by_category.end()
                                                           ? &previous_it->second
                                                           : nullptr))
                : std::nullopt;
        print_console_out(name, t, milliseconds, formatted_diagnostics);
        if (diagnostics != nullptr && !diagnostics->empty()) last_printed_diagnostics_by_category[category] = *diagnostics;
        return;
      }

      auto &stats = console_out_stats_by_category[category];
      stats.add(t, milliseconds, diagnostics);
      if (milliseconds - stats.first_ms < 1000) return;

      const auto previous_it = last_printed_diagnostics_by_category.find(category);
      const auto formatted_diagnostics =
          stats.has_diagnostics
              ? std::optional<std::string>(
                    format_diagnostics(stats.diagnostics, previous_it != last_printed_diagnostics_by_category.end()
                                                              ? &previous_it->second
                                                              : nullptr))
              : std::nullopt;
      print_console_out(category, stats.latest_t, stats.last_ms, formatted_diagnostics);
      if (stats.has_diagnostics) last_printed_diagnostics_by_category[category] = stats.diagnostics;
      stats = {};
    }
  };
} // namespace DiFfRG
