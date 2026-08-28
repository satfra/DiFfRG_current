#pragma once

#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/data/csv_output.hh>
#include <DiFfRG/discretization/data/diagnostic_port.hh>
#include <DiFfRG/discretization/data/fe_output.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>
#include <DiFfRG/discretization/data/hdf5_writer.hh>
#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/output_settings.hh>
#include <DiFfRG/discretization/data/output_timings.hh>

#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace DiFfRG
{
  using namespace dealii;

  namespace internal
  {
    template <typename...> inline constexpr bool output_frame_api_removed = false;
  }

  template <uint dim, typename VectorType> class OutputSession;

  /**
   * Scoped collection surface for one complete output event.
   *
   * Frames cannot be copied, stored, or committed independently. OutputSession
   * creates one inside write_frame() and validates the complete contribution
   * before flushing the frame sinks. A contribution or flush error makes the
   * session fail-stop so partially staged state can never leak into a later frame.
   */
  template <uint dim, typename VectorType> class OutputFrame
  {
  public:
    class FieldCollector
    {
    public:
      void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::string &name)
      {
        sink->attach(dof_handler, solution, name);
      }
      void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::vector<std::string> &names)
      {
        sink->attach(dof_handler, solution, names);
      }

    private:
      explicit FieldCollector(FEOutput<dim, VectorType> &sink) : sink(&sink) {}
      FEOutput<dim, VectorType> *sink;
      friend class OutputFrame;
    };

    class TableRecord
    {
    public:
      void value(const std::string &name, double value) { sink->value(name, value); }
      void set_Lambda(double Lambda) { sink->set_Lambda(Lambda); }

    private:
      explicit TableRecord(CsvOutput &sink) : sink(&sink) {}
      CsvOutput *sink;
      friend class OutputFrame;
    };

    class Hdf5Record
    {
    public:
      template <typename... Args> void scalar(Args &&...args) { sink->scalar(std::forward<Args>(args)...); }
      template <typename... Args> void map(Args &&...args) { sink->map(std::forward<Args>(args)...); }

    private:
      explicit Hdf5Record(HDF5Output &sink) : sink(&sink) {}
      HDF5Output *sink;
      friend class OutputFrame;
    };

    OutputFrame(const OutputFrame &) = delete;
    OutputFrame &operator=(const OutputFrame &) = delete;
    OutputFrame(OutputFrame &&) = delete;
    OutputFrame &operator=(OutputFrame &&) = delete;

    FieldCollector fields() { return FieldCollector(session.fe_out); }
    TableRecord table(const std::string &name) { return TableRecord(session.csv(name)); }
    Hdf5Record hdf5(const std::string &name) { return Hdf5Record(session.hdf5(name)); }
    Hdf5Record hdf5() { return Hdf5Record(session.hdf5()); }
    DiagnosticPort diagnostics() const { return session.diagnostic_port(); }

    template <typename Name>
    [[deprecated("OutputFrame::csv(name) was removed. Use auto table = output.table(name) instead")]] CsvOutput &
    csv(Name &&name)
    {
      static_assert(internal::output_frame_api_removed<Name>,
                    "OutputFrame::csv(name) was removed. Use auto table = output.table(name) instead");
      return session.csv(std::forward<Name>(name));
    }

    template <typename Lambda>
    [[deprecated("OutputFrame::set_Lambda(value) was removed. Call set_Lambda(value) on the TableRecord returned by "
                 "output.table(name) instead")]] void
    set_Lambda(Lambda &&)
    {
      static_assert(internal::output_frame_api_removed<Lambda>,
                    "OutputFrame::set_Lambda(value) was removed. Call set_Lambda(value) on the TableRecord returned "
                    "by output.table(name) instead");
    }

    template <typename LegacyVectorType = VectorType>
    [[deprecated("OutputFrame::fe_output() was removed. Use auto fields = output.fields(); fields.attach(...) "
                 "instead")]] FEOutput<dim, VectorType> &
    fe_output()
    {
      static_assert(internal::output_frame_api_removed<LegacyVectorType>,
                    "OutputFrame::fe_output() was removed. Use auto fields = output.fields(); fields.attach(...) "
                    "instead");
      return session.fe_out;
    }

    void attach_eom_potential(EoMResult<dim, typename VectorType::value_type> result)
    {
      session.attach_eom_potential(std::move(result));
    }

    void attach_raw_potential(ReconstructedRawPotential<dim, typename VectorType::value_type> potential)
    {
      session.attach_raw_potential(std::move(potential));
    }

    void dump_table(const std::string &name, const std::vector<std::vector<double>> &values, bool append = false,
                    const std::vector<std::string> &header = {})
    {
      session.dump_to_csv(name, values, append, header);
    }

    const std::string &run_name() const noexcept { return session.output_name; }

    void register_readout(const std::string &id)
    {
      if (id.empty()) throw std::invalid_argument("OutputFrame::register_readout: id must not be empty.");
      if (!readout_ids.insert(id).second)
        throw std::invalid_argument("OutputFrame::register_readout: duplicate id '" + id + "'.");
    }

  private:
    explicit OutputFrame(OutputSession<dim, VectorType> &session) : session(session) {}
    OutputSession<dim, VectorType> &session;
    std::set<std::string> readout_ids;
    friend class OutputSession<dim, VectorType>;
  };

  /**
   * Run-owned output coordinator.
   *
   * Applications construct one session and inject it into the timestepper. All
   * frame submissions are serialized; VTK writing remains bounded and
   * asynchronous inside FEOutput. No process-global output state is used.
   */
  template <uint dim, typename VectorType> class OutputSession
  {
  public:
    OutputSession() : OutputSession(OutputPath::temporary(), Config::OutputSettings{}) {}
    explicit OutputSession(const ConfigTree &config) : OutputSession(path_from(config), Config::OutputSettings(config))
    {
    }
    explicit OutputSession(OutputPath path, Config::OutputSettings settings = {});
    OutputSession(OutputPath path, const ConfigTree &config)
        : OutputSession(std::move(path), Config::OutputSettings(config))
    {
    }
    ~OutputSession() noexcept;

    OutputSession(const OutputSession &) = delete;
    OutputSession &operator=(const OutputSession &) = delete;
    OutputSession(OutputSession &&) = delete;
    OutputSession &operator=(OutputSession &&) = delete;

    template <typename Contributor> void write_frame(const double time, Contributor &&contributor)
    {
      std::scoped_lock lock(submission_mutex);
      rethrow_deferred_error();
      if (terminal_error) std::rethrow_exception(terminal_error);
      if (finished) throw std::logic_error("OutputSession::write_frame: session has already been finished.");
      if (!active) return;

      try {
        current_frame = FrameTimings{};
        ScopedTimer total_timer(current_frame.total);
        {
          const auto solves_before = internal::potential_solve_stats();
          ScopedTimer contributor_timer(current_frame.contributor);
          OutputFrame<dim, VectorType> frame(*this);
          std::forward<Contributor>(contributor)(frame);
          contributor_timer.stop();
          const auto &solves_after = internal::potential_solve_stats();
          current_frame.potential_solve_count += solves_after.solves - solves_before.solves;
          current_frame.potential_solves += solves_after.seconds - solves_before.seconds;
        }
        flush_frame(time);
        total_timer.stop();
        record_frame(time);
      } catch (...) {
        terminal_error = std::current_exception();
        throw;
      }
    }

    DiagnosticPort diagnostic_port() const { return diagnostics_port; }
    ReportPort report_port() const { return run_reporter.port(); }

    /** Join pending writers and surface their first error without closing the session. */
    void drain();

    /** Drain pending writers and permanently close the session. */
    void finish();
    void set_Lambda(double Lambda);
    const std::string &run_name() const noexcept { return output_name; }
    const OutputPath &path() const noexcept { return output_path; }
    bool is_writer_rank() const noexcept { return active; }

  private:
    static OutputPath path_from(const ConfigTree &config)
    {
      if (config.contains("/output/folder")) return OutputPath(config);
      const auto name = config.get_string("/output/name", "output");
      return OutputPath::temporary(TemporaryRetention::remove_on_destruction, name,
                                   config.get_string("/output/field_directory", name));
    }

    FEOutput<dim, VectorType> &fe_output() { return fe_out; }
    void attach_raw_potential(ReconstructedRawPotential<dim, typename VectorType::value_type> potential);
    void attach_eom_potential(EoMResult<dim, typename VectorType::value_type> result);
    CsvOutput &csv(const std::string &name);
    HDF5Output &hdf5(const std::string &name);
    HDF5Output &hdf5();
    void flush_frame(double time);
    /** Fold `current_frame` into the run totals and, at verbosity >= 3, report it. */
    void record_frame(double time);
    void dump_to_csv(const std::string &name, const std::vector<std::vector<double>> &values, bool append,
                     const std::vector<std::string> &header);
    void rethrow_deferred_error();

    OutputPath output_path;
    Config::OutputSettings settings;
    const std::string top_folder;
    const std::string output_name;
    const std::string output_folder;
    double Lambda = -1.;
    bool active = true;
    bool finished = false;
    std::exception_ptr deferred_error;
    std::exception_ptr terminal_error;
    mutable std::mutex submission_mutex;

    RunReporter run_reporter;
    /** Declared before every sink that submits to it, so it is destroyed last. finish() joins
     * it explicitly before h5_files goes away, because ~HDF5Output closes the file and that
     * must not race the worker. */
    HDF5FrameWriter hdf5_writer;
    FEOutput<dim, VectorType> fe_out;
    std::optional<ReconstructedRawPotential<dim, typename VectorType::value_type>> pending_raw_potential;
    FEOutput<dim, dealii::Vector<typename VectorType::value_type>> potential_fe_out;
    std::vector<ReconstructedEoMPotential<dim, typename VectorType::value_type>> pending_eom_potentials;
    FEOutput<dim, dealii::Vector<typename VectorType::value_type>> eom_potential_fe_out;
    std::map<std::string, CsvOutput> csv_files;

    bool use_hdf5;
    const std::string filename_h5;
    std::map<std::string, HDF5Output> h5_files;
    DiagnosticPort diagnostics_port;

    /** Cost split of the frame currently being written; folded into `timings` by flush_frame. */
    FrameTimings current_frame;
    OutputTimings timings;

    friend class OutputFrame<dim, VectorType>;
  };
} // namespace DiFfRG
