#include <DiFfRG/discretization/data/output_session.hh>

#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/utils.hh>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace DiFfRG
{
  template <uint dim, typename VectorType>
  OutputSession<dim, VectorType>::OutputSession(const OutputPath &path, Config::OutputSettings settings)
      : output_path(path), settings(std::move(settings)), top_folder(make_folder(path.root().string())),
        output_name(path.run_name()), output_folder(make_folder(path.field_directory().generic_string())),
        active(MPI::rank(MPI_COMM_WORLD) == 0), run_logger(path, this->settings, active),
        fe_out(this->top_folder, this->output_name, this->output_folder, this->settings, active),
        potential_fe_out(this->top_folder, this->output_name + "_potential", this->output_folder, this->settings,
                         active),
        use_hdf5(this->settings.write_hdf5), filename_h5(this->output_name + ".h5")
  {
    set_Lambda(this->settings.Lambda);
    diagnostics_port = DiagnosticPort::create(path, Lambda, active);
    if (active) diagnostics_port.write_text(this->output_name + ".log.json", this->settings.configuration_log);

    if (active && use_hdf5) {
      h5_files.emplace(filename_h5, HDF5Output(this->top_folder, filename_h5, this->settings.configuration_json));
      if constexpr (dim > 0) {
        auto root_group = h5_files.at(filename_h5).get_file().root();
        root_group.create_group("FE");
        fe_out.set_hdf5_output(&h5_files.at(filename_h5));
      }
    }
  }

  template <uint dim, typename VectorType> OutputSession<dim, VectorType>::~OutputSession() noexcept
  {
    try {
      finish();
    } catch (...) {
      deferred_error = std::current_exception();
    }
  }

  template <uint dim, typename VectorType>
  void OutputSession<dim, VectorType>::attach_eom_potential(EoMResult<dim, typename VectorType::value_type> result)
  {
    if constexpr (dim > 0) {
      if (!result.potential.has_value()) return;
      auto &potential = result.potential.value();
      const std::string name =
          pending_eom_potentials.empty() ? "potential" : "potential_" + std::to_string(pending_eom_potentials.size());
      potential_fe_out.attach(*potential.dof_handler, potential.values, name);
      pending_eom_potentials.push_back(std::move(potential));
    }
  }

  template <uint dim, typename VectorType> CsvOutput &OutputSession<dim, VectorType>::csv(const std::string &name)
  {
    const auto found = csv_files.find(name);
    if (found != csv_files.end()) return found->second;

    const std::string filename = output_name + "_" + OutputPath::checked_relative(name, "table name").generic_string();
    auto [it, inserted] = csv_files.emplace(name, CsvOutput(top_folder, filename));
    if (inserted) it->second.set_Lambda(Lambda);
    return it->second;
  }

  template <uint dim, typename VectorType> HDF5Output &OutputSession<dim, VectorType>::hdf5(const std::string &name)
  {
    const auto found = h5_files.find(name);
    if (found != h5_files.end()) return found->second;

    const std::string filename = output_name + "_" + OutputPath::checked_relative(name, "HDF5 name").generic_string();
    return h5_files.emplace(name, HDF5Output(top_folder, filename, settings.configuration_json)).first->second;
  }

  template <uint dim, typename VectorType> HDF5Output &OutputSession<dim, VectorType>::hdf5()
  {
    if (!use_hdf5) throw std::logic_error("OutputSession::hdf5: HDF5 output is disabled.");
    return h5_files.at(filename_h5);
  }

  template <uint dim, typename VectorType> void OutputSession<dim, VectorType>::flush_frame(const double time)
  {
    for (const auto &[name, csv] : csv_files)
      csv.validate_frame();

    if constexpr (dim > 0) {
      fe_out.flush(time);
      if (!pending_eom_potentials.empty()) {
        potential_fe_out.flush(time);
        pending_eom_potentials.clear();
      }
    }
    for (auto &[name, csv] : csv_files)
      csv.flush(time);
    for (auto &[name, hdf] : h5_files)
      hdf.flush(time);
  }

  template <uint dim, typename VectorType>
  void OutputSession<dim, VectorType>::dump_to_csv(const std::string &name,
                                                   const std::vector<std::vector<double>> &values, const bool append,
                                                   const std::vector<std::string> &header)
  {
    const auto path = output_path.resolve(OutputPath::checked_relative(name, "table path"));
    create_folder(path.parent_path().string());
    std::ofstream stream(path, append ? std::ofstream::app : std::ofstream::trunc);
    if (!stream) throw std::runtime_error("OutputSession::dump_table: could not open '" + path.string() + "'.");
    stream << std::scientific;
    if (!append && !header.empty()) {
      for (uint i = 0; i < header.size(); ++i) {
        stream << strip_name(header[i]);
        if (i + 1 != header.size()) stream << ',';
      }
      stream << '\n';
    }
    for (const auto &row : values) {
      for (uint i = 0; i < row.size(); ++i) {
        stream << row[i];
        if (i + 1 != row.size()) stream << ',';
      }
      stream << '\n';
    }
    if (!stream) throw std::runtime_error("OutputSession::dump_table: write failed for '" + path.string() + "'.");
  }

  template <uint dim, typename VectorType> void OutputSession<dim, VectorType>::set_Lambda(const double Lambda)
  {
    this->Lambda = Lambda;
    for (auto &[name, csv] : csv_files)
      csv.set_Lambda(Lambda);
  }

  template <uint dim, typename VectorType> void OutputSession<dim, VectorType>::rethrow_deferred_error()
  {
    if (!deferred_error) return;
    auto error = deferred_error;
    deferred_error = nullptr;
    std::rethrow_exception(error);
  }

  template <uint dim, typename VectorType> void OutputSession<dim, VectorType>::drain()
  {
    std::scoped_lock lock(submission_mutex);
    rethrow_deferred_error();
    if (terminal_error) std::rethrow_exception(terminal_error);
    if (finished || !active) return;

    std::exception_ptr drain_error;
    try {
      fe_out.drain();
    } catch (...) {
      drain_error = std::current_exception();
    }
    try {
      potential_fe_out.drain();
    } catch (...) {
      if (!drain_error) drain_error = std::current_exception();
    }
    if (drain_error) {
      terminal_error = drain_error;
      std::rethrow_exception(drain_error);
    }
  }

  template <uint dim, typename VectorType> void OutputSession<dim, VectorType>::finish()
  {
    std::scoped_lock lock(submission_mutex);
    if (finished) {
      rethrow_deferred_error();
      if (terminal_error) std::rethrow_exception(terminal_error);
      return;
    }
    finished = true;
    if (active) {
      try {
        fe_out.finish();
      } catch (...) {
        if (!terminal_error) terminal_error = std::current_exception();
      }
      try {
        potential_fe_out.finish();
      } catch (...) {
        if (!terminal_error) terminal_error = std::current_exception();
      }
    }
    rethrow_deferred_error();
    if (terminal_error) std::rethrow_exception(terminal_error);
  }

  template class OutputSession<0, dealii::Vector<double>>;
  template class OutputSession<1, dealii::Vector<double>>;
  template class OutputSession<2, dealii::Vector<double>>;
  template class OutputSession<3, dealii::Vector<double>>;
  template class OutputSession<1, dealii::BlockVector<double>>;
  template class OutputSession<2, dealii::BlockVector<double>>;
  template class OutputSession<3, dealii::BlockVector<double>>;
} // namespace DiFfRG
