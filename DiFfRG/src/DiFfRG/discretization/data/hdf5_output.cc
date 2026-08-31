// DiFfRG
#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>

#include <deal.II/base/utilities.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>

namespace DiFfRG
{
  HDF5Output::HDF5Output(const std::string top_folder, const std::string _output_name,
                         const std::string &configuration_json)
      : top_folder(make_folder(top_folder)),
        output_name(has_suffix(_output_name, ".h5") ? _output_name : _output_name + ".h5"), opened(false)
  {
    create_folder(this->top_folder);
    path = this->top_folder + this->output_name;
    create_folder(path.parent_path().string());

    h5_file = DiFfRG::hdf5::File::open(path.string(), DiFfRG::hdf5::Access::Truncate);

    auto root = h5_file.root();
    root.write_attribute("configuration_json", configuration_json);
    // Created once, here. Frames reopen them through their own HDF5FrameContext, so no handle
    // is retained between frames.
    root.create_group("scalars");
    root.create_group("maps");
    root.create_group("coordinates");
    h5_file.close();
  }

  void HDF5Output::stage_fe_frame(const unsigned int series_number, const double time, const std::string &group_name,
                                  const std::string &fe_output_name, const unsigned int dim, const std::size_t n_nodes,
                                  std::vector<double> node_data,
                                  std::vector<std::pair<std::string, std::vector<double>>> data_sets)
  {
    staged.actions.emplace_back([series_number, time, group_name, fe_output_name, dim, n_nodes,
                                 nodes = std::move(node_data), sets = std::move(data_sets)](HDF5FrameContext &context) {
      auto group = context.group(group_name).create_group(dealii::Utilities::int_to_string(series_number, 6));
      group.write_attribute("time", time);
      group.write_attribute("series_number", static_cast<int>(series_number));
      group.write_attribute("output_name", fe_output_name);

      auto nodes_space = DiFfRG::hdf5::Dataspace::simple({n_nodes, dim});
      group.create_dataset("nodes", DiFfRG::hdf5::type_of<double>(), nodes_space).write(nodes);

      for (const auto &[name, values] : sets) {
        auto space = DiFfRG::hdf5::Dataspace::simple({n_nodes});
        group.create_dataset(name, DiFfRG::hdf5::type_of<double>(), space).write(values);
      }
    });
  }

  void HDF5Output::flush(const double time)
  {
    if (written_scalars.size() > 0) scalar<double>("time", time);
    if (initial_scalars.size() == 0) initial_scalars = written_scalars;

    for (const auto &name : initial_scalars) {
      if (std::find(written_scalars.begin(), written_scalars.end(), name) == written_scalars.end())
        throw std::runtime_error("HDF5Output::flush: The scalar '" + name +
                                 "' has not been written before the flush at t = " + std::to_string(time) + ".");
    }

    // Stamped last, so that every map group they refer to has been created by an earlier action
    // of the same frame.
    for (const auto &[name, series_number] : written_map_series) {
      staged.actions.emplace_back([name, series_number, time](HDF5FrameContext &context) {
        auto parent = context.group("maps").open_group(name);
        auto group = parent.open_group(std::to_string(series_number));
        group.write_attribute("time", time);
        group.write_attribute("series_number", static_cast<int>(series_number));
      });
    }

    written_scalars.clear();
    written_map_series.clear();

    run_staged_frame(time);
  }

  void HDF5Output::run_staged_frame(const double time)
  {
    if (staged.empty()) {
      staged = HDF5Frame{};
      return;
    }

    if (opened) {
      // A setup-time handle is still open. Closing it is an HDF5 call, so the writer must not
      // be inside one at the same time -- HDF5 is not thread safe in this build.
      if (writer != nullptr) writer->drain();
      close_file();
    }

    staged.path = path.string();
    staged.time = time;

    if (writer != nullptr)
      writer->submit(std::move(staged), &frame_timings);
    else
      staged.run(&frame_timings);

    staged = HDF5Frame{};
  }

  void HDF5Output::drain()
  {
    if (writer != nullptr) writer->drain();
  }

  DiFfRG::hdf5::File &HDF5Output::get_file()
  {
    open_file();
    return h5_file;
  }

  void HDF5Output::open_file()
  {
    if (opened) return;
    ScopedTimer open_timer(frame_timings.hdf5_open_close);
    ++frame_timings.hdf5_open_count;
    h5_file = DiFfRG::hdf5::File::open(path.string(), DiFfRG::hdf5::Access::ReadWrite);
    opened = true;
  }

  void HDF5Output::close_file()
  {
    if (!opened) return;
    ScopedTimer close_timer(frame_timings.hdf5_open_close);
    // No child handles are retained here -- see HDF5FrameContext, which caches them for the
    // duration of a frame and releases them before H5Fclose for the usual reason: HDF5
    // reference-counts the file through them, so a live one keeps it open at the OS level.
    h5_file.flush();
    h5_file.close();
    opened = false;
  }

  HDF5Output::~HDF5Output() { close_file(); }
} // namespace DiFfRG
