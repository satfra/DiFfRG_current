// DiFfRG
#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/hdf5_output.hh>

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace DiFfRG
{
  HDF5Output::HDF5Output(const std::string top_folder, const std::string _output_name, const JSONValue &json)
      : json(json), top_folder(make_folder(top_folder)),
        output_name(has_suffix(_output_name, ".h5") ? _output_name : _output_name + ".h5")
  {
#ifdef H5CPP
    create_folder(this->top_folder);
    std::filesystem::path path = this->top_folder + this->output_name;
    create_folder(path.parent_path().string());

    h5_file = hdf5::file::create(path, hdf5::file::AccessFlags::Truncate);
    auto root = h5_file.root();
    scalars = root.create_group("scalars");
    maps = root.create_group("maps");
    coords = root.create_group("coordinates");
#endif
  }

  void HDF5Output::flush(const double time)
  {
#ifdef H5CPP
    if (written_scalars.size() > 0) scalar<double>("t", time);
    if (initial_scalars.size() == 0) initial_scalars = written_scalars;

    // ensure that initial_scalars have the same content as written_scalars
    for (const auto &name : initial_scalars) {
      if (std::find(written_scalars.begin(), written_scalars.end(), name) == written_scalars.end())
        throw std::runtime_error("HDF5Output::flush: The scalar '" + name +
                                 "' has not been written before the flush at t = " + std::to_string(time) + ".");
    }

    for (const auto &name : written_maps) {
      // the series number got incremented in the call, thus the -1
      const size_t series_number = map_series_numbers[name] - 1;
      auto n_group = maps.get_group(name);
      auto group = n_group.get_group(int_to_string(series_number, 6));
      group.attributes.create_from<double>("time", time);
      group.attributes.create_from<int>("series_number", series_number);
    }

    written_scalars.clear();
    written_maps.clear();
#endif
  }

  // Maps are not implemented yet.

#ifdef H5CPP
  hdf5::file::File &HDF5Output::get_file() { return h5_file; }
#endif
} // namespace DiFfRG