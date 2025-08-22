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
#endif
  }

  template <typename T> void HDF5Output::scalar(const std::string &name, const T value)
  {
#ifndef H5CPP
    throw std::runtime_error("HDF5Output::scalar: HDF5 support is not enabled. Please compile with H5CPP support.");
#else
    if (initial_scalars.size() > 0 &&
        std::find(initial_scalars.begin(), initial_scalars.end(), name) == initial_scalars.end())
      throw std::runtime_error("HDF5Output::scalar: The scalar '" + name + "' has not been registered before!");

    if (!scalars.has_dataset(name)) {
      hdf5::property::LinkCreationList lcpl;
      hdf5::property::DatasetCreationList dcpl;

      // in order to append data we have to use a chunked layout of the dataset
      dcpl.layout(hdf5::property::DatasetLayout::Chunked);
      dcpl.chunk({128});

      hdf5::dataspace::Simple space({1}, {hdf5::dataspace::Simple::unlimited});
      auto type = hdf5::datatype::create<std::decay_t<T>>();

      auto data_set = scalars.create_dataset(name, type, space, dcpl, lcpl);
      data_set.write(value); // write data
      written_scalars.push_back(name);
    } else {
      auto data_set = scalars.get_dataset(name);

      auto type = hdf5::datatype::create<std::decay_t<T>>();
      // Check if the type matches the dataset type
      if (data_set.datatype() != type)
        throw std::runtime_error("HDF5Output::scalar: The type of the value does not match the type of the dataset '" +
                                 name + "' in the file '" + output_name + "'.");

      if (std::find(written_scalars.begin(), written_scalars.end(), name) == written_scalars.end()) {
        written_scalars.push_back(name);
      } else {
        // If the scalar has already been written, we do not write it again.
        throw std::runtime_error("HDF5Output::scalar: The scalar '" + name +
                                 "' has already been written to the file '" + output_name + "'.");
      }

      const size_t cur_size = data_set.dataspace().size();
      const size_t sel_start = cur_size;
      data_set.resize({cur_size + 1}); // grow dataset

      hdf5::dataspace::Hyperslab selection{{sel_start}, {1}, {1}, {1}};
      data_set.write(value, selection); // write data
    }
#endif
  }

  template <> void HDF5Output::scalar<char const *>(const std::string &name, const char *value)
  {
    scalar<std::string>(name, std::string(value));
  }
  template void HDF5Output::scalar<std::string>(const std::string &name, const std::string value);

  // explicitly instantiate the scalar method for double, float, and int types
  template void HDF5Output::scalar<double>(const std::string &name, const double value);
  template void HDF5Output::scalar<complex<double>>(const std::string &name, const complex<double> value);
  template void HDF5Output::scalar<float>(const std::string &name, const float value);
  template void HDF5Output::scalar<complex<float>>(const std::string &name, const complex<float> value);
  template void HDF5Output::scalar<unsigned int>(const std::string &name, const unsigned int value);
  template void HDF5Output::scalar<unsigned long>(const std::string &name, const unsigned long value);
  template void HDF5Output::scalar<int>(const std::string &name, const int value);
  template void HDF5Output::scalar<long>(const std::string &name, const long value);
  template void HDF5Output::scalar<autodiff::Real<1, double>>(const std::string &name,
                                                              const autodiff::Real<1, double> value);
  template void HDF5Output::scalar<autodiff::Real<1, float>>(const std::string &name,
                                                             const autodiff::Real<1, float> value);
  template void HDF5Output::scalar<autodiff::Real<2, double>>(const std::string &name,
                                                              const autodiff::Real<2, double> value);
  template void HDF5Output::scalar<autodiff::Real<2, float>>(const std::string &name,
                                                             const autodiff::Real<2, float> value);
  template void HDF5Output::scalar<autodiff::Real<3, double>>(const std::string &name,
                                                              const autodiff::Real<3, double> value);
  template void HDF5Output::scalar<autodiff::Real<3, float>>(const std::string &name,
                                                             const autodiff::Real<3, float> value);

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

    written_scalars.clear();
    written_maps.clear();
#endif
  }

  // Maps are not implemented yet.

#ifdef H5CPP
  hdf5::file::File &HDF5Output::get_file() { return h5_file; }
#endif
} // namespace DiFfRG