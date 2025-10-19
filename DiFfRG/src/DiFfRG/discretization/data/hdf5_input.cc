// DiFfRG
#include <DiFfRG/discretization/data/hdf5_input.hh>

namespace DiFfRG
{
  HDF5Input::HDF5Input(const std::string file_name)
      : file_name(has_suffix(file_name, ".h5") ? file_name : file_name + ".h5")
  {
#ifdef H5CPP
    path = this->file_name;

    // Check if file exists
    if (!std::filesystem::exists(path))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not exist.");

    // Open the file
    h5_file = hdf5::file::open(path, hdf5::file::AccessFlags::ReadWrite);

    auto root = h5_file.root();
    if (!root.has_group("scalars"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'scalars' group.");
    if (!root.has_group("maps"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'maps' group.");
    if (!root.has_group("coordinates"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'coordinates' group.");

    scalars = root.get_group("scalars");
    maps = root.get_group("maps");
    coords = root.get_group("coordinates");
#endif
  }

#ifdef H5CPP
  hdf5::file::File &HDF5Input::get_file() { return h5_file; }
#endif
} // namespace DiFfRG
