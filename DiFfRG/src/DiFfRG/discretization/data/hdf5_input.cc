// DiFfRG
#include <DiFfRG/discretization/data/hdf5_input.hh>

namespace DiFfRG
{
  HDF5Input::HDF5Input(const std::string file_name)
      : file_name(has_suffix(file_name, ".h5") ? file_name : file_name + ".h5")
  {
#ifdef H5CPP
    path = this->file_name;

    if (!std::filesystem::exists(path))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not exist.");

    h5_file = DiFfRG::hdf5::File::open(path.string(), DiFfRG::hdf5::Access::ReadWrite);

    auto root = h5_file.root();
    if (!root.has_group("scalars"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'scalars' group.");
    if (!root.has_group("maps"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'maps' group.");
    if (!root.has_group("coordinates"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'coordinates' group.");

    scalars = root.open_group("scalars");
    maps = root.open_group("maps");
    coords = root.open_group("coordinates");
#endif
  }

#ifdef H5CPP
  DiFfRG::hdf5::File &HDF5Input::get_file() { return h5_file; }
#endif
} // namespace DiFfRG
