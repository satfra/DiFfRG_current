#pragma once

// DiFfRG
#include <DiFfRG/common/json.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/hdf5.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <filesystem>
#include <list>

namespace DiFfRG
{
  /**
   * @brief A class to output data to a CSV file.
   *
   * In every time step, the user can add values to the output, which will be written to the file when flush is called.
   * Every timestep has to contain the same values, but the order in which they are added can change.
   */
  class HDF5Output
  {
  public:
    /**
     * @brief Construct a new Csv Output object
     *
     * @param top_folder The top folder to store the output in.
     * @param output_name The name of the output file.
     * @param json The JSON object containing the parameters.
     */
    HDF5Output(const std::string top_folder, const std::string output_name, const JSONValue &json);

    void write_series_record(DiFfRG::hdf5::Group &group, const int series_number)
    {
      const std::string value = std::to_string(series_number);

      if (!group.has_dataset("series_numbers")) {
        // chunked layout so we can append later
        auto space = DiFfRG::hdf5::Dataspace::simple_unlimited({1});
        auto type = DiFfRG::hdf5::type_of<std::string>();
        auto data_set = group.create_chunked_dataset("series_numbers", type, space, {128});
        data_set.write_at(0, value);
      } else {
        auto data_set = group.open_dataset("series_numbers");
        const std::size_t cur_size = static_cast<std::size_t>(data_set.dataspace().size());
        data_set.resize({cur_size + 1});
        data_set.write_at(static_cast<hsize_t>(cur_size), value);
      }
    }

    /**
     * @brief Add a value to the output.
     *
     * @param name The name of the value.
     * @param value The value to add.
     */
    template <typename T> void scalar(const std::string &name, const T value)
    {
#ifndef H5CPP
      throw std::runtime_error("HDF5Output::scalar: HDF5 support is not enabled. Please compile with H5CPP support.");
#else
      if (initial_scalars.size() > 0 &&
          std::find(initial_scalars.begin(), initial_scalars.end(), name) == initial_scalars.end())
        throw std::runtime_error("HDF5Output::scalar: The scalar '" + name + "' has not been registered before!");

      open_file();

      if (!scalars.has_dataset(name)) {
        auto space = DiFfRG::hdf5::Dataspace::simple_unlimited({1});
        auto type = DiFfRG::hdf5::type_of<std::decay_t<T>>();
        auto data_set = scalars.create_chunked_dataset(name, type, space, {128});
        data_set.write_at(0, value);
        written_scalars.push_back(name);
      } else {
        auto data_set = scalars.open_dataset(name);
        auto type = DiFfRG::hdf5::type_of<std::decay_t<T>>();
        if (data_set.datatype() != type)
          throw std::runtime_error(
              "HDF5Output::scalar: The type of the value does not match the type of the dataset '" + name +
              "' in the file '" + output_name + "'.");

        if (std::find(written_scalars.begin(), written_scalars.end(), name) == written_scalars.end()) {
          written_scalars.push_back(name);
        } else {
          throw std::runtime_error("HDF5Output::scalar: The scalar '" + name +
                                   "' has already been written to the file '" + output_name + "'.");
        }

        const std::size_t cur_size = static_cast<std::size_t>(data_set.dataspace().size());
        data_set.resize({cur_size + 1});
        data_set.write_at(static_cast<hsize_t>(cur_size), value);
      }
#endif
    }
    void scalar(const std::string &name, const char *value) { scalar<std::string>(name, std::string(value)); }

    template <typename COORD>
      requires is_coordinates<COORD>
    void map(const std::string &name, const COORD &coordinates)
    {
#ifndef H5CPP
      throw std::runtime_error("HDF5Output::map: HDF5 support is not enabled. Please compile with H5CPP support.");
#else
      if (coords.has_dataset(name)) {
        throw std::runtime_error("HDF5Output::map: The coordinates '" + name +
                                 "' have already been written to the file '" + output_name + "'.");
      }

      open_file();

      const auto grid_data = make_grid(coordinates);

      DiFfRG::hdf5::Dims dims;
      for (const auto &dim : coordinates.sizes())
        dims.push_back(dim);
      auto space = DiFfRG::hdf5::Dataspace::simple(dims);
      using coord_type = std::decay_t<decltype(grid_data[0])>;
      auto c_type = DiFfRG::hdf5::type_of<coord_type>();
      auto dataset = coords.create_dataset(name, c_type, space);

      dataset.write(grid_data);

      coord_identifiers[name] = coordinates.to_string();
#endif
    }

    template <typename COORD, typename T>
      requires is_coordinates<COORD>
    void map(const std::string &name, const COORD &coordinates, T *data)
    {
#ifndef H5CPP
      throw std::runtime_error("HDF5Output::map: HDF5 support is not enabled. Please compile with H5CPP support.");
#else

      open_file();

      const std::string coord_name = coordinates.to_string();
      if (coord_identifiers.find(coord_name) == coord_identifiers.end()) map(coord_name, coordinates);

      using value_type = std::decay_t<T>;

      DiFfRG::hdf5::Group n_group;
      if (!maps.has_group(name)) {
        n_group = maps.create_group(name);
      } else {
        n_group = maps.open_group(name);
      }

      if (n_group.has_group(std::to_string(map_series_numbers[name])))
        throw std::runtime_error("HDF5Output::map: The map '" + name + "' has already been written to the file '" +
                                 output_name + "' for series number " + std::to_string(map_series_numbers[name]) + ".");

      auto group = n_group.create_group(std::to_string(map_series_numbers[name]));
      write_series_record(group, map_series_numbers[name]);

      DiFfRG::hdf5::Dims dims;
      for (const auto &dim : coordinates.sizes())
        dims.push_back(dim);
      auto space = DiFfRG::hdf5::Dataspace::simple(dims);
      auto d_type = DiFfRG::hdf5::type_of<value_type>();
      auto dataset = group.create_dataset("data", d_type, space);

      dataset.write(data, coordinates.size());

      group.create_soft_link("coordinates", "/coordinates/" + coord_name);

      written_maps.push_back(name);
      map_series_numbers[name]++;
#endif
    }

    template <typename INTERP>
      requires is_interpolator<INTERP>
    void map(const std::string &name, const INTERP &_interpolator)
    {
      map(name, _interpolator.get_coordinates().to_string(), _interpolator);
    }

    template <typename INTERP>
      requires is_interpolator<INTERP>
    void map(const std::string &name, const std::string &coord_name, const INTERP &_interpolator)
    {
#ifndef H5CPP
      throw std::runtime_error("HDF5Output::map: HDF5 support is not enabled. Please compile with H5CPP support.");
#else

      open_file();

      using value_type = typename std::decay_t<INTERP>::value_type;

      const auto &interpolator = _interpolator.template get_on<CPU_memory>();

      check_coordinates(coord_name, interpolator.get_coordinates());

      DiFfRG::hdf5::Group n_group;
      if (!maps.has_group(name)) {
        n_group = maps.create_group(name);
      } else {
        n_group = maps.open_group(name);
      }

      if (n_group.has_group(std::to_string(map_series_numbers[name])))
        throw std::runtime_error("HDF5Output::map: The map '" + name + "' has already been written to the file '" +
                                 output_name + "' for series number " + std::to_string(map_series_numbers[name]) + ".");

      auto group = n_group.create_group(std::to_string(map_series_numbers[name]));
      write_series_record(group, map_series_numbers[name]);

      DiFfRG::hdf5::Dims dims;
      const auto coordinates = interpolator.get_coordinates();
      for (const auto &dim : coordinates.sizes())
        dims.push_back(dim);
      auto space = DiFfRG::hdf5::Dataspace::simple(dims);
      auto d_type = DiFfRG::hdf5::type_of<value_type>();
      auto dataset = group.create_dataset("data", d_type, space);

      std::vector<value_type> data(interpolator.get_coordinates().size());
      for (size_t i = 0; i < data.size(); ++i) {
        const auto lcoord = coordinates.forward(coordinates.from_linear_index(i));
        data[i] = device::apply([&](const auto &...x) { return interpolator(x...); }, lcoord);
      }
      dataset.write(data);

      group.create_soft_link("coordinates", "/coordinates/" + coord_name);

      written_maps.push_back(name);
      map_series_numbers[name]++;

#endif
    }

#ifdef H5CPP
    DiFfRG::hdf5::File &get_file();
#endif

    void flush(const double time);

    void open_file();
    void close_file();

  private:
    [[maybe_unused]] const JSONValue &json;
    const std::string top_folder;
    const std::string output_name;

    bool opened;

    std::list<std::string> written_scalars;
    std::list<std::string> written_maps;
    std::map<std::string, size_t> map_series_numbers;
    std::map<std::string, std::string> coord_identifiers;

    std::list<std::string> initial_scalars;

    template <typename COORD> void check_coordinates(const std::string &coord_name, const COORD &coordinates)
    {
      if (!coords.has_dataset(coord_name)) {
        map(coord_name, coordinates);
        return;
      }

      if (coordinates.to_string() != coord_identifiers[coord_name])
        throw std::runtime_error("HDF5Output::map: The coordinates '" + coord_name +
                                 "' do not match the interpolator's coordinates.");
    }

#ifdef H5CPP
    DiFfRG::hdf5::File h5_file;
    DiFfRG::hdf5::Group scalars;
    DiFfRG::hdf5::Group maps;
    DiFfRG::hdf5::Group coords;
#endif

    std::filesystem::path path;
  };
} // namespace DiFfRG
