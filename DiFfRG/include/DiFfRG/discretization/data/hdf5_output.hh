#pragma once

// DiFfRG
#include <DiFfRG/common/json.hh>
#include <DiFfRG/common/kokkos.hh>

#include <autodiff/forward/real/real.hpp>

#ifdef H5CPP
#include <h5cpp/hdf5.hpp>
#endif

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
          throw std::runtime_error(
              "HDF5Output::scalar: The type of the value does not match the type of the dataset '" + name +
              "' in the file '" + output_name + "'.");

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

    void scalar(const std::string &name, const char *value) { scalar<std::string>(name, std::string(value)); }

    template <typename COORD> void map_coords(const std::string &name, const COORD &coordinates)
    {
#ifndef H5CPP
      throw std::runtime_error(
          "HDF5Output::map_coords: HDF5 support is not enabled. Please compile with H5CPP support.");
#else
      if (coords.has_dataset(name)) {
        throw std::runtime_error("HDF5Output::map_coords: The coordinates '" + name +
                                 "' have already been written to the file '" + output_name + "'.");
      }

      const auto grid_data = make_grid(coordinates);

      hdf5::Dimensions dims;
      for (const auto &dim : coordinates.sizes())
        dims.push_back(dim);
      hdf5::dataspace::Simple space(dims);
      using coord_type = std::decay_t<decltype(grid_data[0])>;
      auto c_type = hdf5::datatype::create<coord_type>();
      auto dataset = coords.create_dataset(name, c_type, space);

      dataset.write(grid_data); // write data
      written_coords.push_back(name);
#endif
    }

    template <typename INTERP>
    void map_interp(const std::string &name, const std::string &coord_name, const INTERP &_interpolator)
    {
#ifndef H5CPP
      throw std::runtime_error(
          "HDF5Output::map_interp: HDF5 support is not enabled. Please compile with H5CPP support.");
#else
      if (initial_maps.size() > 0 && std::find(initial_maps.begin(), initial_maps.end(), name) == initial_maps.end())
        throw std::runtime_error("HDF5Output::map_interp: The map '" + name + "' has not been registered before!");

      using value_type = typename std::decay_t<INTERP>::value_type;

      const auto &interpolator = _interpolator.template get_on<CPU_memory>();

      if (!coords.has_dataset(coord_name)) {
        // If the coordinates have not been written yet, we write them now.
        map_coords(coord_name, interpolator.get_coordinates());
      }

      if (!maps.has_group(name)) {
        hdf5::node::Group group = maps.create_group(name);
        hdf5::Dimensions dims;
        const auto coordinates = interpolator.get_coordinates();
        for (const auto &dim : coordinates.sizes())
          dims.push_back(dim);
        hdf5::dataspace::Simple space(dims);
        auto d_type = hdf5::datatype::create<value_type>();
        auto dataset = group.create_dataset("data", d_type, space);

        std::vector<value_type> data(interpolator.get_coordinates().size());
        for (size_t i = 0; i < data.size(); ++i) {
          const auto lcoord = coordinates.forward(coordinates.from_linear_index(i));
          data[i] = device::apply([&](const auto &...x) { return interpolator(x...); }, lcoord);
        }
        dataset.write(data);

        written_maps.push_back(name);
        // Link the coordinates to the map
        group.create_link("coordinates", "/coordinates/" + coord_name);
      } else {
      }
#endif
    }

#ifdef H5CPP
    hdf5::file::File &get_file();
#endif

    void flush(const double time);

  private:
    const JSONValue &json;
    const std::string top_folder;
    const std::string output_name;

    std::list<std::string> written_scalars;
    std::list<std::string> written_maps;
    std::list<std::string> written_coords;

    std::list<std::string> initial_scalars;
    std::list<std::string> initial_maps;
    std::list<std::string> initial_coords;

#ifdef H5CPP
    hdf5::file::File h5_file;
    hdf5::node::Group scalars;
    hdf5::node::Group maps;
    hdf5::node::Group coords;
#endif
  };
} // namespace DiFfRG

#ifdef H5CPP
namespace hdf5
{
  namespace datatype
  {
    template <typename T> class TypeTrait<DiFfRG::complex<T>>
    {
    private:
      using element_type = TypeTrait<T>;

    public:
      using Type = DiFfRG::complex<T>;
      using TypeClass = Compound;

      static TypeClass create(const Type & = Type())
      {
        datatype::Compound type = datatype::Compound::create(2 * sizeof(T));

        type.insert("real", 0, element_type::create(T()));
        type.insert("imag", alignof(T), element_type::create(T()));

        return type;
      }
      const static TypeClass &get(const Type & = Type())
      {
        const static TypeClass &cref_ = create();
        return cref_;
      }
    };

    template <size_t N, typename T> class TypeTrait<autodiff::Real<N, T>>
    {
    private:
      using element_type = TypeTrait<T>;

    public:
      using Type = autodiff::Real<N, T>;
      using TypeClass = Compound;

      static TypeClass create(const Type & = Type())
      {
        datatype::Compound type = datatype::Compound::create(N * sizeof(autodiff::Real<N, T>));

        type.insert("val", 0, element_type::create(T()));
        for (size_t i = 1; i <= N; ++i) {
          type.insert("d_" + std::to_string(i), i * sizeof(T), element_type::create(T()));
        }

        return type;
      }
      const static TypeClass &get(const Type & = Type())
      {
        const static TypeClass &cref_ = create();
        return cref_;
      }
    };

    template <typename T, size_t N> class TypeTrait<std::array<T, N>>
    {
    private:
      using element_type = TypeTrait<T>;

    public:
      using Type = std::array<T, N>;
      using TypeClass = Compound;

      static TypeClass create(const Type & = Type())
      {
        datatype::Compound type = datatype::Compound::create(N * sizeof(T));

        for (size_t i = 0; i < N; ++i) {
          type.insert("component " + std::to_string(i), i * sizeof(T), element_type::create(T()));
        }

        return type;
      }
      const static TypeClass &get(const Type & = Type())
      {
        const static TypeClass &cref_ = create();
        return cref_;
      }
    };

    template <typename T, size_t N>
      requires(!std::is_same_v<std::array<T, N>, DiFfRG::device::array<T, N>>)
    class TypeTrait<DiFfRG::device::array<T, N>>
    {
    private:
      using element_type = TypeTrait<T>;

    public:
      using Type = DiFfRG::device::array<T, N>;
      using TypeClass = Compound;

      static TypeClass create(const Type & = Type())
      {
        datatype::Compound type = datatype::Compound::create(N * sizeof(T));

        for (size_t i = 0; i < N; ++i) {
          type.insert("component " + std::to_string(i), i * sizeof(T), element_type::create(T()));
        }

        return type;
      }
      const static TypeClass &get(const Type & = Type())
      {
        const static TypeClass &cref_ = create();
        return cref_;
      }
    };

  } // namespace datatype
} // namespace hdf5
#endif