#pragma once

// DiFfRG
#include <DiFfRG/common/json.hh>

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
    template <typename T> void scalar(const std::string &name, const T value);
    template <typename COORD, typename INTERP>
    void map_interp(const std::string &name, const COORD &coordinates, const INTERP &interpolator)
    {
#ifndef H5CPP
      throw std::runtime_error(
          "HDF5Output::map_interp: HDF5 support is not enabled. Please compile with H5CPP support.");
#else
      if (initial_maps.size() > 0 && std::find(initial_maps.begin(), initial_maps.end(), name) == initial_maps.end())
        throw std::runtime_error("HDF5Output::map_interp: The map '" + name + "' has not been registered before!");

      using value_type = typename std::decay<INTERP>::value_type;
      using ctype = typename std::decay<COORD>::ctype;

      if (!maps.has_group(name)) {
        hdf5::node::Group group = maps.create_group(name);
        hdf5::Dimensions dims;
        for (const auto &dim : coordinates.sizes())
          dims.push_back(dim);
        hdf5::dataspace::Simple space(dims);
        auto c_type = hdf5::datatype::create<std::decay_t<ctype>>();
        auto d_type = hdf5::datatype::create<std::decay_t<value_type>>();
        group.create_dataset("coordinates", c_type, space);
        group.create_dataset("data", d_type, space);
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

    std::list<std::string> initial_scalars;
    std::list<std::string> initial_maps;

#ifdef H5CPP
    hdf5::file::File h5_file;
    hdf5::node::Group scalars;
    hdf5::node::Group maps;
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
        datatype::Compound type = datatype::Compound::create(sizeof(DiFfRG::complex<T>));

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
        datatype::Compound type = datatype::Compound::create(sizeof(DiFfRG::complex<T>));

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
  } // namespace datatype
} // namespace hdf5
#endif