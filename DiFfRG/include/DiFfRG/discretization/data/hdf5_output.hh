#pragma once

// DiFfRG
#include <DiFfRG/common/json.hh>

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
  } // namespace datatype
} // namespace hdf5
#endif