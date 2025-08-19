#pragma once

// DiFfRG
#include <DiFfRG/common/json.hh>

// standard library
#include <fstream>
#include <map>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief A class to output data to a CSV file.
   *
   * In every time step, the user can add values to the output, which will be written to the file when flush is called.
   * Every timestep has to contain the same values, but the order in which they are added can change.
   */
  class CsvOutput
  {
  public:
    /**
     * @brief Construct a new Csv Output object
     *
     * @param top_folder The top folder to store the output in.
     * @param output_name The name of the output file.
     * @param json The JSON object containing the parameters.
     */
    CsvOutput(const std::string top_folder, const std::string output_name, const JSONValue &json);

    /**
     * @brief Add a value to the output.
     *
     * @param name The name of the value.
     * @param value The value to add.
     */
    void value(const std::string &name, const double value);

    /**
     * @brief Add a value to the output.
     *
     * @param name The name of the value.
     * @param value The value to add.
     */
    void flush(const double time);

    /**
     * @brief Set the value of Lambda. If Lambda is set, the output will contain a column for k = exp(-t) * Lambda.
     *
     * @param Lambda The value of Lambda.
     */
    void set_Lambda(const double Lambda);

  private:
    const JSONValue &json;
    const std::string top_folder;
    const std::string output_name;
    std::ofstream output_stream;

    std::vector<std::string> insertion_order;
    std::map<std::string, std::vector<double>> values;

    std::vector<std::string> header;
    std::vector<double> time_values;
    std::vector<double> k_values;

    double Lambda;
  };
} // namespace DiFfRG