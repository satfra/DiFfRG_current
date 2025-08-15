// standard library
#include <filesystem>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/csv_output.hh>

namespace DiFfRG
{
  CsvOutput::CsvOutput(const std::string top_folder, const std::string output_name, const JSONValue &json)
      : json(json), top_folder(make_folder(top_folder)), output_name(output_name), Lambda(-1.)
  {
    create_folder(this->top_folder);
    std::filesystem::path path = this->top_folder + this->output_name;
    create_folder(path.parent_path().string());

    output_stream = std::ofstream(top_folder + output_name, std::ofstream::trunc);
    output_stream << std::scientific;
  }

  void CsvOutput::value(const std::string &name, const double value)
  {
    if ((time_values.size() == 0) &&
        std::find(insertion_order.begin(), insertion_order.end(), name) == insertion_order.end())
      insertion_order.push_back(name);
    values[name].push_back(value);
  }

  void CsvOutput::flush(const double time)
  {
    if (time_values.size() == 0) {
      // Create the header.
      header.clear();
      header.push_back("t");
      if (Lambda > 0) header.push_back("k [GeV]");
      for (const auto &entry : insertion_order)
        header.push_back(entry);

      // Write the header to the file.
      for (const auto &entry : header) {
        output_stream << strip_name(entry);
        if (entry != header.back()) output_stream << ",";
      }
      output_stream << std::endl;
    } else {
      // Check that the number of values matches the number of values in the header.
      if (values.size() != header.size() - 1 - int(Lambda > 0))
        throw std::runtime_error("CsvOutput::flush: The number of values in '" + output_name +
                                 "' does not match the number of values registered in the header.");
    }

    time_values.push_back(time);
    k_values.push_back(std::exp(-time) * Lambda);

    // Write the values to the file.
    output_stream << time << ",";
    if (Lambda > 0) output_stream << k_values.back() << ",";
    for (const auto &entry : header) {
      if (entry == "t" || entry == "k [GeV]") continue;

      output_stream << values[entry].back();
      if (entry != header.back()) output_stream << ",";
    }
    output_stream << std::endl;
  }

  void CsvOutput::set_Lambda(const double Lambda)
  {
    // This needs safety checks, so that Lambda can only be set once, before any call to flush().
    if (time_values.size() > 0 && !is_close(this->Lambda, Lambda))
      throw std::runtime_error("Lambda has either already been set or there has been an attempt to change it.");
    this->Lambda = Lambda;
  }
} // namespace DiFfRG