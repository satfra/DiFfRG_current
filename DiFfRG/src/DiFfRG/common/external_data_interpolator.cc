// standard library
#include <iostream>

// external libraries
#include <rapidcsv.h>

// DiFfRG
#include <DiFfRG/common/external_data_interpolator.hh>

namespace DiFfRG
{
  ExternalDataInterpolator::ExternalDataInterpolator() {}

  ExternalDataInterpolator::ExternalDataInterpolator(std::vector<std::string> input_files,
                                                     std::vector<std::function<double(double)>> post_processors,
                                                     char separator, bool has_header, uint x_column, uint order)
  {
    setup(input_files, post_processors, separator, has_header, x_column, order);
  }

  ExternalDataInterpolator::ExternalDataInterpolator(std::vector<std::string> input_files, char separator,
                                                     bool has_header, uint x_column, uint order)
  {
    setup(input_files, {}, separator, has_header, x_column, order);
  }

  void ExternalDataInterpolator::setup(const std::vector<std::string> &input_files,
                                       const std::vector<std::function<double(double)>> &post_processors,
                                       char separator, bool has_header, uint x_column, uint order)
  {
    for (const auto &file : input_files) {
      // Open the file
      int label_0 = has_header ? 0 : -1;
      rapidcsv::Document doc(file, rapidcsv::LabelParams(label_0, -1), rapidcsv::SeparatorParams(separator));

      // Now we read all columns into the vector
      for (uint i = 0; i < doc.GetColumnCount(); ++i)
        file_data.emplace_back(doc.GetColumn<double>(i));
    }

    // Find rows that contain NaN in any column
    const size_t n_rows = file_data[0].size();
    std::vector<bool> keep(n_rows, true);
    size_t n_removed = 0;
    for (size_t i = 0; i < n_rows; ++i) {
      for (size_t j = 0; j < file_data.size(); ++j) {
        if (std::isnan(file_data[j][i])) {
          keep[i] = false;
          n_removed++;
          break;
        }
      }
    }

    if (n_removed != 0)
      std::cout << "ExternalDataInterpolator::setup: Removed " << n_removed << " rows with NaN values." << std::endl;

    // Remove NaN rows using copy-if pattern (O(n) per column instead of O(n*removed))
    if (n_removed > 0) {
      for (size_t col = 0; col < file_data.size(); ++col) {
        size_t write = 0;
        for (size_t row = 0; row < n_rows; ++row) {
          if (keep[row]) file_data[col][write++] = file_data[col][row];
        }
        file_data[col].resize(write);
      }
    }

    // Apply the post processors
    for (uint i = 0; i < std::min(file_data.size(), post_processors.size()); ++i)
      for (auto &value : file_data[i])
        value = post_processors[i](value);

    // Create the interpolators
    for (uint i = 0; i < file_data.size(); ++i)
      interpolators.emplace_back(file_data[x_column].begin(), file_data[x_column].end(), file_data[i].begin(), order);

    max_x = 0.;
    min_x = 1e200;
    for (const auto &x : file_data[x_column]) {
      max_x = std::max(max_x, x);
      min_x = std::min(min_x, x);
    }

    // Check consistency
    if (!check_consistency(1e-4))
      throw std::runtime_error("ExternalDataInterpolator::setup: Interpolator "
                               "is not consistent with the original data!");
  }

  bool ExternalDataInterpolator::check_consistency(double tolerance) const
  {
    for (uint i = 0; i < file_data.size(); ++i)
      for (uint j = 0; j < file_data[i].size(); ++j)
        if (std::abs(interpolators[i](file_data[0][j]) - file_data[i][j]) > tolerance * std::abs(file_data[i][j])) {
          std::cout << "ExternalDataInterpolator::check_consistency: "
                       "Interpolator is not consistent with the original data!"
                    << std::endl;
          std::cout << "ExternalDataInterpolator::check_consistency: "
                       "Interpolator value: "
                    << interpolators[i](file_data[0][j]) << std::endl;
          std::cout << "ExternalDataInterpolator::check_consistency: Original value: " << file_data[i][j] << std::endl;
          std::cout << "ExternalDataInterpolator::check_consistency: Relative "
                       "difference: "
                    << std::abs(interpolators[i](file_data[0][j]) - file_data[i][j]) / std::abs(file_data[i][j])
                    << std::endl;
          return false;
        }
    return true;
  }

  double ExternalDataInterpolator::value(double x, uint i) const
  {
    if (x >= max_x)
      return interpolators[i](max_x);
    else if (x <= min_x)
      return interpolators[i](min_x);
    else
      return interpolators[i](x);
  }

  double ExternalDataInterpolator::derivative(double x, uint i) const
  {
    if (x >= max_x)
      return interpolators[i].prime(max_x);
    else if (x <= min_x)
      return interpolators[i].prime(min_x);
    else
      return interpolators[i].prime(x);
  }
} // namespace DiFfRG