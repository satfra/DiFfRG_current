// DiFfRG
#include <DiFfRG/common/csv_reader.hh>

namespace DiFfRG
{
  CSVReader::CSVReader(std::string input_file, char separator, bool has_header)
  {
    // Open the file
    int label_0 = has_header ? 0 : -1;
    document = std::make_unique<rapidcsv::Document>(input_file, rapidcsv::LabelParams(label_0, -1),
                                                    rapidcsv::SeparatorParams(separator));
  }

  double CSVReader::value(const size_t col, const size_t row) const { return document->GetCell<double>(col, row); }

  double CSVReader::value(const std::string &col, const size_t row) const
  {
    return document->GetCell<double>(col, row);
  }

  size_t CSVReader::n_rows() const { return document->GetRowCount(); }

  size_t CSVReader::n_cols() const { return document->GetColumnCount(); }

} // namespace DiFfRG