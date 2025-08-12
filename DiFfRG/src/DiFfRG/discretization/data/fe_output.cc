// standard library
#include <fstream>

// external libraries
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/fe_output.hh>
#include <stdexcept>

namespace DiFfRG
{
  using namespace dealii;

  template <uint dim, typename VectorType>
  FEOutput<dim, VectorType>::FEOutput(std::string top_folder, std::string output_name, std::string output_folder,
                                      const JSONValue &json)
      : top_folder(make_folder(top_folder)), output_name(output_name), output_folder(make_folder(output_folder)),
        filename_pvd(output_name + ".pvd"), series_number(0),
        subdivisions(json.get_uint("/discretization/output_subdivisions")), buffer_size([&]() {
          uint read_size = json.get_uint("/discretization/output_buffer_size", 100);
          // Ensure that the buffer size is at least 1.
          return read_size == 0 ? 1 : read_size;
        }()),
        data_outs(buffer_size)
  {
    create_folder(this->top_folder);
    create_folder(this->top_folder + this->output_folder);

    // Initialize the attached_solutions list.
    attached_solutions.emplace_back();
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::update_buffers()
  {
    if (buffer_size == 1) return;

    if (output_threads.size() + 1 != attached_solutions.size())
      throw std::logic_error(
          "FEOutput::flush: The number of output threads does not match the size of the attached_solutions lisea.");

    // If the number of attached solutions exceeds the buffer size, we clear the buffers.
    while (attached_solutions.size() > buffer_size) {
      // first, join the thread if it is still running
      if (output_threads.front().joinable()) output_threads.front().join();
      output_threads.pop_front();
      // then, remove the first attached_solutions
      attached_solutions.pop_front();
    }
  }

  template <uint dim, typename VectorType> FEOutput<dim, VectorType>::~FEOutput()
  {
    // Join all threads before destruction.
    for (auto &thread : output_threads) {
      if (thread.joinable()) thread.join();
    }

    // Clear the data_out and attached_solutions lists.
    output_threads.clear();
    attached_solutions.clear();
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::flush(double time)
  {
    update_buffers();

    // The .vtu file will be named like output_name_000001.vtu, where the number is the
    // series number. This imposes a limit of 1 million series, which is more than enough for most applications.
    const std::string filename_vtu =
        output_folder + output_name + "_" + Utilities::int_to_string(series_number, 6) + ".vtu";

    // We add the .vtu file to the time series and write the .pvd file.
    time_series.emplace_back(time, filename_vtu);
    try {
      std::ofstream output_pvd(top_folder + filename_pvd);
      DataOutBase::write_pvd_record(output_pvd, time_series);
    } catch (const std::exception &e) {
      throw std::runtime_error("FEOutput::flush: Could not write pvd file.");
    }

    auto output_func = [=, this](const uint m_series_number, const double m_time) {
      auto &m_data_out = data_outs[m_series_number % buffer_size];

      // Start by writing the FE-function to a .vtu file.
      m_data_out.build_patches(subdivisions);

      auto flags = DataOutBase::VtkFlags(m_time, m_series_number);
      m_data_out.set_flags(flags);

      std::ofstream output_vtu(top_folder + filename_vtu);
      m_data_out.write_vtu(output_vtu);

      m_data_out.clear();
    };

    // If the buffer size is 1, we save ourselves the cost of spawning a thread
    if (buffer_size == 1) {
      output_func(series_number, time);
      attached_solutions.back().clear();
    } else {
      output_threads.emplace_back(output_func, series_number, time);
      // We need to prepare the next attached_solutions list
      attached_solutions.emplace_back();
    }

    series_number++;
  }

  template <uint dim, typename VectorType>
  void FEOutput<dim, VectorType>::attach(const DoFHandler<dim> &dof_handler, const VectorType &solution,
                                         const std::string &name)
  {
    update_buffers();

    attached_solutions.back().emplace_back(mem);
    *(attached_solutions.back().back()) = solution;

    auto &m_data_out = data_outs[series_number % buffer_size];
    m_data_out.add_data_vector(dof_handler, *(attached_solutions.back().back()), name);
  }

  template <uint dim, typename VectorType>
  void FEOutput<dim, VectorType>::attach(const DoFHandler<dim> &dof_handler, const VectorType &solution,
                                         const std::vector<std::string> &names)
  {
    update_buffers();

    attached_solutions.back().emplace_back(mem);
    *(attached_solutions.back().back()) = solution;

    auto &m_data_out = data_outs[series_number % buffer_size];
    m_data_out.add_data_vector(dof_handler, *(attached_solutions.back().back()), names);
  }

  template class FEOutput<1, dealii::Vector<double>>;
  template class FEOutput<2, dealii::Vector<double>>;
  template class FEOutput<3, dealii::Vector<double>>;
  template class FEOutput<1, dealii::BlockVector<double>>;
  template class FEOutput<2, dealii::BlockVector<double>>;
  template class FEOutput<3, dealii::BlockVector<double>>;
} // namespace DiFfRG