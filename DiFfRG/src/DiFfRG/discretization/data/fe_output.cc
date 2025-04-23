// standard library
#include <fstream>

// external libraries
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/fe_output.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <uint dim, typename VectorType>
  FEOutput<dim, VectorType>::FEOutput(std::string top_folder, std::string output_name, std::string output_folder,
                                      const JSONValue &json)
      : top_folder(make_folder(top_folder)), output_name(output_name), output_folder(make_folder(output_folder)),
        series_number(0), subdivisions(json.get_uint("/discretization/output_subdivisions"))
  {
    create_folder(top_folder);
    create_folder(top_folder + output_folder);
  }

  template <uint dim, typename VectorType> FEOutput<dim, VectorType>::~FEOutput()
  {
    if (output_thread.joinable()) output_thread.join();
    data_out.clear();
    attached_solutions.clear();
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::flush(double time)
  {
    auto output_func = [&](double time) {
      // Start by writing the FE-function to a .vtu file.
      data_out.build_patches(subdivisions);

      auto flags = DataOutBase::VtkFlags(time, series_number);
      data_out.set_flags(flags);

      const std::string filename_vtu =
          output_folder + output_name + "_" + Utilities::int_to_string(series_number++, 5) + ".vtu";
      std::ofstream output_vtu(top_folder + filename_vtu);
      data_out.write_vtu(output_vtu);

      // Now we add the .vtu file to the time series and write the .pvd file.
      time_series.emplace_back(time, filename_vtu);
      const std::string filename_pvtu = output_name + ".pvd";

      try {
        std::ofstream output_pvd(top_folder + filename_pvtu);
        DataOutBase::write_pvd_record(output_pvd, time_series);
      } catch (const std::exception &e) {
        std::cerr << "Error: Could not write pvd file." << std::endl << e.what() << '\n';
      }

      data_out.clear();
      attached_solutions.clear();
    };

    if (output_thread.joinable()) output_thread.join();
    output_thread = std::thread(output_func, time);
  }

  template <uint dim, typename VectorType>
  void FEOutput<dim, VectorType>::attach(const DoFHandler<dim> &dof_handler, const VectorType &solution,
                                         const std::string &name)
  {
    if (output_thread.joinable()) output_thread.join();
    typename VectorMemory<VectorType>::Pointer tmp(mem);
    *tmp = solution;
    data_out.add_data_vector(dof_handler, *tmp, name);
    attached_solutions.push_back(std::move(tmp));
  }

  template <uint dim, typename VectorType>
  void FEOutput<dim, VectorType>::attach(const DoFHandler<dim> &dof_handler, const VectorType &solution,
                                         const std::vector<std::string> &names)
  {
    if (output_thread.joinable()) output_thread.join();
    typename VectorMemory<VectorType>::Pointer tmp(mem);
    *tmp = solution;
    data_out.add_data_vector(dof_handler, *tmp, names);
    attached_solutions.push_back(std::move(tmp));
  }

  template class FEOutput<1, dealii::Vector<double>>;
  template class FEOutput<2, dealii::Vector<double>>;
  template class FEOutput<3, dealii::Vector<double>>;
  template class FEOutput<1, dealii::BlockVector<double>>;
  template class FEOutput<2, dealii::BlockVector<double>>;
  template class FEOutput<3, dealii::BlockVector<double>>;
} // namespace DiFfRG