#pragma once

// DiFfRG
#include <DiFfRG/common/utils.hh>

// external libraries
#include <deal.II/base/hdf5.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/numerics/data_out.h>

// standard library
#include <list>
#include <thread>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief A class to output finite element data to disk as .vtu files and .pvd time series.
   *
   * @tparam dim Dimension of the problem
   * @tparam VectorType Type of the vector
   */
  template <uint dim, typename VectorType> class FEOutput
  {
  public:
    /**
     * @brief Construct a new FEOutput object
     *
     * @param dof_handler DoFHandler associated with the solution
     * @param top_folder Folder where the output will be written, i.e. the folder containing the .pvd file.
     * @param output_name Name of the output, i.e. the name of the .pvd file.
     * @param output_folder Folder where the .vtu files will be saved. Should be relative to top_folder.
     * @param subdivisions Number of subdivisions of the cells in the .vtu files.
     */
    FEOutput(std::string top_folder, std::string output_name, std::string output_folder, const JSONValue &json);

    FEOutput();
    ~FEOutput();

    void set_h5_group(std::shared_ptr<HDF5::Group> h5_group);

    /**
     * @brief Attach a solution to the output.
     *
     * @param dof_handler The DoFHandler associated with the solution.
     * @param solution The solution to attach.
     * @param name The name of the solution.
     */
    void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::string &name);

    /**
     * @brief Attach a solution to the output.
     *
     * @param dof_handler The DoFHandler associated with the solution.
     * @param solution The solution to attach.
     * @param names The names of each component of the solution.
     */
    void attach(const DoFHandler<dim> &dof_handler, const VectorType &solution, const std::vector<std::string> &names);

    /**
     * @brief Flush all attached solutions to disk.
     *
     * @param time The time tag to attach to the solution.
     */
    void flush(double time);

  private:
    const std::string top_folder;
    const std::string output_name;
    const std::string output_folder;
    const std::string filename_pvd;
    const uint buffer_size;

    uint series_number, subdivisions;
    std::vector<std::pair<double, std::string>> time_series;

    constexpr static uint safe_dim = dim == 0 ? 1 : dim;
    std::vector<DataOut<safe_dim>> data_outs;
    std::list<std::thread> output_threads;
    std::list<std::list<typename VectorMemory<VectorType>::Pointer>> attached_solutions;

    GrowingVectorMemory<VectorType> mem;

    void update_buffers();

    std::shared_ptr<HDF5::Group> h5_group;
    bool save_vtk;
  };
} // namespace DiFfRG