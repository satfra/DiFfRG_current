// standard library
#include "DiFfRG/discretization/mesh/configuration_mesh.hh"
#include <fstream>

// external libraries
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

// DiFfRG
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

namespace DiFfRG
{
  using namespace dealii;

  namespace internal
  {
    std::array<double, 3> string_to_range(const std::string str);
    std::vector<std::string> string_to_substrings_array(const std::string str);
    void check_ranges_consistency(const std::vector<std::array<double, 3>> &ranges);
    void append_range(std::vector<std::vector<double>> &dest, const std::array<double, 3> &range, const uint d);
  } // namespace internal

  template <uint dim>
  RectangularMesh<dim>::RectangularMesh(const JSONValue &json)
      : DiFfRG::RectangularMesh<dim>(Config::ConfigurationMesh<dim>(json))
  {
  }

  template <uint dim>
  RectangularMesh<dim>::RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config) : mesh_config(mesh_config)
  {
    static_assert(dim > 0 && dim < 4);
    make_grid();
  }

  template <uint dim> void RectangularMesh<dim>::make_grid()
  {
    const auto lower_left = mesh_config.get_lower_left();
    const auto upper_right = mesh_config.get_upper_right();
    const auto step_sizes = mesh_config.get_step_withs_for_triangulation();
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, lower_left, upper_right);

    triangulation.refine_global(mesh_config.refine);

    if constexpr (dim == 2) {
      std::ofstream out("grid.svg");
      dealii::GridOut grid_out;
      grid_out.write_svg(triangulation, out);
    }
  }

  template class RectangularMesh<1>;
  template class RectangularMesh<2>;
  template class RectangularMesh<3>;
} // namespace DiFfRG