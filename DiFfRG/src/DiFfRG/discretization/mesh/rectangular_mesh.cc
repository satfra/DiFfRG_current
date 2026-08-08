// standard library
#include "DiFfRG/discretization/mesh/configuration_mesh.hh"

// external libraries
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

// DiFfRG
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <uint dim>
  RectangularMesh<dim>::RectangularMesh(const ConfigTree &json)
      : RectangularMesh<dim>(Config::ConfigurationMesh<dim>(json), RectangularMeshOptions{})
  {
  }

  template <uint dim>
  RectangularMesh<dim>::RectangularMesh(const ConfigTree &json, const RectangularMeshOptions options)
      : RectangularMesh<dim>(Config::ConfigurationMesh<dim>(json), options)
  {
  }

  template <uint dim>
  RectangularMesh<dim>::RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config)
      : RectangularMesh<dim>(mesh_config, RectangularMeshOptions{})
  {
  }

  template <uint dim>
  RectangularMesh<dim>::RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config,
                                        const RectangularMeshOptions options)
      : mesh_config(mesh_config), options(options)
  {
    static_assert(dim > 0 && dim < 4);
    make_grid();
  }

  template <uint dim> void RectangularMesh<dim>::make_grid()
  {
    const auto triangulation_data = mesh_config.get_triangulation_data(options.origin_cell_centered);
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, triangulation_data.step_sizes,
                                                      triangulation_data.lower_left, triangulation_data.upper_right);

    triangulation.refine_global(mesh_config.refine);
  }

  template class RectangularMesh<1>;
  template class RectangularMesh<2>;
  template class RectangularMesh<3>;
} // namespace DiFfRG
