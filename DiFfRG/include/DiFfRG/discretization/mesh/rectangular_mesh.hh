#pragma once

// external libraries
#include <deal.II/grid/tria.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>

namespace DiFfRG
{
  using namespace dealii;
  /**
   * @brief Class to manage the discretization mesh, also called grid and triangluation, on which we simulate.
   * This class only builds cartesian, regular grids, however cell density in all directions can be chosen
   * independently.
   *
   * @tparam dim dimensionality of the spatial discretization.
   */
  template <uint dim_> class RectangularMesh
  {
  public:
    static constexpr uint dim = dim_;
    static constexpr bool is_rectangular = true;

    /**
     * @brief Construct a new RectangularMesh object.
     * @param json JSONValue object containing the parameters for the mesh.
     */
    [[deprecated("Please use RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config) instead")]]
    RectangularMesh(const JSONValue &json);

    RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config);

    Triangulation<dim> &get_triangulation() { return triangulation; }
    const Triangulation<dim> &get_triangulation() const { return triangulation; }

  protected:
    virtual void make_grid();

    const Config::ConfigurationMesh<dim> &mesh_config;
    Triangulation<dim> triangulation;
  };
} // namespace DiFfRG