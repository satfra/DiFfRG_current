#pragma once

// external libraries
#include <deal.II/grid/tria.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>

namespace DiFfRG
{
  using namespace dealii;

  struct RectangularMeshOptions {
    bool origin_cell_centered = false;
  };

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

    /**
     * @brief Construct a new RectangularMesh object from JSON configuration.
     * @param json JSONValue object containing the parameters for the mesh.
     * @param options Options controlling rectangular mesh construction.
     *
     * All configured axes must start at zero when origin centering is enabled.
     */
    [[deprecated("Please use RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config, "
                 "RectangularMeshOptions options) instead")]]
    RectangularMesh(const JSONValue &json, RectangularMeshOptions options);

    RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config);

    /**
     * @brief Construct a new RectangularMesh object.
     * @param mesh_config Configuration of the rectangular mesh.
     * @param options Options controlling rectangular mesh construction.
     *
     * All configured axes must start at zero when origin centering is enabled.
     */
    RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config, RectangularMeshOptions options);

    Triangulation<dim> &get_triangulation() { return triangulation; }
    const Triangulation<dim> &get_triangulation() const { return triangulation; }

  protected:
    virtual void make_grid();

    const Config::ConfigurationMesh<dim> mesh_config;
    const RectangularMeshOptions options;
    Triangulation<dim> triangulation;
  };
} // namespace DiFfRG
