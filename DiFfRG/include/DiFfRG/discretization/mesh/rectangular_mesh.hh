#pragma once

// external libraries
#include <deal.II/grid/tria.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>

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
    RectangularMesh(const JSONValue &json);

    Triangulation<dim> &get_triangulation() { return triangulation; }
    const Triangulation<dim> &get_triangulation() const { return triangulation; }

  protected:
    virtual void make_grid();

    const JSONValue &json;
    Triangulation<dim> triangulation;
  };
} // namespace DiFfRG