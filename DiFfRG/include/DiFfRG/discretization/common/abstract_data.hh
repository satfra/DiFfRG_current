#pragma once

// external libraries
#include <deal.II/lac/vector.h>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief A class to set up initial data for whatever discretization we have chosen.
   *        Also used to switch/manage memory, vectors, matrices over interfaces between spatial discretization and
   * separate variables.
   *
   * @tparam Discretization Spatial Discretization used in the system
   */
  template <typename NumberType> class AbstractFlowingVariables
  {
  public:
    /**
     * @brief Obtain the data vector holding both spatial (block 0) and variable (block 1) data.
     *
     * @return BlockVector<NumberType>& The data vector.
     */
    virtual BlockVector<NumberType> &data() = 0;
    virtual const BlockVector<NumberType> &data() const = 0;

    /**
     * @brief Obtain the spatial data vector.
     *
     * @return Vector<NumberType>& The spatial data vector.
     */
    virtual Vector<NumberType> &spatial_data() = 0;
    virtual const Vector<NumberType> &spatial_data() const = 0;

    /**
     * @brief Obtain the variable data vector.
     *
     * @return Vector<NumberType>& The variable data vector.
     */
    virtual Vector<NumberType> &variable_data() = 0;
    virtual const Vector<NumberType> &variable_data() const = 0;
  };

} // namespace DiFfRG