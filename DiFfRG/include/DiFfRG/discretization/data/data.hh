#pragma once

// external libraries
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>

namespace DiFfRG
{
  using namespace dealii;

  namespace internal
  {
    template <uint dim, typename NumberType> class FunctionFromLambda : public Function<dim, NumberType>
    {
      using FUN = std::function<void(const Point<dim> &, Vector<NumberType> &)>;

    public:
      FunctionFromLambda(FUN &&fun_, uint components) : Function<dim, NumberType>(components), fun(std::move(fun_)) {}
      virtual void vector_value(const Point<dim> &p, Vector<NumberType> &values) const override { fun(p, values); }

    private:
      FUN fun;
    };
  } // namespace internal

  /**
   * @brief A class to set up initial data for whatever discretization we have chosen.
   *        Also used to switch/manage memory, vectors, matrices over interfaces between spatial discretization and
   * separate variables.
   *
   * @tparam Discretization Spatial Discretization used in the system
   */
  template <typename NT = double> class FlowingVariables : public AbstractFlowingVariables<NT>
  {
  public:
    using NumberType = NT;

    /**
     * @brief Construct a new Flowing Variables object
     */
    FlowingVariables() {}

    /**
     * @brief Interpolates the initial condition from a numerical model.
     *
     * @param model The model to interpolate from. Must provide a method initial_condition(const Point<dim> &,
     * Vector<NumberType> &)
     */
    template <typename Model> void interpolate(const Model &model)
    {
      std::vector<uint> block_structure{0};
      block_structure.push_back(Model::Components::count_variables());
      m_data = (block_structure);
      if (m_data.n_blocks() > 1) model.initial_condition_variables(m_data.block(1));
    }

    /**
     * @brief Obtain the data vector holding both spatial (block 0) and variable (block 1) data.
     *
     * @return BlockVector<NumberType>& The data vector.
     */
    virtual BlockVector<NumberType> &data() override { return m_data; }
    virtual const BlockVector<NumberType> &data() const override { return m_data; }

    /**
     * @brief Obtain the spatial data vector.
     *
     * @return Vector<NumberType>& The spatial data vector.
     */
    virtual Vector<NumberType> &spatial_data() override { return m_data.block(0); }
    virtual const Vector<NumberType> &spatial_data() const override { return m_data.block(0); }

    /**
     * @brief Obtain the variable data vector.
     *
     * @return Vector<NumberType>& The variable data vector.
     */
    virtual Vector<NumberType> &variable_data() override { return m_data.block(1); }
    virtual const Vector<NumberType> &variable_data() const override { return m_data.block(1); }

  private:
    BlockVector<NumberType> m_data;
  };

  namespace FE
  {
    /**
     * @brief A class to set up initial data for whatever discretization we have chosen.
     *        Also used to switch/manage memory, vectors, matrices over interfaces between spatial discretization and
     * separate variables.
     *
     * @tparam Discretization Spatial Discretization used in the system
     */
    template <typename Discretization>
    class FlowingVariables : public AbstractFlowingVariables<typename Discretization::NumberType>
    {
    public:
      using NumberType = typename Discretization::NumberType;
      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;

      /**
       * @brief Construct a new Flowing Variables object
       *
       * @param discretization The spatial discretization to use
       */
      FlowingVariables(const Discretization &discretization)
          : discretization(discretization), dof_handler(discretization.get_dof_handler())
      {
      }

      /**
       * @brief Interpolates the initial condition from a numerical model.
       *
       * @param model The model to interpolate from. Must provide a method initial_condition(const Point<dim> &,
       * Vector<NumberType> &)
       */
      template <typename Model> void interpolate(const Model &model)
      {
        auto block_structure = discretization.get_block_structure();
        m_data = (block_structure);

        if constexpr (Model::Components::count_fe_functions() > 0) {
          auto interpolating_function = [&model](const auto &p, auto &values) { model.initial_condition(p, values); };
          internal::FunctionFromLambda<dim, NumberType> initial_condition_function(std::move(interpolating_function),
                                                                                   dof_handler.get_fe().n_components());
          VectorTools::interpolate(dof_handler, initial_condition_function, m_data.block(0));
        }
        if (m_data.n_blocks() > 1) model.initial_condition_variables(m_data.block(1));
      }

      /**
       * @brief Obtain the data vector holding both spatial (block 0) and variable (block 1) data.
       *
       * @return BlockVector<NumberType>& The data vector.
       */
      virtual BlockVector<NumberType> &data() override { return m_data; }
      virtual const BlockVector<NumberType> &data() const override { return m_data; }

      /**
       * @brief Obtain the spatial data vector.
       *
       * @return Vector<NumberType>& The spatial data vector.
       */
      virtual Vector<NumberType> &spatial_data() override { return m_data.block(0); }
      virtual const Vector<NumberType> &spatial_data() const override { return m_data.block(0); }

      /**
       * @brief Obtain the variable data vector.
       *
       * @return Vector<NumberType>& The variable data vector.
       */
      virtual Vector<NumberType> &variable_data() override { return m_data.block(1); }
      virtual const Vector<NumberType> &variable_data() const override { return m_data.block(1); }

    private:
      const Discretization &discretization;
      const DoFHandler<dim> &dof_handler;
      BlockVector<NumberType> m_data;
    };
  } // namespace FE

  namespace FV
  {
    /**
     * @brief A class to set up initial data for whatever discretization we have chosen.
     *        Also used to switch/manage memory, vectors, matrices over interfaces between spatial discretization and
     * separate variables.
     *
     * @tparam Discretization Spatial Discretization used in the system
     */
    template <typename Discretization>
    class FlowingVariables : public AbstractFlowingVariables<typename Discretization::NumberType>
    {
    public:
      using NumberType = typename Discretization::NumberType;
      using Components = typename Discretization::Components;
      static constexpr uint dim = Discretization::dim;

      /**
       * @brief Construct a new Flowing Variables object
       *
       * @param discretization The spatial discretization to use
       */
      FlowingVariables(const Discretization &discretization) : discretization(discretization) {}

      /**
       * @brief Interpolates the initial condition from a numerical model.
       *
       * @param model The model to interpolate from. Must provide a method initial_condition(const Point<dim> &,
       * Vector<NumberType> &)
       */
      template <typename Model> void interpolate(const Model &model)
      {
        auto block_structure = discretization.get_block_structure();
        m_data = (block_structure);

        if constexpr (Model::Components::count_fe_functions() > 0) {
          // TODO
        }
        if (m_data.n_blocks() > 1) model.initial_condition_variables(m_data.block(1));
      }

      /**
       * @brief Obtain the data vector holding both spatial (block 0) and variable (block 1) data.
       *
       * @return BlockVector<NumberType>& The data vector.
       */
      virtual BlockVector<NumberType> &data() override { return m_data; }
      virtual const BlockVector<NumberType> &data() const override { return m_data; }

      /**
       * @brief Obtain the spatial data vector.
       *
       * @return Vector<NumberType>& The spatial data vector.
       */
      virtual Vector<NumberType> &spatial_data() override { return m_data.block(0); }
      virtual const Vector<NumberType> &spatial_data() const override { return m_data.block(0); }

      /**
       * @brief Obtain the variable data vector.
       *
       * @return Vector<NumberType>& The variable data vector.
       */
      virtual Vector<NumberType> &variable_data() override { return m_data.block(1); }
      virtual const Vector<NumberType> &variable_data() const override { return m_data.block(1); }

    private:
      const Discretization &discretization;
      BlockVector<NumberType> m_data;
    };
  } // namespace FV

} // namespace DiFfRG
