#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>
#include <DiFfRG/timestepping/boost_rk.hh>

namespace DiFfRG
{
  /**
   * @brief A class to perform time stepping using the adaptive Boost Runge-Kutta method for the explicit part and
   * SUNDIALS IDA for the implicit part. In this scheme, the IDA stepper is the controller and the Boost RK stepper
   * solves the explicit part of the problem on-demand.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   * @tparam prec Algorithm choice: 0 for Cash-Karp54 (5th order), 1 for Fehlberg78 (8th order)
   */
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver, int prec>
  class TimeStepperSUNDIALS_IDA_BoostRK : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    using Base = AbstractTimestepper<VectorType, SparseMatrixType, dim>;

  public:
    using NumberType = typename Base::NumberType;
    using InverseSparseMatrixType = typename Base::InverseSparseMatrixType;
    using BlockVectorType = typename Base::BlockVectorType;

    using Base::assembler, Base::data_out, Base::json, Base::adaptor;
    using Base::Base;
    using Base::console_out;
    using Base::verbosity, Base::output_dt, Base::impl, Base::expl;

    using error_stepper_type = typename stepperChoice<prec>::value;

    virtual void run(AbstractFlowingVariables<NumberType> *initial_condition, const double t_start,
                     const double t_stop) override;

  private:
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void update_variables(VectorType &variable_y, const VectorType &spatial_y, const double t);
  };

  /**
   * @brief A class to perform time stepping using the adaptive Boost Runge-Kutta Cash-Karp54 method for the explicit
   * part and SUNDIALS IDA for the implicit part. In this scheme, the IDA stepper is the controller and the Boost RK
   * stepper solves the explicit part of the problem on-demand.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   */
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  using TimeStepperSUNDIALS_IDA_BoostRK54 =
      TimeStepperSUNDIALS_IDA_BoostRK<VectorType, SparseMatrixType, dim, LinearSolver, 0>;

  /**
   * @brief A class to perform time stepping using the adaptive Boost Runge-Kutta Fehlberg78 method for the explicit
   * part and SUNDIALS IDA for the implicit part. In this scheme, the IDA stepper is the controller and the Boost RK
   * stepper solves the explicit part of the problem on-demand.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   */
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  using TimeStepperSUNDIALS_IDA_BoostRK78 =
      TimeStepperSUNDIALS_IDA_BoostRK<VectorType, SparseMatrixType, dim, LinearSolver, 1>;
} // namespace DiFfRG