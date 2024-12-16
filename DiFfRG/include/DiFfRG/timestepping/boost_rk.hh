#pragma once

// external libraries
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>

namespace DiFfRG
{
  template <int prec> struct stepperChoice {
    static_assert(prec == 0 || prec == 1, "Only precisions 0 and 1 are supported!");

    using error_stepper_type_0 = boost::numeric::odeint::runge_kutta_cash_karp54<Eigen::VectorXd>;
    using error_stepper_type_1 = boost::numeric::odeint::runge_kutta_fehlberg78<Eigen::VectorXd>;

    using value = std::conditional_t<prec == 0, error_stepper_type_0, error_stepper_type_1>;
  };

  /**
   * @brief A class to perform time stepping using adaptive Boost Runge-Kutta methods.
   * This stepper uses adaptive time steps and is fully explicit.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   * @tparam prec Algorithm choice: 0 for Cash-Karp54 (5th order), 1 for Fehlberg78 (8th order)
   *
   */
  template <typename VectorType, typename SparseMatrixType, uint dim, int prec>
  class TimeStepperBoostRK : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
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

    /**
     * @brief Run the time stepping algorithm.
     */
    virtual void run(AbstractFlowingVariables<NumberType> *initial_condition, const double t_start,
                     const double t_stop) override;

  private:
    void run(VectorType &initial_data, const double t_start, const double t_stop);
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void run_vars(VectorType &initial_data, const double t_start, const double t_stop);
  };

  /**
   * @brief A class to perform time stepping using the adaptive Boost Cash-Karp54 method.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   */
  template <typename VectorType, typename SparseMatrixType = dealii::SparseMatrix<get_type::NumberType<VectorType>>,
            uint dim = 0>
  using TimeStepperBoostRK54 = TimeStepperBoostRK<VectorType, SparseMatrixType, dim, 0>;

  /**
   * @brief A class to perform time stepping using the adaptive Boost Fehlberg78 method.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   */
  template <typename VectorType, typename SparseMatrixType = dealii::SparseMatrix<get_type::NumberType<VectorType>>,
            uint dim = 0>
  using TimeStepperBoostRK78 = TimeStepperBoostRK<VectorType, SparseMatrixType, dim, 1>;
} // namespace DiFfRG