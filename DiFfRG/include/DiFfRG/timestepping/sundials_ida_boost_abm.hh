#pragma once

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
  /**
   * @brief A class to perform time stepping using the Boost Adams-Bashforth-Moulton method for the explicit part and
   * SUNDIALS IDA for the implicit part. This stepper uses fixed time steps in the explicit part and adaptive time steps
   * in the implicit part. IDA acts as the controller and the ABM stepper solves the explicit part of the problem
   * on-demand.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   */
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  class TimeStepperSUNDIALS_IDA_BoostABM : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
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

    virtual void run(AbstractFlowingVariables<NumberType> *initial_condition, const double t_start,
                     const double t_stop) override;

  private:
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void update_variables(VectorType &variable_y, const VectorType &spatial_y, const double t);
  };
} // namespace DiFfRG