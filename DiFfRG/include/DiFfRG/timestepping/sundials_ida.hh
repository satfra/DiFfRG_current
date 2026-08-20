#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>

#include <functional>
#include <utility>

namespace DiFfRG
{
  /**
   * @brief A class to perform time stepping using the SUNDIALS IDA solver.
   * This stepper uses adaptive time steps and is fully implicit. Furthermore, IDA allows for the solution of DAEs.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   */
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  class TimeStepperSUNDIALS_IDA : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    using Base = AbstractTimestepper<VectorType, SparseMatrixType, dim>;

  public:
    using NumberType = typename Base::NumberType;
    using BlockVectorType = typename Base::BlockVectorType;
    using IDAErrorDofCallback = std::function<void(const IDAErrorDofDiagnostics &)>;

    using Base::assembler, Base::data_out, Base::config, Base::adaptor;
    /// Forwards to the base with this stepper's kind. Not `using Base::Base;`: the base needs to
    /// know which /timestepping/ section to read, and it cannot ask a virtual function for that
    /// from inside its own constructor.
    TimeStepperSUNDIALS_IDA(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler,
                            OutputSession<dim, VectorType> *data_out, AbstractAdaptor<VectorType> *adaptor = nullptr)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/true, /*explicit=*/false)
    {
    }
    using Base::console_out;
    using Base::verbosity, Base::output_dt, Base::impl, Base::expl;

    virtual void run(AbstractFlowingVariables<NumberType, VectorType> *initial_condition, const double t_start,
                     const double t_stop) override;


    void set_ida_error_dof_callback(IDAErrorDofCallback callback) { ida_error_dof_callback = std::move(callback); }

  private:
    void run(VectorType &initial_data, const double t_start, const double t_stop);
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void run_vars(VectorType &initial_data, const double t_start, const double t_stop);

    IDAErrorDofCallback ida_error_dof_callback;
  };
} // namespace DiFfRG
