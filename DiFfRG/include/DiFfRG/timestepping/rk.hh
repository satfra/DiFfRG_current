#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType = dealii::SparseMatrix<get_type::NumberType<VectorType>>,
            uint dim = 0>
  class TimeStepperRK : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    using Base = AbstractTimestepper<VectorType, SparseMatrixType, dim>;

  public:
    using NumberType = typename Base::NumberType;
    using BlockVectorType = typename Base::BlockVectorType;

    using Base::assembler, Base::data_out, Base::config, Base::adaptor;
    /// Forwards to the base with this stepper's kind. Not `using Base::Base;`: the base needs to
    /// know which /timestepping/ section to read, and it cannot ask a virtual function for that
    /// from inside its own constructor.
    TimeStepperRK(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler,
                  OutputSession<dim, VectorType> *data_out, AbstractAdaptor<VectorType> *adaptor = nullptr)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/false, /*explicit=*/true)
    {
    }
    using Base::console_out;
    using Base::verbosity, Base::output_dt, Base::impl, Base::expl;

    virtual void run(AbstractFlowingVariables<NumberType, VectorType> *initial_condition, const double t_start,
                     const double t_stop) override;

    void set_method(const uint method);


  private:
    void run(VectorType &initial_data, const double t_start, const double t_stop);
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void run_vars(VectorType &initial_data, const double t_start, const double t_stop);

    uint method = 4;
  };
} // namespace DiFfRG
