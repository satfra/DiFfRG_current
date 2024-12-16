#pragma once

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/discretization/mesh/no_adaptivity.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>

namespace DiFfRG
{
  template <typename VectorType_, typename SparseMatrixType_, uint dim_> class AbstractTimestepper
  {
  protected:
    static constexpr uint dim = dim_;
    using VectorType = VectorType_;
    using NumberType = typename get_type::NumberType<VectorType>;
    using SparseMatrixType = SparseMatrixType_;
    using InverseSparseMatrixType = typename get_type::InverseSparseMatrixType<SparseMatrixType>;
    static_assert(std::is_same_v<VectorType, Vector<NumberType>>, "VectorType must not be Vector<double>!");
    using BlockVectorType = dealii::BlockVector<NumberType>;

  public:
    AbstractTimestepper(const JSONValue &json, 
                        AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler,
                        DataOutput<dim, VectorType> *data_out = nullptr,
                        AbstractAdaptor<VectorType> *adaptor = nullptr
                        )
        : json(json), assembler(assembler), data_out(data_out), adaptor(adaptor),
          start_time(std::chrono::high_resolution_clock::now())
    {
      verbosity = json.get_int("/output/verbosity");
      output_dt = json.get_double("/timestepping/output_dt");

      impl.dt = json.get_double("/timestepping/implicit/dt");
      impl.minimal_dt = json.get_double("/timestepping/implicit/minimal_dt");
      impl.maximal_dt = json.get_double("/timestepping/implicit/maximal_dt");
      impl.abs_tol = json.get_double("/timestepping/implicit/abs_tol");
      impl.rel_tol = json.get_double("/timestepping/implicit/rel_tol");

      expl.dt = json.get_double("/timestepping/explicit/dt");
      expl.minimal_dt = json.get_double("/timestepping/explicit/minimal_dt");
      expl.maximal_dt = json.get_double("/timestepping/explicit/maximal_dt");
      expl.abs_tol = json.get_double("/timestepping/explicit/abs_tol");
      expl.rel_tol = json.get_double("/timestepping/explicit/rel_tol");

      try {
        Lambda = json.get_double("/physical/Lambda");
      } catch (std::exception &e) {
        Lambda = -1.0;
      }
    }

    DataOutput<dim, VectorType> *get_data_out() { 
      if(data_out == nullptr) {
        data_out_default = std::make_shared<DataOutput<dim, VectorType>>(json);
        return data_out_default.get();
      }
      return data_out; 
    }

    AbstractAdaptor<VectorType> *get_adaptor() { 
      if(adaptor == nullptr) {
        adaptor_default = std::make_shared<NoAdaptivity<VectorType>>();
        return adaptor_default.get();
      }
      return adaptor; 
    }

    virtual void run(AbstractFlowingVariables<NumberType> *initial_condition, const double t_start,
                     const double t_stop) = 0;

  protected:
    const JSONValue json;
    AbstractAssembler<VectorType, SparseMatrixType, dim> *assembler;
    DataOutput<dim, VectorType> *data_out;
    AbstractAdaptor<VectorType> *adaptor;

    const std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

    std::shared_ptr<NoAdaptivity<VectorType>> adaptor_default;
    std::shared_ptr<DataOutput<dim, VectorType>> data_out_default;

    double Lambda;
    int verbosity;
    double output_dt;
    struct implicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
    } impl;

    struct explicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
    } expl;

    void console_out(const double t, const std::string name, const int verbosity_level,
                     const double calc_dt_ms = -1.0) const
    {
      if (verbosity >= verbosity_level) {
        std::ios_base::fmtflags oldflags = std::cout.flags();
        std::cout << "[" << name << "]";
        std::cout << "      t: " << std::setw(10) << std::left << std::setprecision(4) << std::scientific << t;
        if (Lambda > 0.0) {
          std::cout << " | k: " << std::setw(10) << std::left << std::setprecision(4) << std::scientific
                    << exp(-t) * Lambda;
        }
        if (calc_dt_ms >= 0.0) {
          std::cout << " | calc_dt: " << time_format_ms(size_t(calc_dt_ms));
        }
        size_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
        std::cout << " | calc_t: " << time_format_ms(milliseconds);
        std::cout << std::endl;
        std::cout.flags(oldflags);
      }
    }
  };
} // namespace DiFfRG