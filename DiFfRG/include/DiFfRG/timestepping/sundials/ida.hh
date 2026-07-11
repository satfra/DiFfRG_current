#pragma once

#include <DiFfRG/timestepping/sundials/common.hh>
#include <DiFfRG/timestepping/sundials/nvector.hh>
#include <DiFfRG/timestepping/sundials/sun_linear_solver.hh>
#include <DiFfRG/timestepping/sundials/sun_matrix.hh>

#include <deal.II/base/index_set.h>

#include <ida/ida.h>
#include <ida/ida_ls.h>

#include <algorithm>
#include <functional>
#include <memory>

namespace DiFfRG::sundials
{
  template <typename VectorType> class IDA
  {
  public:
    class AdditionalData
    {
    public:
      enum InitialConditionCorrection
      {
        none = 0,
        use_y_diff = IDA_YA_YDP_INIT,
        use_y_dot = IDA_Y_INIT,
      };

      AdditionalData(const double initial_time_ = 0., const double final_time_ = 1.,
                     const double initial_step_size_ = 1e-2, const double output_period_ = 1e-1,
                     const double minimum_step_size_ = 1e-6, const unsigned int maximum_order_ = 5,
                     const unsigned int maximum_non_linear_iterations_ = 10, const double ls_norm_factor_ = 0.,
                     const double absolute_tolerance_ = 1e-6, const double relative_tolerance_ = 1e-5,
                     const bool ignore_algebraic_terms_for_errors_ = true,
                     const InitialConditionCorrection ic_type_ = use_y_diff,
                     const InitialConditionCorrection reset_type_ = use_y_diff,
                     const unsigned int maximum_num_steps_ = 100000)
          : initial_time(initial_time_), final_time(final_time_), initial_step_size(initial_step_size_),
            output_period(output_period_), minimum_step_size(minimum_step_size_), maximum_order(maximum_order_),
            maximum_non_linear_iterations(maximum_non_linear_iterations_), ls_norm_factor(ls_norm_factor_),
            absolute_tolerance(absolute_tolerance_), relative_tolerance(relative_tolerance_),
            ignore_algebraic_terms_for_errors(ignore_algebraic_terms_for_errors_), ic_type(ic_type_),
            reset_type(reset_type_), maximum_num_steps(maximum_num_steps_)
      {}

      double initial_time;
      double final_time;
      double initial_step_size;
      double output_period;
      double minimum_step_size;
      unsigned int maximum_order;
      unsigned int maximum_non_linear_iterations;
      double ls_norm_factor;
      double absolute_tolerance;
      double relative_tolerance;
      bool ignore_algebraic_terms_for_errors;
      InitialConditionCorrection ic_type;
      InitialConditionCorrection reset_type;
      unsigned int maximum_num_steps;
    };

    explicit IDA(const AdditionalData &data_ = AdditionalData()) : data(data_) {}

    std::function<void(VectorType &)> reinit_vector;
    std::function<int(double, const VectorType &, const VectorType &, VectorType &)> residual;
    std::function<int(double, const VectorType &, const VectorType &, double)> setup_jacobian;
    std::function<int(const VectorType &, VectorType &, double)> solve_with_jacobian;
    std::function<void(double, const VectorType &, const VectorType &, unsigned int)> output_step;
    std::function<bool(double, VectorType &, VectorType &)> solver_should_restart;
    std::function<dealii::IndexSet()> differential_components;

    void solve_dae(VectorType &y, VectorType &y_dot)
    {
      initialize(data.initial_time, y, y_dot, data.ic_type);

      VectorType residual_probe;
      reinit_vector(residual_probe);
      check_callback(residual(data.initial_time, y, y_dot, residual_probe), "IDA residual");

      double current_time = data.initial_time;
      unsigned int step_number = 0;
      if (output_step) output_step(current_time, y, y_dot, step_number++);

      while (current_time < data.final_time) {
        const double target_time =
            std::min(data.final_time, current_time + (data.output_period > 0. ? data.output_period : data.final_time));
        double reached_time = current_time;
        const int flag = IDASolve(ida_mem.get(), target_time, &reached_time, y_vec.get(), y_dot_vec.get(), IDA_NORMAL);
        check_sundials_callback(flag, "IDASolve");
        current_time = reached_time;

        if (solver_should_restart && solver_should_restart(current_time, y, y_dot))
          initialize(current_time, y, y_dot, data.reset_type);

        check_callback(residual(current_time, y, y_dot, residual_probe), "IDA residual");
        if (output_step) output_step(current_time, y, y_dot, step_number++);
      }
    }

  private:
    static int residual_callback(sunrealtype t, N_Vector y, N_Vector y_dot, N_Vector res, void *user_data)
    {
      auto &self = *static_cast<IDA *>(user_data);
      try {
        return self.residual(t, unwrap_const<VectorType>(y), unwrap_const<VectorType>(y_dot), unwrap<VectorType>(res));
      } catch (...) {
        return callback_failed(self.pending_exception);
      }
    }

    static int jacobian_callback(sunrealtype t, sunrealtype alpha, N_Vector y, N_Vector y_dot, N_Vector, SUNMatrix, void *user_data,
                                 N_Vector, N_Vector, N_Vector)
    {
      auto &self = *static_cast<IDA *>(user_data);
      try {
        return self.setup_jacobian(t, unwrap_const<VectorType>(y), unwrap_const<VectorType>(y_dot), alpha);
      } catch (...) {
        return callback_failed(self.pending_exception);
      }
    }

    void initialize(const double time, VectorType &y, VectorType &y_dot,
                    const typename AdditionalData::InitialConditionCorrection correction)
    {
      if (!reinit_vector || !residual || !setup_jacobian || !solve_with_jacobian)
        throw std::runtime_error("DiFfRG::sundials::IDA callbacks are incomplete");

      pending_exception = nullptr;
      ida_mem = nullptr;
      matrix.reset();
      linear_solver.reset();
      id_vec.reset();
      y_vec.reset();
      y_dot_vec.reset();

      ida_mem.reset(IDACreate(context.get()));
      if (ida_mem == nullptr) throw std::bad_alloc();

      y_vec.reset(create_view(y, context.get()));
      y_dot_vec.reset(create_view(y_dot, context.get()));

      check_flag(IDAInit(ida_mem.get(), residual_callback, time, y_vec.get(), y_dot_vec.get()), "IDAInit");
      check_flag(IDASetUserData(ida_mem.get(), this), "IDASetUserData");
      check_flag(IDASStolerances(ida_mem.get(), data.relative_tolerance, data.absolute_tolerance), "IDASStolerances");
      check_flag(IDASetMaxOrd(ida_mem.get(), static_cast<int>(data.maximum_order)), "IDASetMaxOrd");
      check_flag(IDASetMaxNumSteps(ida_mem.get(), static_cast<long int>(data.maximum_num_steps)), "IDASetMaxNumSteps");
      check_flag(IDASetMaxNonlinIters(ida_mem.get(), static_cast<int>(data.maximum_non_linear_iterations)),
                 "IDASetMaxNonlinIters");
      check_flag(IDASetStopTime(ida_mem.get(), data.final_time), "IDASetStopTime");
      if (correction != AdditionalData::none || data.ignore_algebraic_terms_for_errors) {
        auto ids = std::make_unique<VectorType>();
        reinit_vector(*ids);
        *ids = differential_components ? 0. : 1.;
        if (differential_components) {
          const auto differential = differential_components();
          for (unsigned int i = 0; i < ids->size(); ++i)
            if (differential.is_element(i)) (*ids)[i] = 1.;
        }
        id_vec.reset(create_owned(std::move(ids), context.get()));
        check_flag(IDASetId(ida_mem.get(), id_vec.get()), "IDASetId");
      }
      if (data.initial_step_size > 0.) check_flag(IDASetInitStep(ida_mem.get(), data.initial_step_size), "IDASetInitStep");
      if (data.minimum_step_size > 0.) check_flag(IDASetMinStep(ida_mem.get(), data.minimum_step_size), "IDASetMinStep");
      if (data.ignore_algebraic_terms_for_errors) check_flag(IDASetSuppressAlg(ida_mem.get(), SUNTRUE), "IDASetSuppressAlg");

      matrix.reset(create_matrix(context.get()));
      linear_solver.reset(create_linear_solver<VectorType>(solve_with_jacobian, pending_exception, context.get()));
      check_flag(IDASetLinearSolver(ida_mem.get(), linear_solver.get(), matrix.get()), "IDASetLinearSolver");
      check_flag(IDASetJacFn(ida_mem.get(), jacobian_callback), "IDASetJacFn");
      if (data.ls_norm_factor > 0.) check_flag(IDASetLSNormFactor(ida_mem.get(), data.ls_norm_factor), "IDASetLSNormFactor");

      if (correction != AdditionalData::none) {
        const double tout = time + std::max(data.initial_step_size, data.output_period);
        check_sundials_callback(IDACalcIC(ida_mem.get(), static_cast<int>(correction), tout), "IDACalcIC");
        check_sundials_callback(IDAGetConsistentIC(ida_mem.get(), y_vec.get(), y_dot_vec.get()), "IDAGetConsistentIC");
      }
    }

    void check_sundials_callback(const int flag, const char *function)
    {
      rethrow_pending(pending_exception);
      check_flag(flag, function);
    }

    void check_callback(const int flag, const char *function)
    {
      rethrow_pending(pending_exception);
      if (flag < 0) throw std::runtime_error(std::string(function) + " failed unrecoverably");
    }

    struct IDADeleter {
      void operator()(void *mem) const
      {
        if (mem != nullptr) IDAFree(&mem);
      }
    };

    AdditionalData data;
    Context context;
    NVectorHandle y_vec;
    NVectorHandle y_dot_vec;
    NVectorHandle id_vec;
    MatrixHandle matrix;
    LinearSolverHandle linear_solver;
    std::unique_ptr<void, IDADeleter> ida_mem;
    std::exception_ptr pending_exception;
  };
} // namespace DiFfRG::sundials
