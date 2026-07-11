#pragma once

#include <DiFfRG/timestepping/sundials/common.hh>
#include <DiFfRG/timestepping/sundials/nvector.hh>
#include <DiFfRG/timestepping/sundials/sun_linear_solver.hh>
#include <DiFfRG/timestepping/sundials/sun_matrix.hh>

#include <kinsol/kinsol.h>
#include <kinsol/kinsol_ls.h>

#include <functional>
#include <memory>

namespace DiFfRG::sundials
{
  template <typename VectorType> class KINSOL
  {
  public:
    class AdditionalData
    {
    public:
      enum SolutionStrategy
      {
        newton = KIN_NONE,
        linesearch = KIN_LINESEARCH,
        fixed_point = KIN_FP,
        picard = KIN_PICARD,
      };

      SolutionStrategy strategy = linesearch;
      unsigned int maximum_non_linear_iterations = 200;
      double function_tolerance = 0.;
      double step_tolerance = 0.;
      bool no_init_setup = false;
      unsigned int maximum_setup_calls = 0;
      double maximum_newton_step = 0.;
    };

    explicit KINSOL(const AdditionalData &data_ = AdditionalData()) : data(data_) {}

    std::function<void(VectorType &)> reinit_vector;
    std::function<int(const VectorType &, VectorType &)> residual;
    std::function<int(const VectorType &, const VectorType &)> setup_jacobian;
    std::function<int(const VectorType &, VectorType &, double)> solve_with_jacobian;

    void solve(VectorType &iterate)
    {
      if (!reinit_vector || !residual || !solve_with_jacobian)
        throw std::runtime_error("DiFfRG::sundials::KINSOL callbacks are incomplete");

      kin_mem = nullptr;
      iterate_vec.reset();
      scale_vec.reset();
      matrix.reset();
      linear_solver.reset();

      kin_mem.reset(KINCreate(context.get()));
      if (kin_mem == nullptr) throw std::bad_alloc();

      iterate_vec.reset(create_view(iterate, context.get()));
      auto scale = std::make_unique<VectorType>();
      reinit_vector(*scale);
      *scale = 1.;
      scale_vec.reset(create_owned(std::move(scale), context.get()));

      check_flag(KINInit(kin_mem.get(), residual_callback, iterate_vec.get()), "KINInit");
      check_flag(KINSetUserData(kin_mem.get(), this), "KINSetUserData");
      check_flag(KINSetNumMaxIters(kin_mem.get(), static_cast<long int>(data.maximum_non_linear_iterations)),
                 "KINSetNumMaxIters");
      check_flag(KINSetNoInitSetup(kin_mem.get(), data.no_init_setup ? SUNTRUE : SUNFALSE), "KINSetNoInitSetup");
      if (data.maximum_setup_calls > 0)
        check_flag(KINSetMaxSetupCalls(kin_mem.get(), static_cast<long int>(data.maximum_setup_calls)),
                   "KINSetMaxSetupCalls");
      if (data.maximum_newton_step > 0.)
        check_flag(KINSetMaxNewtonStep(kin_mem.get(), data.maximum_newton_step), "KINSetMaxNewtonStep");
      if (data.function_tolerance > 0.) check_flag(KINSetFuncNormTol(kin_mem.get(), data.function_tolerance), "KINSetFuncNormTol");
      if (data.step_tolerance > 0.) check_flag(KINSetScaledStepTol(kin_mem.get(), data.step_tolerance), "KINSetScaledStepTol");

      matrix.reset(create_matrix(context.get()));
      linear_solver.reset(create_linear_solver<VectorType>(solve_with_jacobian, pending_exception, context.get()));
      check_flag(KINSetLinearSolver(kin_mem.get(), linear_solver.get(), matrix.get()), "KINSetLinearSolver");
      if (setup_jacobian) check_flag(KINSetJacFn(kin_mem.get(), jacobian_callback), "KINSetJacFn");

      const int flag =
          KINSol(kin_mem.get(), iterate_vec.get(), static_cast<int>(data.strategy), scale_vec.get(), scale_vec.get());
      rethrow_pending(pending_exception);
      check_solve_flag(flag, "KINSol");
    }

  private:
    static int residual_callback(N_Vector u, N_Vector f, void *user_data)
    {
      auto &self = *static_cast<KINSOL *>(user_data);
      try {
        return self.residual(unwrap_const<VectorType>(u), unwrap<VectorType>(f));
      } catch (...) {
        return callback_failed(self.pending_exception);
      }
    }

    static int jacobian_callback(N_Vector u, N_Vector f, SUNMatrix, void *user_data, N_Vector, N_Vector)
    {
      auto &self = *static_cast<KINSOL *>(user_data);
      try {
        return self.setup_jacobian(unwrap_const<VectorType>(u), unwrap_const<VectorType>(f));
      } catch (...) {
        return callback_failed(self.pending_exception);
      }
    }

    struct KINDeleter {
      void operator()(void *mem) const
      {
        if (mem != nullptr) KINFree(&mem);
      }
    };

    AdditionalData data;
    Context context;
    NVectorHandle iterate_vec;
    NVectorHandle scale_vec;
    MatrixHandle matrix;
    LinearSolverHandle linear_solver;
    std::unique_ptr<void, KINDeleter> kin_mem;
    std::exception_ptr pending_exception;
  };
} // namespace DiFfRG::sundials
