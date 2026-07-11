#pragma once

#include <DiFfRG/timestepping/sundials/common.hh>
#include <DiFfRG/timestepping/sundials/nvector.hh>

#include <sundials/sundials_linearsolver.h>

#include <functional>
#include <memory>
#include <new>

namespace DiFfRG::sundials
{
  template <typename VectorType> struct LinearSolverContent {
    std::function<int(const VectorType &, VectorType &, double)> solve;
    std::exception_ptr *pending_exception = nullptr;
    sunindextype last_flag = 0;
  };

  namespace linear_solver_ops
  {
    template <typename VectorType> LinearSolverContent<VectorType> &content(SUNLinearSolver solver)
    {
      return *static_cast<LinearSolverContent<VectorType> *>(solver->content);
    }

    template <typename VectorType> SUNLinearSolver_Type get_type(SUNLinearSolver)
    {
      return SUNLINEARSOLVER_DIRECT;
    }

    template <typename VectorType> SUNLinearSolver_ID get_id(SUNLinearSolver)
    {
      return SUNLINEARSOLVER_CUSTOM;
    }

    template <typename VectorType> SUNErrCode unsupported_atimes(SUNLinearSolver, void *, SUNATimesFn)
    {
      return SUN_SUCCESS;
    }

    template <typename VectorType> SUNErrCode unsupported_preconditioner(SUNLinearSolver, void *, SUNPSetupFn, SUNPSolveFn)
    {
      return SUN_SUCCESS;
    }

    template <typename VectorType> SUNErrCode set_scaling_vectors(SUNLinearSolver, N_Vector, N_Vector)
    {
      return SUN_SUCCESS;
    }

    template <typename VectorType> SUNErrCode set_options(SUNLinearSolver, const char *, const char *, int, char **)
    {
      return SUN_SUCCESS;
    }

    template <typename VectorType> SUNErrCode set_zero_guess(SUNLinearSolver, sunbooleantype)
    {
      return SUN_SUCCESS;
    }

    template <typename VectorType> SUNErrCode initialize(SUNLinearSolver) { return SUN_SUCCESS; }
    template <typename VectorType> int setup(SUNLinearSolver) { return 0; }
    template <typename VectorType> int setup(SUNLinearSolver, SUNMatrix) { return 0; }

    template <typename VectorType> int solve(SUNLinearSolver solver, SUNMatrix, N_Vector x, N_Vector b, sunrealtype tol)
    {
      auto &state = content<VectorType>(solver);
      try {
        const int flag = state.solve(unwrap_const<VectorType>(b), unwrap<VectorType>(x), tol);
        state.last_flag = flag;
        return flag;
      } catch (...) {
        if (state.pending_exception != nullptr) *state.pending_exception = std::current_exception();
        state.last_flag = SUNLS_PSOLVE_FAIL_UNREC;
        return SUNLS_PSOLVE_FAIL_UNREC;
      }
    }

    template <typename VectorType> int num_iters(SUNLinearSolver) { return 1; }
    template <typename VectorType> sunrealtype res_norm(SUNLinearSolver) { return 0.; }

    template <typename VectorType> sunindextype last_flag(SUNLinearSolver solver)
    {
      return content<VectorType>(solver).last_flag;
    }

    template <typename VectorType> SUNErrCode space(SUNLinearSolver, long int *lrw, long int *liw)
    {
      if (lrw != nullptr) *lrw = 0;
      if (liw != nullptr) *liw = 0;
      return SUN_SUCCESS;
    }

    template <typename VectorType> N_Vector resid(SUNLinearSolver) { return nullptr; }

    template <typename VectorType> SUNErrCode free(SUNLinearSolver solver)
    {
      delete static_cast<LinearSolverContent<VectorType> *>(solver->content);
      SUNLinSolFreeEmpty(solver);
      return SUN_SUCCESS;
    }
  } // namespace linear_solver_ops

  class LinearSolverHandle
  {
  public:
    LinearSolverHandle() = default;
    explicit LinearSolverHandle(SUNLinearSolver solver_) : solver(solver_) {}
    LinearSolverHandle(const LinearSolverHandle &) = delete;
    LinearSolverHandle &operator=(const LinearSolverHandle &) = delete;

    LinearSolverHandle(LinearSolverHandle &&other) noexcept : solver(other.solver) { other.solver = nullptr; }

    LinearSolverHandle &operator=(LinearSolverHandle &&other) noexcept
    {
      if (this != &other) {
        reset();
        solver = other.solver;
        other.solver = nullptr;
      }
      return *this;
    }

    ~LinearSolverHandle() { reset(); }

    SUNLinearSolver get() const { return solver; }

    SUNLinearSolver release() noexcept
    {
      SUNLinearSolver released = solver;
      solver = nullptr;
      return released;
    }

    void reset(SUNLinearSolver replacement = nullptr)
    {
      if (solver != nullptr) SUNLinSolFree(solver);
      solver = replacement;
    }

  private:
    SUNLinearSolver solver = nullptr;
  };

  template <typename VectorType>
  SUNLinearSolver create_linear_solver(std::function<int(const VectorType &, VectorType &, double)> solve,
                                       std::exception_ptr &pending_exception, SUNContext context)
  {
    LinearSolverHandle guard(SUNLinSolNewEmpty(context));
    if (guard.get() == nullptr) throw std::bad_alloc();
    auto content =
        std::make_unique<LinearSolverContent<VectorType>>(LinearSolverContent<VectorType>{std::move(solve), &pending_exception, 0});
    guard.get()->content = content.release();
    guard.get()->ops->gettype = linear_solver_ops::get_type<VectorType>;
    guard.get()->ops->getid = linear_solver_ops::get_id<VectorType>;
    guard.get()->ops->setatimes = linear_solver_ops::unsupported_atimes<VectorType>;
    guard.get()->ops->setpreconditioner = linear_solver_ops::unsupported_preconditioner<VectorType>;
    guard.get()->ops->setscalingvectors = linear_solver_ops::set_scaling_vectors<VectorType>;
    guard.get()->ops->setoptions = linear_solver_ops::set_options<VectorType>;
    guard.get()->ops->setzeroguess = linear_solver_ops::set_zero_guess<VectorType>;
    guard.get()->ops->initialize = linear_solver_ops::initialize<VectorType>;
    guard.get()->ops->setup = linear_solver_ops::setup<VectorType>;
    guard.get()->ops->solve = linear_solver_ops::solve<VectorType>;
    guard.get()->ops->numiters = linear_solver_ops::num_iters<VectorType>;
    guard.get()->ops->resnorm = linear_solver_ops::res_norm<VectorType>;
    guard.get()->ops->lastflag = linear_solver_ops::last_flag<VectorType>;
    guard.get()->ops->space = linear_solver_ops::space<VectorType>;
    guard.get()->ops->resid = linear_solver_ops::resid<VectorType>;
    guard.get()->ops->free = linear_solver_ops::free<VectorType>;
    return guard.release();
  }
} // namespace DiFfRG::sundials
