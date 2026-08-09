#pragma once

#include <deal.II/sundials/ida.h>

#include <DiFfRG/timestepping/abstract_timestepper.hh>
#include <DiFfRG/timestepping/jacobian_diagnostics.hh>

namespace DiFfRG
{
  struct IDACallbackDiagnostics {
    size_t nonfinite_solution_failures = 0;
    size_t nonfinite_residual_failures = 0;
    size_t residual_exceptions = 0;
    size_t jacobian_failures = 0;
    size_t linear_solver_failures = 0;

    bool has_failures() const
    {
      return nonfinite_solution_failures > 0 || nonfinite_residual_failures > 0 || residual_exceptions > 0 ||
             jacobian_failures > 0 || linear_solver_failures > 0;
    }
  };

  template <typename VectorType>
  TimesteppingDiagnostics make_timestepping_diagnostics(const SUNDIALS::IDA<VectorType> &time_stepper,
                                                        const IDACallbackDiagnostics &callbacks)
  {
    TimesteppingDiagnostics diagnostics;

    if constexpr (requires { time_stepper.get_statistics(); }) {
      const auto ida = time_stepper.get_statistics();
      diagnostics.has_ida = ida.valid;
      diagnostics.ida_steps = ida.num_steps;
      diagnostics.ida_error_test_failures = ida.num_error_test_failures;
      diagnostics.ida_nonlinear_convergence_failures = ida.num_nonlinear_convergence_failures;
      diagnostics.ida_step_solve_failures = ida.num_step_solve_failures;
      diagnostics.ida_residual_evaluations = ida.num_residual_evaluations;
      diagnostics.ida_nonlinear_iterations = ida.num_nonlinear_iterations;
      diagnostics.ida_last_step_size = ida.last_step_size;
      diagnostics.ida_current_step_size = ida.current_step_size;
      diagnostics.ida_current_time = ida.current_time;
    }

    diagnostics.has_callback = callbacks.has_failures();
    diagnostics.nonfinite_solution_failures = callbacks.nonfinite_solution_failures;
    diagnostics.nonfinite_residual_failures = callbacks.nonfinite_residual_failures;
    diagnostics.residual_exceptions = callbacks.residual_exceptions;
    diagnostics.jacobian_failures = callbacks.jacobian_failures;
    diagnostics.linear_solver_failures = callbacks.linear_solver_failures;

    return diagnostics;
  }

  template <typename VectorType>
  TimestepperJacobianBuildDiagnostics
  make_ida_jacobian_build_diagnostics(const SUNDIALS::IDA<VectorType> &time_stepper, const std::size_t build_id,
                                      const double alpha,
                                      const ImplicitTimestepperKind stepper_kind = ImplicitTimestepperKind::ida)
  {
    TimestepperJacobianBuildDiagnostics diagnostics;
    diagnostics.jacobian_build_id = build_id;
    diagnostics.stepper_kind = stepper_kind;
    diagnostics.alpha = alpha;
    diagnostics.beta = 1.;
    diagnostics.residual_weight = 1.;
    if constexpr (requires { time_stepper.get_statistics(); }) {
      const auto ida = time_stepper.get_statistics();
      if (ida.valid) {
        diagnostics.step = static_cast<double>(ida.num_steps);
        diagnostics.ida_step = static_cast<double>(ida.num_steps);
        diagnostics.last_h = ida.last_step_size;
        diagnostics.current_h = ida.current_step_size;
      }
    }
    return diagnostics;
  }
} // namespace DiFfRG
