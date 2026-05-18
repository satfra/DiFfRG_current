// external libraries
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <boost/numeric/odeint/stepper/adams_bashforth.hpp>
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/sundials/ida.h>
#include <algorithm>
#include <cstddef>
#include <limits>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/sundials_ida_boost_abm.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      AbstractFlowingVariables<NumberType> *initial_condition, const double t_start, const double t_stop)
  {
    this->data_out = this->get_data_out();
    this->adaptor = this->get_adaptor();

    auto &full_data = initial_condition->data();
    if (full_data.n_blocks() == 2)
      run(full_data, t_start, t_stop);
    else
      throw std::runtime_error(
          "TimeStepperSUNDIALS_IDA_BoostABM::run: initial condition must have exactly two blocks!");
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA_BoostABM<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      BlockVectorType &initial_data, const double t_start, const double t_stop)
  {
    if (initial_data.n_blocks() != 2)
      throw std::runtime_error("TimeStepperSUNDIALS_BoostABM::run: y must have two blocks!");
    if (initial_data.block(1).size() == 0)
      throw std::runtime_error(
          "TimeStepperSUNDIALS_BoostABM::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    SparseMatrixType spatial_jacobian(assembler->get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    const uint n_FE_dofs = initial_data.block(0).size();

    // Create a SUNDIALS IDA object with the right settings for spatial data
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                1e6, 0, impl.abs_tol, impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Initialize initial condition
    VectorType spatial_y = initial_data.block(0);
    VectorType spatial_y_dot = initial_data.block(0);
    VectorType variable_y = initial_data.block(1);

    // Initialize initial condition in eigen
    Eigen::VectorXd variable_y_eigen(variable_y.size());
    dealii_to_eigen(variable_y, variable_y_eigen);

    // These are just buffers
    dealii::Vector<double> variable_y_dealii(variable_y.size());
    dealii::Vector<double> spatial_y_dealii(spatial_y.size());
    dealii::Vector<double> variable_dy_dealii(variable_y.size());

    // The explicit variables are integrated "on demand" with the spatial solution held
    // fixed over each IDA trial step. Rather than pinning it to the step's right endpoint
    // -- which biases the explicit variable, because the coupling then always lags or
    // leads the true spatial trajectory -- the spatial solution is linearly interpolated
    // across the step between these two endpoints: spatial_lo at spatial_lo_time (the
    // start of the segment) and spatial_hi at spatial_hi_time (its end).
    dealii::Vector<double> spatial_lo(spatial_y.size());
    dealii::Vector<double> spatial_hi(spatial_y.size());
    double spatial_lo_time = t_start;
    double spatial_hi_time = t_start;

    using namespace boost::numeric::odeint;

    constexpr size_t Steps = 8;
    using State = Eigen::VectorXd;
    using Value = double;
    using Deriv = State;
    using Time = Value;
    using Algebra = algebra_dispatcher<State>::algebra_type;
    using Operations = operations_dispatcher<State>::operations_type;
    using Resizer = initially_resizer;

    using stepper_type_0 = boost::numeric::odeint::runge_kutta_cash_karp54<State>;
    using stepper_type_1 = boost::numeric::odeint::runge_kutta_fehlberg78<State>;

    using InitializingStepper = stepper_type_0;

    adams_bashforth_moulton<Steps, State, Value, Deriv, Time, Algebra, Operations, Resizer, InitializingStepper>
        variable_stepper;

    auto get_variable_residual = [&](const Eigen::VectorXd &x, Eigen::VectorXd &dxdt, const double t) {
      const auto now = std::chrono::high_resolution_clock::now();

      eigen_to_dealii(x, variable_y_dealii);

      variable_dy_dealii = 0;

      // linearly interpolate the spatial solution across the current segment so the
      // coupling tracks the spatial trajectory instead of being pinned to one endpoint
      double alpha = (spatial_hi_time > spatial_lo_time)
                         ? (t - spatial_lo_time) / (spatial_hi_time - spatial_lo_time)
                         : 1.;
      alpha = std::clamp(alpha, 0., 1.);
      spatial_y_dealii = spatial_lo;
      spatial_y_dealii *= (1. - alpha);
      spatial_y_dealii.add(alpha, spatial_hi);

      assembler->set_time(t);
      assembler->residual_variables(variable_dy_dealii, variable_y_dealii, spatial_y_dealii);

      if (!std::isfinite(variable_dy_dealii.l2_norm()))
        throw std::runtime_error("TimeStepperBoostABM::run_vars: dy is not finite!");

      dealii_to_eigen(variable_dy_dealii, dxdt);
      dxdt *= -1;

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "explicit residual", 1, ms_passed);
    };

    Eigen::VectorXd variable_sol;
    Eigen::VectorXd variable_ret;
    // Trajectory of the explicit variables, sampled at the times the explicit stepper
    // visited. All entries are committed -- they correspond to IDA trial steps that have
    // been accepted. No speculative entries are ever appended, so a rejected trial step
    // cannot pollute the buffer.
    std::vector<Eigen::VectorXd> variable_buffer;
    std::vector<double> variable_buffer_times;
    // End time of the last accepted IDA trial step that has been *folded into the
    // committed buffer*. The committed segments cover [t_start, t_committed].
    double t_committed = t_start;
    // Latest IDA trial endpoint that has been confirmed accepted (by IDA later
    // requesting a strictly larger time) but not yet folded into the committed buffer.
    // We defer running an ABM substep until the gap (t_pending - t_committed) is at
    // least cur_dt -- otherwise marginal points where IDA collapses dt to << cur_dt
    // would trigger one full ABM substep + dt_variables evaluation per IDA step, which
    // is far too expensive (dt_variables is the dominant cost in production FRG runs).
    double t_pending = t_start;
    dealii::Vector<double> spatial_pending(spatial_y.size());
    // End time of the IDA trial step the controller is currently solving. A later
    // request at a strictly larger time means that trial step was accepted (the trial's
    // converged spatial state is then transferred to spatial_pending); a request at a
    // smaller time means IDA rejected it (spatial_pending is left untouched).
    double frontier = -std::numeric_limits<double>::infinity();
    // Cached values of dv/dt at the two most recent committed points. These drive a
    // quadratic predictor used to serve `v` to IDA during Newton iterations on a
    // speculative trial step:
    //
    //   v(t) ~ v_committed + dv_committed * (t - t_committed)
    //                       + 1/2 * d2v_committed * (t - t_committed)^2
    //
    //   d2v_committed = (dv_committed - dv_prev) / (t_committed - t_prev_committed)
    //
    // dv_committed is estimated by backward-differencing the two most recent buffer
    // entries (no extra dt_variables call is required); dv_prev_committed is the
    // dv_committed value cached one commit ago. Before the second commit, only the
    // linear term is used. The predictor is deterministic (it does not depend on IDA's
    // still-unconverged trial spatial guess), which is what Newton needs to converge
    // consistently when the FEM equation reads `v` back through an extractor.
    Eigen::VectorXd dv_committed;
    Eigen::VectorXd dv_prev_committed;
    double t_prev_committed = -std::numeric_limits<double>::infinity();
    bool have_prev_dv = false;
    double cur_dt = expl.dt;

    // Linearly interpolate the buffered trajectory to time t (clamped to the buffer range).
    auto interpolate_buffer = [&](VectorType &variable_y, const double t) {
      if (t <= variable_buffer_times.front()) {
        eigen_to_dealii(variable_buffer.front(), variable_y);
        return;
      }
      if (t >= variable_buffer_times.back()) {
        eigen_to_dealii(variable_buffer.back(), variable_y);
        return;
      }
      std::size_t idx = 0;
      for (std::size_t i = 0; i + 1 < variable_buffer_times.size(); ++i)
        if (variable_buffer_times[i] <= t && t <= variable_buffer_times[i + 1]) idx = i;
      const double t_prev = variable_buffer_times[idx];
      const double t_next = variable_buffer_times[idx + 1];
      const double alpha = (t - t_prev) / (t_next - t_prev);
      variable_ret = alpha * variable_buffer[idx + 1] + (1. - alpha) * variable_buffer[idx];
      eigen_to_dealii(variable_ret, variable_y);
    };

    // Refresh the predictor's cached dv/dt by evaluating the variable residual at the
    // current committed point. Costs ONE dt_variables call per commit (commits happen
    // once per cur_dt of accepted t under the batching scheme, so this is not per-IDA-
    // step). A pure-finite-difference alternative was tried but it lags by ~cur_dt/2
    // and cannot follow sharp curvature in v -- destabilising the FEM Newton solve in
    // regions where v changes rapidly. Reducing cur_dt does not fix this lag.
    //
    // Promotes the previous dv_committed value to dv_prev_committed so the next
    // predictor can use the slope of dv/dt for a quadratic extrapolation.
    auto refresh_dv_committed_from_buffer = [&]() {
      if (dv_committed.size() > 0) {
        if (dv_prev_committed.size() != dv_committed.size()) dv_prev_committed.resize(dv_committed.size());
        dv_prev_committed = dv_committed;
        t_prev_committed = t_committed;  // t_committed has just been updated by the caller
        have_prev_dv = true;
      }
      const std::size_t N = variable_buffer.size();
      if (N == 0) return;
      eigen_to_dealii(variable_buffer[N - 1], variable_y_dealii);
      variable_dy_dealii = 0;
      spatial_y_dealii = spatial_lo;  // = accepted spatial at t_committed
      assembler->set_time(t_committed);
      assembler->residual_variables(variable_dy_dealii, variable_y_dealii, spatial_y_dealii);
      if (dv_committed.size() != static_cast<Eigen::Index>(variable_dy_dealii.size()))
        dv_committed.resize(variable_dy_dealii.size());
      dealii_to_eigen(variable_dy_dealii, dv_committed);
      dv_committed *= -1.;
    };

    // Quadratic predictor at time t (>= t_committed). Falls back to linear before the
    // first second-derivative estimate is available.
    auto eval_predictor = [&](Eigen::VectorXd &out, const double t) {
      out = variable_buffer.back();
      const double dt = t - t_committed;
      out += dt * dv_committed;
      if (have_prev_dv && t_committed > t_prev_committed) {
        const double inv_dtc = 1. / (t_committed - t_prev_committed);
        // d2v ~ (dv_c - dv_prev) / (t_c - t_prev)
        out += (0.5 * dt * dt * inv_dtc) * (dv_committed - dv_prev_committed);
      }
    };

    // Re-integrate the explicit variables forward from the last committed time to time
    // `t` -- which is required to be the endpoint of an *accepted* IDA trial step --
    // using `accepted_spatial` as the spatial solution at `t`. The spatial solution is
    // linearly interpolated across the segment between `spatial_lo` (at `t_committed`)
    // and `accepted_spatial` (at `t`), both of which are converged values, so the
    // explicit integration sees a faithful spatial trajectory and never speculative
    // intermediate Newton iterates.
    auto commit_segment_to = [&](const dealii::Vector<double> &accepted_spatial, const double t) {
      spatial_hi = accepted_spatial;
      spatial_hi_time = t;
      // spatial_lo / spatial_lo_time were left at the previous committed point and are
      // the correct left endpoint for this segment.
      variable_sol = variable_buffer.back();
      double step_time = variable_buffer_times.back();
      // The multistep history is stitched across consecutive accepted segments: the
      // residual function is the same lambda and the segments are contiguous in time, so
      // ABM keeps its design order. The history is only reset on a rollback path (which
      // we never exercise here -- rejected trials never make it into the buffer).
      while (step_time < t && !is_close(step_time, t)) {
        variable_stepper.do_step(get_variable_residual, variable_sol, step_time, cur_dt);
        step_time += cur_dt;
        variable_buffer.push_back(variable_sol);
        variable_buffer_times.push_back(step_time);
      }
      // Slide the committed point forward
      t_committed = t;
      spatial_lo = accepted_spatial;
      spatial_lo_time = t;
      // Refresh the predictor used by the next speculative Newton solve. This costs
      // zero dt_variables calls -- the slope is read off the buffer.
      refresh_dv_committed_from_buffer();
    };

    // Serve the explicit variables at time t for a speculative IDA call (residual /
    // jacobian). The trial endpoint `frontier` is only committed once IDA confirms its
    // acceptance (signalled by a later request at a strictly larger time, in which case
    // the saved `spatial_hi` is the converged spatial state at `frontier`). Within a
    // single still-unaccepted Newton solve we serve a first-order predictor in v built
    // entirely from the last committed state -- this makes the FEM residual a
    // deterministic function of IDA's spatial trial state, which is required for Newton
    // convergence when the FEM source reads v through an extractor.
    auto request_variables = [&](VectorType &variable_y, const VectorType &spatial_y, const double t) {
      if (variable_buffer.empty()) {
        // Initialise at t = t_start; the initial condition is by construction accepted.
        variable_y = initial_data.block(1);
        dealii_to_eigen(variable_y, variable_sol);
        variable_buffer.push_back(variable_sol);
        variable_buffer_times.push_back(t);
        t_committed = t;
        t_pending = t;
        frontier = t;
        spatial_lo = spatial_y;
        spatial_hi = spatial_y;
        spatial_pending = spatial_y;
        spatial_lo_time = t;
        spatial_hi_time = t;
        // Seed the predictor with a real dv estimate at the initial condition; this is
        // the only dt_variables call that does not double up with one inside an ABM
        // substep.
        refresh_dv_committed_from_buffer();
        assembler->set_time(t);
        return;
      }

      if (t > frontier && !is_close(t, frontier)) {
        // IDA advanced -> the previous trial at `frontier` was accepted. spatial_hi
        // (saved from the last call at `frontier`) is the converged spatial state
        // there; commit the segment immediately.
        if (frontier > t_committed && !is_close(frontier, t_committed))
          commit_segment_to(spatial_hi, frontier);
        // Move on to the new trial endpoint; its spatial state is still speculative.
        frontier = t;
        spatial_hi = spatial_y;
        spatial_hi_time = t;
      } else if (t < frontier && !is_close(t, frontier)) {
        // IDA rejected the previous trial at `frontier`; the speculative spatial_hi is
        // discarded and we restart at a smaller trial endpoint.
        frontier = t;
        spatial_hi = spatial_y;
        spatial_hi_time = t;
      } else {
        // Another residual / jacobian evaluation at the same trial endpoint; track the
        // latest spatial guess so that, if this trial is later accepted, the eventual
        // commit uses the converged spatial state.
        spatial_hi = spatial_y;
      }

      // Quadratic predictor (linear before the first d2v estimate is available). The
      // result is independent of `spatial_y`, so repeated Newton iterations at the same
      // trial endpoint (and even retries after rejection) all see the same v(t).
      eval_predictor(variable_ret, t);
      eigen_to_dealii(variable_ret, variable_y);
      assembler->set_time(t);
    };

    // Sample the explicit variables at an accepted time t (called from output_step and
    // at the end of the run). Any pending accepted-but-not-yet-committed segment is
    // folded into the committed buffer first regardless of the cur_dt threshold; the
    // buffer is then extended to t using the converged spatial state passed in.
    auto commit_variables = [&](VectorType &variable_y, const VectorType &spatial_y, const double t) {
      if (variable_buffer.empty()) {
        request_variables(variable_y, spatial_y, t);
        return;
      }
      if (frontier > t_committed && !is_close(frontier, t_committed))
        commit_segment_to(spatial_hi, frontier);
      if (t > t_committed && !is_close(t, t_committed)) {
        commit_segment_to(spatial_y, t);
        if (t > frontier) frontier = t;
      }
      interpolate_buffer(variable_y, t);
      assembler->set_time(t);
    };

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = 0.;
    uint failure_counter = 0;

    // Pointer to current residual for monitoring
    VectorType *residual;

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement
    time_stepper.solver_should_restart = [&](const double t, VectorType &sol, VectorType &sol_dot) -> bool {
      if ((*adaptor)(t, sol)) {
        assembler->reinit_vector(sol_dot);
        spatial_jacobian.reinit(assembler->get_sparsity_pattern_jacobian());
        return true;
      }
      return false;
    };

    time_stepper.differential_components = [&]() {
      IndexSet dof_indices = assembler->get_differential_indices();
      IndexSet differential_indices(n_FE_dofs);
      differential_indices.add_indices(dof_indices.begin(), dof_indices.end());
      return differential_indices;
    };

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](VectorType &v) { assembler->reinit_vector(v); };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    time_stepper.output_step = [&](const double t, const VectorType &sol, const VectorType &sol_dot,
                                   uint /*step_number*/) {
      if (t < t_start) return;
      if (!is_close(last_save, t, 1e-10)) {
        assembler->set_time(t);
        commit_variables(variable_y, sol, t);
        assembler->attach_data_output(*data_out, sol, variable_y, sol_dot, (*residual));
        data_out->flush(t);

        last_save = t;
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      const auto now = std::chrono::high_resolution_clock::now();

      if (is_close(t, stuck_t, impl.minimal_dt * 1e-1))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (stuck > 200) throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (failure_counter > 200) {
        std::cerr << "timestep failure, at t = " << t << std::endl;
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!std::isfinite(y.l1_norm())) {
        std::cerr << "residual: spatial solution is not finite!" << std::endl;
        return ++failure_counter;
      }

      try {
        res = 0;
        assembler->set_time(t);
        console_out(t, "requesting variables", 2);
        request_variables(variable_y, y, t);
        assembler->residual(res, y, 1., y_dot, 1., variable_y);
        residual = &res;
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return ++failure_counter;
      }

      if (!std::isfinite(res.l1_norm())) {
        std::cerr << "residual: spatial residual is not finite!" << std::endl;
        return ++failure_counter;
      }

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "implicit residual", 1, ms_passed);

      failure_counter = 0;
      return 0;
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const VectorType &y, const VectorType &y_dot,
                                      const double alpha) -> int {
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      try {
        auto now = std::chrono::high_resolution_clock::now();

        spatial_jacobian = 0;
        assembler->set_time(t);
        console_out(t, "requesting variables", 2);
        request_variables(variable_y, y, t);
        assembler->jacobian(spatial_jacobian, y, 1., y_dot, alpha, 1., variable_y);
        linSolver.init(spatial_jacobian);

        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(t, "jacobian construction", 1, ms_passed);
        now = std::chrono::high_resolution_clock::now();

        if (linSolver.invert()) {
          const auto ms_passed =
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                  .count();
          console_out(t, "jacobian inversion", 2, ms_passed);
        }
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return ++failure_counter;
      }

      failure_counter = 0;
      return 0;
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double tol) -> int {
      try {
        const auto now = std::chrono::high_resolution_clock::now();
        const auto sol_iterations = linSolver.solve(src, dst, tol);
        if (sol_iterations >= 0) {
          const auto ms_passed =
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                  .count();
          console_out(stuck_t, "linear solver (" + std::to_string(sol_iterations) + " it)", 2, ms_passed);
        }
      } catch (std::exception &) {
        return ++failure_counter;
      }
      return 0;
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(spatial_y, spatial_y_dot);
    } catch (const std::exception &e) {
      spdlog::get("log")->error("Timestepping failed: {}", e.what());
      time_stepper.output_step(stuck_t, spatial_y, spatial_y_dot, 0);
      throw;
    }

    initial_data.block(0) = spatial_y;
    commit_variables(initial_data.block(1), spatial_y, t_stop);
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                        DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                        DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                        DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                        DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                        DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                        DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                        DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                        DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                        DiFfRG::GMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                        DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                        DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostABM<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                        DiFfRG::GMRES>;