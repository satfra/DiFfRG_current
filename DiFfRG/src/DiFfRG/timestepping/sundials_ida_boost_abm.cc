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

    // Ring of the most recent *accepted* spatial checkpoints (time, converged state),
    // kept at length <= 3 so the explicit RHS can interpolate the spatial coupling
    // quadratically in time (O(dt^3)) instead of linearly. Bounded memory: at most three
    // FEM vectors. Only converged accepted states are ever stored -- never a speculative
    // Newton trial iterate.
    std::vector<double> spatial_ckpt_times;
    std::vector<dealii::Vector<double>> spatial_ckpts;

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

    // [PREDICT, mode 2] Persistent cache for the forward lookahead: a copy of the multistep
    // stepper plus the lookahead trajectory it has produced beyond the committed buffer
    // front. The cache is valid until the committed buffer / checkpoints change (i.e. until
    // commit_segment_to runs, which happens between IDA steps, never within one Newton
    // solve), so all residual / jacobian evaluations of a step reuse it. This is what turns
    // the lookahead cost from O(extra steps) * (Newton iterations) into ~O(extra steps).
    decltype(variable_stepper) look_stepper;
    std::vector<double> look_times;
    std::vector<Eigen::VectorXd> look_states;
    bool look_valid = false;

    // When true, the spatial coupling read by the explicit RHS is *extrapolated* in time
    // (no clamp into the checkpoint span) rather than interpolated. Used only by the
    // opt-in staggered-lookahead serving below, where the explicit stepper is advanced
    // into not-yet-accepted territory and therefore genuinely needs spatial data beyond
    // the last accepted checkpoint. The committed (accepted-region) integration always
    // runs with this false, i.e. pure interpolation.
    bool spatial_extrapolate = false;

    auto get_variable_residual = [&](const Eigen::VectorXd &x, Eigen::VectorXd &dxdt, const double t) {
      const auto now = std::chrono::high_resolution_clock::now();

      eigen_to_dealii(x, variable_y_dealii);

      variable_dy_dealii = 0;

      // Interpolate the spatial solution in time across the accepted checkpoints so the
      // coupling tracks the spatial trajectory. Quadratic (3-point Lagrange, O(dt^3)) once
      // three checkpoints exist, falling back to linear / constant early on. The query is
      // normally clamped into the stencil span (interpolation only); when
      // spatial_extrapolate is set (staggered-lookahead serving) the clamp is lifted so the
      // explicit step can read the spatial trajectory extrapolated forward to its endpoint.
      const std::size_t nck = spatial_ckpt_times.size();
      if (nck >= 3) {
        const std::size_t i0 = nck - 3, i1 = nck - 2, i2 = nck - 1;
        const double t0 = spatial_ckpt_times[i0], t1 = spatial_ckpt_times[i1], t2 = spatial_ckpt_times[i2];
        const double tc = spatial_extrapolate ? t : std::clamp(t, t0, t2);
        const double L0 = ((tc - t1) * (tc - t2)) / ((t0 - t1) * (t0 - t2));
        const double L1 = ((tc - t0) * (tc - t2)) / ((t1 - t0) * (t1 - t2));
        const double L2 = ((tc - t0) * (tc - t1)) / ((t2 - t0) * (t2 - t1));
        spatial_y_dealii = spatial_ckpts[i0];
        spatial_y_dealii *= L0;
        spatial_y_dealii.add(L1, spatial_ckpts[i1]);
        spatial_y_dealii.add(L2, spatial_ckpts[i2]);
      } else if (nck == 2) {
        const double t0 = spatial_ckpt_times[0], t1 = spatial_ckpt_times[1];
        double alpha = (t1 > t0) ? (t - t0) / (t1 - t0) : 1.;
        if (!spatial_extrapolate) alpha = std::clamp(alpha, 0., 1.);
        spatial_y_dealii = spatial_ckpts[0];
        spatial_y_dealii *= (1. - alpha);
        spatial_y_dealii.add(alpha, spatial_ckpts[1]);
      } else {
        spatial_y_dealii = spatial_ckpts.empty() ? spatial_lo : spatial_ckpts.back();
      }

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
    // Honour the user's explicit-step cap, exactly as the RK path does. Set once and held
    // fixed: the ABM multistep order requires a uniform dt across its history, so cur_dt
    // must never change after this point.
    const double cur_dt = std::min(expl.dt, expl.maximal_dt);

    // How the explicit variable couples to the (implicit) spatial block, via
    // /timestepping/explicit/coupling_mode:
    //   0 LAG             : the explicit buffer lags IDA; the committed/output trajectory
    //                        reads the spatial coupling *interpolated* between accepted
    //                        checkpoints (best one-way accuracy), and v is served to IDA by
    //                        a forward predictor (cheap, but the served v plateaus -> the
    //                        two-way coupling error floors). One set of explicit steps.
    //   1 STAGGER (default): the explicit grid leads the *accepted* region by half a step
    //                        (midpoint-triggered, advance_lead_half); the spatial coupling
    //                        is extrapolated only half an explicit step, and IDA's
    //                        sub-queries past the midpoint are served by interpolation.
    //                        One set of explicit steps (minimal cost); trades a bounded
    //                        half-step extrapolation in the output for an interpolated v.
    //   2 PREDICT (expensive): keeps the LAG committed/output path (interpolated coupling)
    //                        AND additionally serves v to IDA from a separate forward
    //                        lookahead integration interpolated to the query time -- both
    //                        coupling arrows interpolated (best accuracy) at the cost of a
    //                        second set of explicit RHS evaluations.
    const int coupling_mode = json.get_int("/timestepping/explicit/coupling_mode", 1);
    const bool mode_lag = (coupling_mode == 0);
    const bool mode_stagger = (coupling_mode == 1);
    const bool mode_predict = (coupling_mode == 2);
    // Largest IDA time confirmed accepted (drives the half-step trigger in STAGGER mode).
    double t_accepted = t_start;

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

    // [STAGGER, investigation] Re-anchor the forward predictor at the explicit buffer's
    // front, with a slope from backward-differencing the last two buffer points (no extra
    // RHS eval). Needed because mode 1 never calls commit_segment_to, so otherwise the
    // predictor stays frozen at t_start with t_start's slope and is served (stale) for
    // every first-half query.
    auto refresh_predictor_from_buffer_tail = [&]() {
      const std::size_t N = variable_buffer.size();
      if (N < 2) return;
      const double dt = variable_buffer_times[N - 1] - variable_buffer_times[N - 2];
      if (dt <= 0) return;
      t_committed = variable_buffer_times[N - 1];
      if (dv_committed.size() != variable_buffer[N - 1].size()) dv_committed.resize(variable_buffer[N - 1].size());
      dv_committed = (variable_buffer[N - 1] - variable_buffer[N - 2]) / dt;
      have_prev_dv = false; // linear extrapolation off the fresh anchor
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

    // [STAGGER] Advance the REAL explicit buffer forward in whole cur_dt steps -- each
    // taken exactly ONCE over the run -- until its front leads the accepted spatial region
    // (t_accepted) by no more than half a step. Equivalently, the step landing on a grid
    // point t_{k+1} is taken once t_accepted has reached the interval midpoint t_k + dt/2,
    // so the explicit RHS reads the spatial coupling interpolated over [t_k, midpoint] and
    // EXTRAPOLATED only over (midpoint, t_{k+1}] -- a half-step extrapolation. When IDA
    // leaps far ahead (t_accepted >> front) the same loop drains several whole steps whose
    // endpoints lie inside the accepted region, i.e. read the coupling by interpolation:
    // the "explicit catches up" degeneration. Reads only accepted checkpoints (never IDA's
    // trial iterate), so the served value stays a deterministic function of t.
    auto advance_lead_half = [&]() {
      spatial_extrapolate = true;
      double step_time = variable_buffer_times.back();
      variable_sol = variable_buffer.back();
      int guard = 0;
      while (t_accepted + 1e-12 * std::max(1.0, std::abs(t_accepted)) >= step_time + 0.5 * cur_dt &&
             guard++ < 1000000) {
        variable_stepper.do_step(get_variable_residual, variable_sol, step_time, cur_dt);
        step_time += cur_dt;
        variable_buffer.push_back(variable_sol);
        variable_buffer_times.push_back(step_time);
      }
      spatial_extrapolate = false;
    };

    // [STAGGER] Drain whole cur_dt steps until the buffer brackets `target` (used at output
    // / t_stop, where the accepted spatial state at `target` is already a checkpoint, so the
    // coupling is interpolated up to it).
    auto advance_lead_to = [&](const double target) {
      spatial_extrapolate = true;
      double step_time = variable_buffer_times.back();
      variable_sol = variable_buffer.back();
      int guard = 0;
      while (step_time < target && !is_close(step_time, target) && guard++ < 1000000) {
        variable_stepper.do_step(get_variable_residual, variable_sol, step_time, cur_dt);
        step_time += cur_dt;
        variable_buffer.push_back(variable_sol);
        variable_buffer_times.push_back(step_time);
      }
      spatial_extrapolate = false;
    };

    // Push an accepted spatial checkpoint (converged state at an accepted time) onto the
    // ring, keeping at most the three most recent. Consumed by get_variable_residual for
    // the quadratic-in-time spatial interpolation.
    auto push_checkpoint = [&](const double t, const dealii::Vector<double> &S) {
      // Guard against near-duplicate times: two checkpoints closer than ~0.1*cur_dt would
      // make a Lagrange denominator (t_i - t_j) near zero and blow up the quadratic. This
      // happens e.g. when an output-time checkpoint coincides with a recently pushed one.
      // Replace (update) the last checkpoint instead of appending a degenerate stencil node.
      if (!spatial_ckpt_times.empty() && t - spatial_ckpt_times.back() < 0.1 * cur_dt) {
        spatial_ckpt_times.back() = t;
        spatial_ckpts.back() = S;
        return;
      }
      spatial_ckpt_times.push_back(t);
      spatial_ckpts.push_back(S);
      if (spatial_ckpt_times.size() > 3) {
        spatial_ckpt_times.erase(spatial_ckpt_times.begin());
        spatial_ckpts.erase(spatial_ckpts.begin());
      }
    };

    // Push an accepted checkpoint only once its time has advanced by ~cur_dt/2 since the
    // last one, i.e. downsample IDA's acceptances to ~half the EXPLICIT cadence. This keeps
    // the 3-point stencil ~cur_dt wide with its latest point ~half a step behind the
    // explicit front, so the half-step extrapolation in advance_lead_half is a
    // well-conditioned ~half-stencil extrapolation. Without downsampling, when
    // dt_implicit << dt_explicit the stencil would be only ~dt_implicit wide and
    // extrapolating half an explicit step would amplify wildly.
    auto maybe_push_checkpoint = [&](const double t, const dealii::Vector<double> &S) {
      if (spatial_ckpt_times.empty() || t - spatial_ckpt_times.back() >= 0.5 * cur_dt - 1e-12)
        push_checkpoint(t, S);
    };

    // [PREDICT, mode 2] Serve v(t) to IDA from a forward LOOKAHEAD, CACHED per accepted
    // frontier. The committed/output buffer keeps advancing via the LAG path's
    // commit_segment_to (interpolated coupling), so both arrows are interpolated -- best
    // accuracy. The lookahead trajectory (look_times / look_states), produced by a copy of
    // the multistep stepper that reads the spatial coupling extrapolated to each endpoint,
    // is built ONCE per validity epoch and only EXTENDED (never recomputed) as IDA's trial
    // time grows. commit_segment_to invalidates it (look_valid=false) whenever the committed
    // buffer / checkpoints change, which happens between IDA steps, never within a Newton
    // solve -- so every residual / jacobian evaluation of a step reuses the same lookahead.
    // The result depends only on the committed buffer + accepted checkpoints, never on IDA's
    // trial iterate, so it is a deterministic function of t (required for Newton).
    auto serve_lookahead = [&](VectorType &variable_y, const double t) {
      if (t <= variable_buffer_times.back() || is_close(t, variable_buffer_times.back())) {
        interpolate_buffer(variable_y, t);
        return;
      }
      if (variable_buffer.size() < 2) {
        eval_predictor(variable_ret, t);
        eigen_to_dealii(variable_ret, variable_y);
        return;
      }
      // (Re)seed the cache from the current committed front when it has been invalidated.
      if (!look_valid) {
        look_stepper = variable_stepper; // copy the committed multistep history
        look_times.assign(1, variable_buffer_times.back());
        look_states.assign(1, variable_buffer.back());
        look_valid = true;
      }
      // Extend the cached lookahead until it brackets t (no work if it already does).
      if (look_times.back() < t && !is_close(look_times.back(), t)) {
        spatial_extrapolate = true;
        Eigen::VectorXd s = look_states.back();
        double st = look_times.back();
        int guard = 0;
        while (st < t && !is_close(st, t) && guard++ < 100000) {
          look_stepper.do_step(get_variable_residual, s, st, cur_dt);
          st += cur_dt;
          look_states.push_back(s);
          look_times.push_back(st);
        }
        spatial_extrapolate = false;
      }
      // Interpolate the cached lookahead at t.
      std::size_t idx = look_times.size() - 2;
      for (std::size_t i = 0; i + 1 < look_times.size(); ++i)
        if (look_times[i] <= t && t <= look_times[i + 1]) idx = i;
      const double t0 = look_times[idx], t1 = look_times[idx + 1];
      const double a = (t1 > t0) ? std::clamp((t - t0) / (t1 - t0), 0., 1.) : 1.;
      variable_ret = (1. - a) * look_states[idx] + a * look_states[idx + 1];
      eigen_to_dealii(variable_ret, variable_y);
    };

    // Advance the ABM integrator in uniform cur_dt steps while a *whole* step still fits
    // inside [step_time, limit]. Never overshoots: the trailing gap (< cur_dt) is left to
    // be served by interpolation/prediction. Every do_step uses the same cur_dt and the
    // same RHS lambda, so the multistep history stays contiguous at uniform step size --
    // which is what keeps ABM at its design order. No reset() on this (accepted) path.
    auto advance_abm_to = [&](const double limit) {
      double step_time = variable_buffer_times.back();
      variable_sol = variable_buffer.back();
      while (step_time + cur_dt <= limit + 1e-12 * std::max(1.0, std::abs(limit))) {
        variable_stepper.do_step(get_variable_residual, variable_sol, step_time, cur_dt);
        step_time += cur_dt;
        variable_buffer.push_back(variable_sol);
        variable_buffer_times.push_back(step_time);
      }
    };

    // Extend the accepted (pending) window to time `t` -- the endpoint of an *accepted*
    // IDA trial step, with `accepted_spatial` its converged spatial state -- and, once at
    // least one whole cur_dt fits, drain uniform ABM steps across it. The committed point
    // is slid to the last ABM grid time actually produced (NOT the raw IDA endpoint `t`),
    // which keeps the invariant t_committed == variable_buffer_times.back() and stops the
    // explicit trajectory from desyncing or freezing relative to IDA. While the window is
    // still shorter than cur_dt nothing is appended: accepted time simply accumulates over
    // successive calls -- the documented `t_pending` batching -- until a full step fits.
    auto commit_segment_to = [&](const dealii::Vector<double> &accepted_spatial, const double t) {
      t_pending = t;
      spatial_pending = accepted_spatial;
      spatial_hi = accepted_spatial;
      spatial_hi_time = t;
      if (variable_buffer_times.back() + cur_dt <= t_pending + 1e-12 * std::max(1.0, std::abs(t_pending))) {
        // Make the window's right endpoint available to the spatial interpolation before
        // integrating across it, then drain whole uniform steps.
        push_checkpoint(t_pending, spatial_pending);
        advance_abm_to(t_pending);
        // Slide the committed point to the last grid time we actually produced.
        t_committed = variable_buffer_times.back();
        spatial_lo = spatial_pending;
        spatial_lo_time = t_committed;
        // Refresh the predictor used by the next speculative Newton solve.
        refresh_dv_committed_from_buffer();
        // The committed front / checkpoints moved -> the PREDICT lookahead cache is stale.
        look_valid = false;
      }
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
        // Seed the checkpoint ring with the (accepted) initial condition so the spatial
        // interpolation has a left endpoint.
        push_checkpoint(t, spatial_y);
        // Seed the predictor with a real dv estimate at the initial condition; this is
        // the only dt_variables call that does not double up with one inside an ABM
        // substep.
        refresh_dv_committed_from_buffer();
        assembler->set_time(t);
        return;
      }

      if (t > frontier && !is_close(t, frontier)) {
        // IDA advanced -> the previous trial at `frontier` was accepted. spatial_hi
        // (saved from the last call at `frontier`) is the converged spatial state there.
        t_accepted = std::max(t_accepted, frontier);
        if (mode_stagger) {
          // Record it as a (downsampled) checkpoint feeding the spatial extrapolation; the
          // explicit buffer is advanced lazily in the serving step below, not here.
          maybe_push_checkpoint(frontier, spatial_hi);
        } else if (frontier > t_committed && !is_close(frontier, t_committed)) {
          // LAG / PREDICT: fold the accepted segment into the committed (interpolated)
          // output buffer.
          commit_segment_to(spatial_hi, frontier);
        }
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

      // Serve v(t) to IDA. All three modes are independent of `spatial_y`, so repeated
      // Newton iterations at the same trial endpoint (and retries after rejection) all see
      // the same v(t) -- which is what keeps Newton convergent when the FEM source reads v.
      if (mode_stagger) {
        // Keep the explicit grid half a step ahead of the accepted region, then: queries
        // past the current interval's midpoint fall inside the buffer -> interpolate
        // (centered); queries still in the first half are served by the forward predictor
        // (re-anchored at the buffer front).
        advance_lead_half();
        refresh_predictor_from_buffer_tail();
        if (t <= variable_buffer_times.back() || is_close(t, variable_buffer_times.back())) {
          interpolate_buffer(variable_y, t);
        } else {
          eval_predictor(variable_ret, t);
          eigen_to_dealii(variable_ret, variable_y);
        }
      } else if (mode_predict) {
        serve_lookahead(variable_y, t);
      } else {
        eval_predictor(variable_ret, t);
        eigen_to_dealii(variable_ret, variable_y);
      }
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
      if (mode_stagger) {
        // The accepted spatial state at t is known here, so record it as a checkpoint
        // (anchoring the spatial coupling at the right end), then advance the explicit
        // buffer up to t and interpolate. Each step is still taken exactly once.
        if (frontier > spatial_ckpt_times.back() && !is_close(frontier, spatial_ckpt_times.back()))
          maybe_push_checkpoint(frontier, spatial_hi);
        push_checkpoint(t, spatial_y);
        if (t > frontier) frontier = t;
        t_accepted = std::max(t_accepted, t);
        advance_lead_to(t);
        interpolate_buffer(variable_y, t);
        assembler->set_time(t);
        return;
      }
      if (frontier > t_committed && !is_close(frontier, t_committed))
        commit_segment_to(spatial_hi, frontier);
      if (t > t_committed && !is_close(t, t_committed)) {
        commit_segment_to(spatial_y, t);
        if (t > frontier) frontier = t;
      }
      // Serve v at t. If t lies within the committed buffer, interpolate it; the trailing
      // remainder (< cur_dt) that was intentionally left un-stepped is served by the
      // quadratic predictor (O(dt^3)) rather than the clamped buffer value, which would
      // otherwise lag by up to cur_dt at output points and at t_stop.
      if (t > variable_buffer_times.back() && !is_close(t, variable_buffer_times.back())) {
        eval_predictor(variable_ret, t);
        eigen_to_dealii(variable_ret, variable_y);
      } else {
        interpolate_buffer(variable_y, t);
      }
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