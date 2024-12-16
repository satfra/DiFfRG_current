// external libraries
#include <deal.II/base/config.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector_memory.h>

// DiFfRG
#include <DiFfRG/timestepping/solver/kinsol.hh>

namespace DiFfRG
{
  using namespace dealii;
  template <typename VectorType_>
  KINSOL<VectorType_>::KINSOL(double abstol_, double reltol_, double assemble_threshold_, uint max_steps_,
                              uint n_stepsize_iterations_)
      : abstol(abstol_), reltol(reltol_), assemble_threshold(assemble_threshold_), max_steps(max_steps_),
        n_stepsize_iterations(n_stepsize_iterations_), ignore_nonconv(false)
  {
    typename SUNDIALS::KINSOL<VectorType>::AdditionalData add_data;
    add_data.strategy = SUNDIALS::KINSOL<VectorType>::AdditionalData::SolutionStrategy::linesearch;
    add_data.function_tolerance = abstol;
    add_data.maximum_non_linear_iterations = max_steps;
    add_data.no_init_setup = true;
    kinsol = std::make_shared<SUNDIALS::KINSOL<VectorType>>(add_data);
  }

  template <typename VectorType_> void KINSOL<VectorType_>::reinit(const VectorType &u) { update_jacobian(u); }

  template <typename VectorType_> void KINSOL<VectorType_>::operator()(VectorType &iterate)
  {
    Timer timer;

    initial_vector = iterate;

    kinsol->reinit_vector = [&](VectorType &x) { x.reinit(iterate); };

    kinsol->residual = [&](const VectorType &evaluation_point, VectorType &residual) {
      this->residual(residual, evaluation_point);
      residual_vector = residual;
      return 0;
    };

    kinsol->setup_jacobian = [&](const VectorType &current_u, const VectorType & /*current_f*/) {
      this->update_jacobian(current_u);
      return 0;
    };

    kinsol->solve_with_jacobian = [&](const VectorType &rhs, VectorType &dst, const double /*tolerance*/) {
      this->lin_solve(dst, rhs);
      return 0;
    };

    kinsol->solve(iterate);
    if (iterate.l2_norm() > abstol) kinsol->solve(iterate);
    iterate_vector = iterate;

    timings_newton.push_back(timer.wall_time());
  }

  template <typename VectorType_> double KINSOL<VectorType_>::threshold(const double thr)
  {
    const double t = assemble_threshold;
    assemble_threshold = thr;
    return t;
  }

  template <typename VectorType_> const VectorType_ &KINSOL<VectorType_>::get_residual() const
  {
    return residual_vector;
  }
  template <typename VectorType_> const VectorType_ &KINSOL<VectorType_>::get_initial() const { return initial_vector; }
  template <typename VectorType_> const VectorType_ &KINSOL<VectorType_>::get_iterate() const { return iterate_vector; }

  template <typename VectorType_> double KINSOL<VectorType_>::average_time_newton() const
  {
    double t = 0.;
    double n = timings_newton.size();
    for (const auto &t_ : timings_newton)
      t += t_ / n;
    return t;
  }
  template <typename VectorType_> unsigned int KINSOL<VectorType_>::num_newton_calls() const
  {
    return timings_newton.size();
  }

  template <typename VectorType_> void KINSOL<VectorType_>::set_ignore_nonconv(bool x) { ignore_nonconv = x; }

  template <typename VectorType_> double KINSOL<VectorType_>::get_error() { return resnorm = get_EEst(); }
  template <typename VectorType_> unsigned int KINSOL<VectorType_>::get_step() { return step; }
  template <typename VectorType_> unsigned int KINSOL<VectorType_>::get_jacobians() { return jacobians; }

  template <typename VectorType_> double KINSOL<VectorType_>::get_EEst()
  {
    const VectorType &err = residual_vector;
    const VectorType &u_prev = initial_vector;
    const VectorType &u = iterate_vector;

    GrowingVectorMemory<VectorType> mem;
    typename VectorMemory<VectorType>::Pointer tmp(mem);
    tmp->reinit(u, true);

    using std::abs, std::max, std::sqrt;
    tbb::parallel_for(tbb::blocked_range<unsigned int>(0, tmp->size()), [&](tbb::blocked_range<unsigned int> r) {
      for (unsigned int n = r.begin(); n < r.end(); ++n) {
        (*tmp)[n] = abs(err[n]) / (abstol + max(abs(u_prev[n]), abs(u[n])) * reltol);
      }
    });
    return tmp->l2_norm() / sqrt(tmp->size());
    return tmp->linfty_norm(); // tmp->l2_norm() / sqrt(tmp->size());
  }

  template <typename VectorType_> bool KINSOL<VectorType_>::check()
  {
    if (step > max_steps) return false;
    if (step == 0) return true;
    double err = get_EEst();
    converged = err <= 1.;
    return !converged;
  }
} // namespace DiFfRG

template class DiFfRG::KINSOL<dealii::BlockVector<double>>;
template class DiFfRG::KINSOL<dealii::Vector<double>>;