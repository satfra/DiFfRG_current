#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

namespace DiFfRG
{
  template <int dim> class AbstractRootFinder
  {
  protected:
    using FUN = std::function<bool(const std::array<double, dim> &)>;

  public:
    AbstractRootFinder(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(f), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    uint get_iter() const { return iter; }

    std::array<double, dim> search() { return this->search_impl(); }

  protected:
    FUN f;

    double abs_tol;
    uint max_iter;
    uint iter;

    virtual std::array<double, dim> search_impl() = 0;
  };

  // explicit specialization for 1D search
  template <> class AbstractRootFinder<1>
  {
  protected:
    using FUN = std::function<bool(const double)>;
    using FUN_ARR = std::function<bool(const std::array<double, 1> &)>;

  public:
    AbstractRootFinder(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f([f](const std::array<double, 1> &x) { return f(x[0]); }), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    AbstractRootFinder(const FUN_ARR &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(f), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    uint get_iter() const { return iter; }

    double search() { return this->search_impl()[0]; }

  protected:
    FUN_ARR f;

    double abs_tol;
    uint max_iter;
    uint iter;

    virtual std::array<double, 1> search_impl() = 0;
  };

  class BisectionRootFinder : public AbstractRootFinder<1>
  {
  public:
    BisectionRootFinder(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : AbstractRootFinder<1>(f, abs_tol, max_iter)
    {
    }

    void set_x_min(const double x_min) { this->x_min = x_min; }
    void set_x_max(const double x_max) { this->x_max = x_max; }

    void set_bounds(const double x_min, const double x_max)
    {
      this->x_min = x_min;
      this->x_max = x_max;
    }

    void set_next_x(const std::function<double(double, double)> &next_x) { this->next_x = next_x; }

  protected:
    std::array<double, 1> search_impl() override
    {
      double x_min = this->x_min;
      double x_max = this->x_max;
      double x_test = 0.;

      bool success = false;

      for (uint i = 0; i < this->max_iter; i++) {
        x_test = next_x(x_min, x_max);
        success = this->f({{x_test}});
        if (success)
          x_max = x_test;
        else
          x_min = x_test;

        this->iter = i;
        if (success && std::abs(x_max - x_min) < this->abs_tol) break;
      }

      if (!success) throw std::runtime_error("BisectionRootFinder: search did not converge");

      return {{x_test}};
    }

    double x_min;
    double x_max;

    std::function<double(double, double)> next_x = [](const double x_min, const double x_max) {
      return (x_min + x_max) / 2.;
    };
  };

  /**
   * @brief Bisection search which converges a target value rather than the search variable.
   *
   * The bracketing logic is identical to BisectionRootFinder: the callback returns `true` if the
   * tested point lies at or above the root (tightening the upper bound) and `false` if it lies
   * below (tightening the lower bound). The difference is the stopping criterion. Instead of
   * shrinking the interval in x until it falls below `abs_tol`, the callback additionally writes a
   * target value through its `double &` out-parameter, and the search stops once two consecutive
   * *successful* evaluations produce target values differing by less than `abs_tol`.
   *
   * This is the right tool when the quantity of interest is not x itself but an observable
   * computed at x, whose sensitivity to x is not known a priori.
   *
   * @code
   * BisectionRootFinderTarget search(
   *     [&](const double x, double &target) -> bool {
   *       config.set_double("/physical/m2A", x);
   *       return run_at(x, target); // writes the observable into target
   *     },
   *     tol, 100);
   * search.set_bounds(lower, upper);
   * const double x      = search.search();
   * const double target = search.get_target();
   * @endcode
   *
   * Failed evaluations need not write anything meaningful into the target; only successful ones are
   * compared. If the bracket collapses to machine precision, or `max_iter` is exhausted, before the
   * criterion is met, the search stops, warns, and returns the last successful point with
   * `converged()` reporting `false`. Only a search without any successful evaluation throws.
   *
   * Note that the criterion fires early if the target happens to be insensitive to x over part of
   * the bracket, since two neighbouring successes then look converged. In that case tighten
   * `abs_tol` or install a `next_x` which does not bisect.
   *
   * Two deliberate differences to BisectionRootFinder: the iteration counter is advanced *before*
   * the callback runs, so `get_iter()` inside the callback names the step being taken, and the
   * bounds are NaN-initialised so that a forgotten `set_bounds` throws instead of reading garbage.
   */
  class BisectionRootFinderTarget
  {
  protected:
    using FUN = std::function<bool(const double, double &)>;

  public:
    // the callback is taken by value and moved: callers pass a temporary lambda, and the copy from
    // a const reference also trips a spurious -Wmaybe-uninitialized inside std::function on GCC 16
    BisectionRootFinderTarget(FUN f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(std::move(f)), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    void set_x_min(const double x_min) { this->x_min = x_min; }
    void set_x_max(const double x_max) { this->x_max = x_max; }

    void set_bounds(const double x_min, const double x_max)
    {
      this->x_min = x_min;
      this->x_max = x_max;
    }

    void set_next_x(const std::function<double(double, double)> &next_x) { this->next_x = next_x; }

    /**
     * @brief Width of the x-bracket below which the search gives up. Zero (the default) selects an
     * automatic, magnitude-scaled multiple of the machine epsilon.
     */
    void set_x_collapse_tol(const double x_collapse_tol) { this->x_collapse_tol = x_collapse_tol; }

    uint get_iter() const { return iter; }

    /**
     * @brief Target value belonging to the point returned by search(). NaN before search() ran.
     */
    double get_target() const { return target; }

    /**
     * @brief Whether the last search() met the target tolerance. False if it stopped early because
     * the bracket collapsed or the iteration limit was reached.
     */
    bool converged() const { return m_converged; }

    double search()
    {
      if (!(std::isfinite(this->x_min) && std::isfinite(this->x_max)))
        throw std::runtime_error("BisectionRootFinderTarget: search bounds were not set");

      double x_lo = this->x_min;
      double x_hi = this->x_max;

      double best_x = std::numeric_limits<double>::quiet_NaN();
      double best_target = std::numeric_limits<double>::quiet_NaN();
      bool have_success = false;

      this->m_converged = false;
      this->target = std::numeric_limits<double>::quiet_NaN();

      for (uint i = 0; i < this->max_iter; ++i) {
        this->iter = i;

        const double x_test = next_x(x_lo, x_hi);
        double x_target = std::numeric_limits<double>::quiet_NaN();
        const bool success = this->f(x_test, x_target);

        if (success) {
          x_hi = x_test;
          // Two consecutive successes whose targets agree to abs_tol: the observable is resolved.
          const bool target_converged = have_success && std::abs(x_target - best_target) < this->abs_tol;
          best_x = x_test;
          best_target = x_target;
          have_success = true;
          if (target_converged) {
            this->m_converged = true;
            break;
          }
        } else
          x_lo = x_test;

        const double collapse_tol =
            this->x_collapse_tol > 0.
                ? this->x_collapse_tol
                : 4. * std::numeric_limits<double>::epsilon() *
                      std::max(1., std::max(std::abs(x_lo), std::abs(x_hi)));
        if (std::abs(x_hi - x_lo) <= collapse_tol) break;
      }

      if (!have_success)
        throw std::runtime_error("BisectionRootFinderTarget: no evaluation of the search interval succeeded");

      this->target = best_target;

      if (!this->m_converged)
        spdlog::warn("BisectionRootFinderTarget: stopped after {} iterations without reaching the target tolerance "
                     "{:.6e}; returning x = {:.12e} with target = {:.12e}",
                     this->iter + 1, this->abs_tol, best_x, best_target);

      return best_x;
    }

  protected:
    FUN f;

    double abs_tol;
    uint max_iter;
    uint iter;

    double x_min = std::numeric_limits<double>::quiet_NaN();
    double x_max = std::numeric_limits<double>::quiet_NaN();
    double x_collapse_tol = 0.;

    double target = std::numeric_limits<double>::quiet_NaN();
    bool m_converged = false;

    std::function<double(double, double)> next_x = [](const double x_min, const double x_max) {
      return (x_min + x_max) / 2.;
    };
  };
} // namespace DiFfRG