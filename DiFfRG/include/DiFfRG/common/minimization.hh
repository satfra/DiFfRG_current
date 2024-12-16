#pragma once

// standard library
#include <array>
#include <functional>
#include <stdexcept>

// external libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>

namespace DiFfRG
{
  /**
   * @brief Abstract class for minimization in arbitrary dimensions.
   *
   * @tparam dim Dimension of the minimization problem.
   */
  template <int dim> class AbstractMinimizer
  {
  protected:
    using FUN = std::function<double(const std::array<double, dim> &)>;

  public:
    /**
     * @brief Construct a new AbstractMinimizer object.
     *
     * @param f Objective function to minimize. Must take a std::array<double, dim> as input and return a double.
     * @param abs_tol Absolute tolerance for the minimization.
     * @param max_iter Maximum number of iterations.
     */
    AbstractMinimizer(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(f), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    /**
     * @brief Set the absolute tolerance for the minimization.
     *
     * @param abs_tol Absolute tolerance.
     */
    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    /**
     * @brief Set the maximum number of iterations.
     *
     * @param max_iter Maximum number of iterations.
     */
    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    /**
     * @brief Get the number of iterations used in the last minimization.
     *
     * @return uint Number of iterations.
     */
    uint get_iter() const { return iter; }

    /**
     * @brief Perform the minimization.
     *
     * @return std::array<double, dim> Minimimum of the objective function.
     */
    std::array<double, dim> minimize() { return this->minimize_impl(); }

  protected:
    FUN f;

    double abs_tol;
    uint max_iter;
    uint iter;

    virtual std::array<double, dim> minimize_impl() = 0;
  };

  // explicit specialization for 1D minimization
  template <> class AbstractMinimizer<1>
  {
  protected:
    using FUN = std::function<double(const double)>;
    using FUN_ARR = std::function<double(const std::array<double, 1> &)>;

  public:
    /**
     * @brief Construct a new AbstractMinimizer object.
     *
     * @param f Objective function to minimize. Must take a double as input and return a double.
     * @param abs_tol Absolute tolerance for the minimization.
     * @param max_iter Maximum number of iterations.
     */
    AbstractMinimizer(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f([f](const std::array<double, 1> &x) { return f(x[0]); }), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    /**
     * @brief Construct a new AbstractMinimizer object.
     *
     * @param f Objective function to minimize. Must take a std::array<double, 1> as input and return a double.
     * @param abs_tol Absolute tolerance for the minimization.
     * @param max_iter Maximum number of iterations.
     */
    AbstractMinimizer(const FUN_ARR &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(f), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    /**
     * @brief Set the absolute tolerance for the minimization.
     *
     * @param abs_tol Absolute tolerance.
     */
    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    /**
     * @brief Set the maximum number of iterations.
     *
     * @param max_iter Maximum number of iterations.
     */
    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    /**
     * @brief Get the number of iterations used in the last minimization.
     *
     * @return uint Number of iterations.
     */
    uint get_iter() const { return iter; }

    /**
     * @brief Perform the minimization.
     *
     * @return double Minimimum of the objective function.
     */
    double minimize() { return this->minimize_impl()[0]; }

  protected:
    FUN_ARR f;

    double abs_tol;
    uint max_iter;
    uint iter;

    virtual std::array<double, 1> minimize_impl() = 0;
  };

  /**
   * @brief Minimizer using the Nelder-Mead simplex algorithm from GSL.
   *
   * @tparam dim Dimension of the minimization problem.
   */
  template <int dim> class GSLSimplexMinimizer : public AbstractMinimizer<dim>
  {
    using FUN = AbstractMinimizer<dim>::FUN;

  public:
    /**
     * @brief Construct a new GSLSimplexMinimizer object.
     *
     * @param f Objective function to minimize. Must take a std::array<double, dim> as input and return a double.
     * @param abs_tol Absolute tolerance for the minimization.
     * @param max_iter Maximum number of iterations.
     */
    GSLSimplexMinimizer(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : AbstractMinimizer<dim>(f, abs_tol, max_iter), step_size(1.)
    {
      gsl_set_error_handler_off();
    }

    /**
     * @brief Set the initial step size for the minimization.
     *
     * @param step_size Initial step size.
     */
    void set_step_size(const double step_size) { this->step_size = step_size; }

    /**
     * @brief Set the initial guess for the minimization.
     *
     * @param x0 Initial guess.
     */
    void set_x0(const std::array<double, dim> &x0) { this->x0 = x0; }

    static double gsl_wrap(const gsl_vector *v, void *params)
    {
      GSLSimplexMinimizer<dim> *self = (GSLSimplexMinimizer<dim> *)params;
      std::array<double, dim> x;
      for (int i = 0; i < dim; ++i)
        x[i] = gsl_vector_get(v, i);
      return self->f(x);
    }

  protected:
    virtual std::array<double, dim> minimize_impl() override
    {
      gsl_multimin_function gsl_f;
      gsl_f.n = dim;
      gsl_f.f = &GSLSimplexMinimizer<dim>::gsl_wrap;
      gsl_f.params = this;

      gsl_vector *x = gsl_vector_alloc(dim);
      for (int i = 0; i < dim; ++i)
        gsl_vector_set(x, i, x0[i]);

      gsl_vector *init_step = gsl_vector_alloc(dim);
      gsl_vector_set_all(init_step, step_size);

      const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
      gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, dim);

      gsl_multimin_fminimizer_set(s, &gsl_f, x, init_step);

      int status;
      this->iter = 0;
      do {
        this->iter++;
        status = gsl_multimin_fminimizer_iterate(s);
        if (status) break;

        double size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size, this->abs_tol);

        if (status == GSL_SUCCESS) break;

      } while (status == GSL_CONTINUE && this->iter < this->max_iter);

      if (status != GSL_SUCCESS) throw std::runtime_error("Minimization failed.");

      std::array<double, dim> result;
      for (int i = 0; i < dim; ++i)
        result[i] = gsl_vector_get(s->x, i);

      gsl_multimin_fminimizer_free(s);
      gsl_vector_free(x);

      return result;
    }

    std::array<double, dim> x0;
    double step_size;
  };

  /**
   * @brief Minimizer in 1D using either the golden section, Brent or quadratic method from GSL.
   */
  class GSLMinimizer1D : public AbstractMinimizer<1>
  {
    using FUN = AbstractMinimizer<1>::FUN;

  public:
    /**
     * @brief Construct a new GSLMinimizer1D object.
     *
     * @param f Objective function to minimize. Must take a std::array<double, 1> as input and return a double.
     * @param abs_tol Absolute tolerance for the minimization.
     * @param max_iter Maximum number of iterations.
     */
    GSLMinimizer1D(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : AbstractMinimizer<1>(f, abs_tol, max_iter), x0(0.), min_x(-1.), max_x(1.), rel_tol(0.), m(method::brent)
    {
      gsl_set_error_handler_off();
    }

    /**
     * @brief List of available minimization methods.
     */
    enum method { golden_section, brent, quadratic };

    /**
     * @brief Set the initial guess for the minimization.
     *
     * @param x0 Initial guess.
     */
    void set_x0(const double x0) { this->x0 = x0; }

    /**
     * @brief Set the bounds for the minimization.
     *
     * @param min_x Lower bound.
     * @param max_x Upper bound.
     */
    void set_bounds(const double min_x, const double max_x)
    {
      this->min_x = min_x;
      this->max_x = max_x;
    }

    /**
     * @brief Set the minimization method. Default is Brent.
     *
     * @param m Minimization method.
     */
    void set_method(const method m) { this->m = m; }

    /**
     * @brief Set the relative tolerance for the minimization. Default is 0.
     *
     * @param rel_tol Relative tolerance.
     */
    void set_rel_tol(const double rel_tol) { this->rel_tol = rel_tol; }

  protected:
    virtual std::array<double, 1> minimize_impl() override
    {
      gsl_function gsl_f;
      gsl_f.function = &GSLMinimizer1D::gsl_wrap;
      gsl_f.params = this;

      const gsl_min_fminimizer_type *T;
      switch (this->m) {
      case golden_section:
        T = gsl_min_fminimizer_goldensection;
        break;
      case brent:
        T = gsl_min_fminimizer_brent;
        break;
      case quadratic:
        T = gsl_min_fminimizer_quad_golden;
        break;
      default:
        throw std::runtime_error("Unknown minimization method.");
      }

      gsl_min_fminimizer *s = gsl_min_fminimizer_alloc(T);

      double x = this->x0 < this->min_x || this->x0 > this->max_x ? 0.5 * (this->min_x + this->max_x) : this->x0;
      double x_lo = this->min_x;
      double x_hi = this->max_x;

      gsl_min_fminimizer_set(s, &gsl_f, x, x_lo, x_hi);

      double prev_x = -x;
      int status;
      this->iter = 0;
      int stuck = 0;
      do {
        this->iter++;
        status = gsl_min_fminimizer_iterate(s);

        x = gsl_min_fminimizer_x_minimum(s);
        x_lo = gsl_min_fminimizer_x_lower(s);
        x_hi = gsl_min_fminimizer_x_upper(s);

        status = gsl_min_test_interval(x_lo, x_hi, this->abs_tol, this->rel_tol);

        if (is_close(x, prev_x)) stuck++;
        if (stuck > 3) std::runtime_error("Minimization got stuck at x = " + std::to_string(x));
        prev_x = x;

      } while (status == GSL_CONTINUE && this->iter < this->max_iter);

      if (status != GSL_SUCCESS) throw std::runtime_error("Minimization failed.");

      gsl_min_fminimizer_free(s);

      return {x};
    }

    static double gsl_wrap(double x, void *params)
    {
      GSLMinimizer1D *self = (GSLMinimizer1D *)params;
      return self->f({{x}});
    }

    double x0;
    double min_x, max_x;
    double rel_tol;
    method m;
  };
} // namespace DiFfRG