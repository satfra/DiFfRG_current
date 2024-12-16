#pragma once

#include <array>

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
} // namespace DiFfRG