// DiFfRG
#include <DiFfRG/common/interpolation.hh>

namespace DiFfRG
{
  namespace Interpolation
  {
    Barycentric::Barycentric(std::vector<double> &x, std::vector<double> &y, uint order, bool copy)
    {
      if (copy)
        interpolator = std::make_unique<boost::math::interpolators::barycentric_rational<double>>(x.begin(), x.end(),
                                                                                                  y.begin(), order);
      else
        interpolator = std::make_unique<boost::math::interpolators::barycentric_rational<double>>(std::move(x),
                                                                                                  std::move(y), order);
    }

    Barycentric::Barycentric(const std::vector<double> &x, const std::vector<double> &y, uint order)
    {
      interpolator = std::make_unique<boost::math::interpolators::barycentric_rational<double>>(x.begin(), x.end(),
                                                                                                y.begin(), order);
    }

    Barycentric::Barycentric() : interpolator(nullptr) {}

    double Barycentric::operator()(double x) const { return interpolator->operator()(x); }

    double Barycentric::value(double x) const { return interpolator->operator()(x); }

    double Barycentric::derivative(double x) const { return interpolator->prime(x); }

    CubicSpline::CubicSpline(const std::vector<double> &x, const std::vector<double> &y)
    {
      acc = gsl_interp_accel_alloc();
      spline = gsl_spline_alloc(gsl_interp_cspline, x.size());

      gsl_spline_init(spline, x.data(), y.data(), x.size());
    }

    CubicSpline::~CubicSpline()
    {
      gsl_spline_free(spline);
      gsl_interp_accel_free(acc);
    }

    double CubicSpline::operator()(double x) const { return gsl_spline_eval(spline, x, acc); }

    double CubicSpline::value(double x) const { return gsl_spline_eval(spline, x, acc); }

    double CubicSpline::derivative(double x) const { return gsl_spline_eval_deriv(spline, x, acc); }
  } // namespace Interpolation
} // namespace DiFfRG
