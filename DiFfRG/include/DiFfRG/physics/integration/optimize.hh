#pragma once

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/loop_integrals.hh>

namespace DiFfRG
{
  template <typename Regulator, int dim = 4> double optimize_x_extent(const JSONValue &json)
  {
    static bool already_run = false;
    static double x_extent = 1.;

    if (already_run) return x_extent;

    const uint order = json.get_uint("/integration/x_order");
    const int verbosity = json.get_int("/output/verbosity");
    const double x_extent_tolerance = json.get_double("/integration/x_extent_tolerance");
    constexpr uint quadrature_factor = 8;
    auto optimize_x = [](double x) -> double { return 1. / (x + Regulator::RB(1., x)) * Regulator::RBdot(1., x); };

    // Optimize the extent of the momentum integral
    // we will integrate the optimize_x function over the intervals I1 = [0, x_extent], I2 = [0, x_extent * 2] and I3
    // = [0, x_extent * 10] and then search for an x_extent such that (I2 + I1) / I1 < x_extent_tolerance and (I2 +
    // I3) / I2 < x_extent_tolerance

    const dealii::QGauss<1> quadrature_1(quadrature_factor * 1 * order);
    const dealii::QGauss<1> quadrature_2(quadrature_factor * 2 * order);
    const dealii::QGauss<1> quadrature_3(quadrature_factor * 10 * order);

    double eps1 = 1., eps2 = 1.;
    uint decrease_counter = 0;

    if (verbosity > 1) std::cout << "Optimizing x_extent" << std::endl;

    bool all_zero = true;
    for (uint i = 0; i < 10; i++) {
      const double x = 1. + 1e-3 + (std::pow(1.5, (double)i) - 1.);
      const double x_test = optimize_x(x);
      if (!is_close(x_test, 0.)) all_zero = false;
    }
    if (all_zero) {
      x_extent = 1.;
      if (verbosity > 1) std::cout << "x_extent is set to 1.\n" << std::endl;
      already_run = true;
      return x_extent;
    }

    while (eps1 > x_extent_tolerance || eps2 > x_extent_tolerance) {
      const double I1 = LoopIntegrals::integrate<double, 4>(optimize_x, quadrature_1, x_extent, 1.);
      const double I2 = LoopIntegrals::integrate<double, 4>(optimize_x, quadrature_2, 2. * x_extent, 1.);
      const double I3 = LoopIntegrals::integrate<double, 4>(optimize_x, quadrature_3, 10. * x_extent, 1.);

      if (std::abs((I2 - I1) / I1) > eps1 && std::abs((I2 - I3) / I2) > eps2)
        decrease_counter++;
      else
        decrease_counter = 0;
      if (decrease_counter > 2)
        throw std::runtime_error("Cannot reach requested precision for x_extent - increase x_quadrature_order or "
                                 "decrease x_extent_tolerance.");

      eps1 = std::abs((I2 - I1) / I1);
      eps2 = std::abs((I2 - I3) / I2);

      if (verbosity > 1)
        std::cout << "x_extent: " << x_extent << " I1: " << I1 << " I2: " << I2 << " I3: " << I3 << " eps1: " << eps1
                  << " eps2: " << eps2 << std::endl;

      if (eps1 > x_extent_tolerance || eps2 > x_extent_tolerance) x_extent *= 1.15;
    }
    if (verbosity > 1) std::cout << "Optimizing x_extent done.\n" << std::endl;

    already_run = true;
    return x_extent;
  }
} // namespace DiFfRG