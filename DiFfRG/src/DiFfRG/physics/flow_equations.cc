// external libraries
#include <spdlog/spdlog.h>

// DiFfRG
#include <DiFfRG/physics/flow_equations.hh>

namespace DiFfRG
{
  using namespace dealii;

  constexpr uint quadrature_factor = 5;

  FlowEquations::FlowEquations(const JSONValue &json, std::function<double(double)> optimize_x)
      : x_quadrature_order(json.get_uint("/integration/x_quadrature_order")),
        x_quadrature(QGauss<1>(x_quadrature_order)), x_extent(1.),
        angle_quadrature_order(json.get_uint("/integration/angle_quadrature_order")),
        angle_quadrature(QGauss<1>(angle_quadrature_order)),
        jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")),
        jac_x_quadrature_order(jacobian_quadrature_factor * x_quadrature_order),
        jac_x_quadrature(QGauss<1>(jac_x_quadrature_order)),
        jac_angle_quadrature_order(jacobian_quadrature_factor * angle_quadrature_order),
        jac_angle_quadrature(QGauss<1>(jac_angle_quadrature_order)),
        x_extent_tolerance(json.get_double("/integration/x_extent_tolerance")), optimize_x(optimize_x), k(1),
        unoptimized(true), verbosity(json.get_int("/output/verbosity"))
  {
    update_x_extent();
    unoptimized = false;
    if (verbosity > 1) std::cout << "FlowEquations initialized" << std::endl;
  }

  void FlowEquations::print_parameters(const std::string logname) const
  {
    std::stringstream stream;
    stream << "FlowEquations:" << "\n"
           << std::setw(40) << "x_quadrature_order: " << std::setw(10) << x_quadrature_order << "\n"
           << std::setw(40) << "jac_x_quadrature_order: " << std::setw(10) << jac_x_quadrature_order << "\n"
           << std::setw(40) << "x_extent: " << std::setw(10) << x_extent << "\n"
           << std::setw(40) << "angle_quadrature_order: " << std::setw(10) << angle_quadrature_order << "\n"
           << std::setw(40) << "jac_angle_quadrature_order: " << std::setw(10) << jac_angle_quadrature_order;
    spdlog::get(logname)->info(stream.str());
  }

  void FlowEquations::set_jacobian_quadrature_factor(const double jacobian_quadrature_factor)
  {
    jac_x_quadrature_order = jacobian_quadrature_factor * x_quadrature_order;
    jac_x_quadrature = QGauss<1>(jac_x_quadrature_order);

    jac_angle_quadrature_order = jacobian_quadrature_factor * angle_quadrature_order;
    jac_angle_quadrature = QGauss<1>(jac_angle_quadrature_order);
  }

  void FlowEquations::set_k(const double k) { this->k = k; }

  void FlowEquations::update_x_extent()
  {
    // Optimize the extent of the momentum integral
    // we will integrate the optimize_x function over the intervals I1 = [0, x_extent], I2 = [0, x_extent * 2] and I3 =
    // [0, x_extent * 10] and then search for an x_extent such that (I2 + I1) / I1 < x_extent_tolerance and (I2 + I3) /
    // I2 < x_extent_tolerance

    const QGauss<1> quadrature_1(quadrature_factor * 1 * x_quadrature_order);
    const QGauss<1> quadrature_2(quadrature_factor * 2 * x_quadrature_order);
    const QGauss<1> quadrature_3(quadrature_factor * 10 * x_quadrature_order);
    x_extent = 1.;
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
      return;
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
  }

  FlowEquationsFiniteT::FlowEquationsFiniteT(const JSONValue &json, const double T,
                                             std::function<double(double)> optimize_x,
                                             std::function<double(double)> optimize_x0,
                                             std::function<double(double)> optimize_q0)
      : x_quadrature_order(json.get_uint("/integration/x_quadrature_order")),
        x_quadrature(QGauss<1>(x_quadrature_order)), x_extent(1.), // x
        x0_summands(json.get_uint("/integration/x0_summands")),
        x0_quadrature_order(json.get_uint("/integration/x0_quadrature_order")),
        x0_quadrature(QGauss<1>(x0_quadrature_order)), x0_extent(0), // x0
        q0_summands(json.get_uint("/integration/q0_summands")),
        q0_quadrature_order(json.get_uint("/integration/q0_quadrature_order")),
        q0_quadrature(QGauss<1>(q0_quadrature_order)), q0_extent(0), // q0
        angle_quadrature_order(json.get_uint("/integration/angle_quadrature_order")),
        angle_quadrature(QGauss<1>(angle_quadrature_order)), // angle
        jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")),
        jac_x_quadrature_order(jacobian_quadrature_factor * x_quadrature_order),
        jac_x_quadrature(QGauss<1>(jac_x_quadrature_order)), // jac x
        jac_x0_quadrature_order(jacobian_quadrature_factor * x0_quadrature_order),
        jac_x0_quadrature(QGauss<1>(jac_x0_quadrature_order)), // jac m
        jac_q0_quadrature_order(jacobian_quadrature_factor * q0_quadrature_order),
        jac_q0_quadrature(QGauss<1>(jac_q0_quadrature_order)), // jac m
        jac_angle_quadrature_order(jacobian_quadrature_factor * angle_quadrature_order),
        jac_angle_quadrature(QGauss<1>(jac_angle_quadrature_order)), // jac cos
        x_extent_tolerance(json.get_double("/integration/x_extent_tolerance")),
        x0_extent_tolerance(json.get_double("/integration/x0_extent_tolerance")),
        q0_extent_tolerance(json.get_double("/integration/q0_extent_tolerance")), // tols
        T(T), optimize_x(optimize_x), optimize_x0(optimize_x0), optimize_q0(optimize_q0), k(1.), unoptimized(true),
        verbosity(json.get_int("/output/verbosity"))
  {
    // It does not matter what k is set to here, as long as it is not zero
    update_x_extent();
    if (x0_summands > 0) update_x0_extent();
    if (q0_summands > 0) update_q0_extent();
    unoptimized = false;
    if (verbosity > 0) std::cout << "FlowEquationsFiniteT initialized" << std::endl;
  }

  void FlowEquationsFiniteT::print_parameters(const std::string logname) const
  {
    std::stringstream stream;
    stream << "FlowEquationsFiniteT:" << "\n"
           << std::setw(47) << std::setw(10) << "x_quadrature_order: " << x_quadrature_order << "\n"
           << std::setw(47) << std::setw(10) << "jac_x_quadrature_order: " << jac_x_quadrature_order << "\n"
           << std::setw(47) << std::setw(10) << "x_extent: " << x_extent << "\n"
           << std::setw(47) << std::setw(10) << "x0_quadrature_order: " << x0_quadrature_order << "\n"
           << std::setw(47) << std::setw(10) << "jac_x0_quadrature_order: " << jac_x0_quadrature_order << "\n"
           << std::setw(47) << std::setw(10) << "x0_extent: " << x0_extent << "\n"
           << std::setw(47) << std::setw(10) << "q0_quadrature_order: " << q0_quadrature_order << "\n"
           << std::setw(47) << std::setw(10) << "jac_q0_quadrature_order: " << jac_q0_quadrature_order << "\n"
           << std::setw(47) << std::setw(10) << "q0_extent: " << q0_extent << "\n"
           << std::setw(47) << std::setw(10) << "angle_quadrature_order: " << angle_quadrature_order << "\n"
           << std::setw(47) << std::setw(10) << "jac_angle_quadrature_order: " << jac_angle_quadrature_order << "\n"
           << std::setw(47) << std::setw(10) << "T: " << T;
    spdlog::get(logname)->info(stream.str());
  }

  void FlowEquationsFiniteT::set_k(const double k) { this->k = k; }

  void FlowEquationsFiniteT::update_x_extent()
  {
    // Optimize the extent of the momentum integral
    // we will integrate the optimize_x function over the intervals I1 = [0, x_extent], I2 = [0, x_extent * 2] and I3 =
    // [0, x_extent * 10] and then search for an x_extent such that (I2 + I1) / I1 < x_extent_tolerance and (I2 + I3) /
    // I2 < x_extent_tolerance

    const QGauss<1> quadrature_1(quadrature_factor * x_quadrature_order);
    const QGauss<1> quadrature_2(quadrature_factor * 2 * x_quadrature_order);
    const QGauss<1> quadrature_3(quadrature_factor * 10 * x_quadrature_order);
    x_extent = 1.;
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
      return;
    }

    while (eps1 > x_extent_tolerance || eps2 > x_extent_tolerance) {
      const double I1 = LoopIntegrals::integrate<double, 4>(optimize_x, quadrature_1, x_extent, k);
      const double I2 = LoopIntegrals::integrate<double, 4>(optimize_x, quadrature_2, 2. * x_extent, k);
      const double I3 = LoopIntegrals::integrate<double, 4>(optimize_x, quadrature_3, 10. * x_extent, k);

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
  }

  void FlowEquationsFiniteT::update_q0_extent()
  {
    // Optimize the extent of the momentum integral
    // we will integrate the optimize_q0 function over the intervals I1 = [0, q0_extent], I2 = [0, q0_extent * 2] and I3
    // = [0, q0_extent * 10] and then search for an q0_extent such that (I2 + I1) / I1 < x_extent_tolerance and (I2 +
    // I3) / I2 < x_extent_tolerance

    const QGauss<1> quadrature_1(quadrature_factor * q0_quadrature_order);
    const QGauss<1> quadrature_2(quadrature_factor * 2 * q0_quadrature_order);
    const QGauss<1> quadrature_3(quadrature_factor * 10 * q0_quadrature_order);
    q0_extent = 0.5 * sqrt(x_extent) * k;
    double eps1 = 1., eps2 = 1.;
    uint decrease_counter = 0;
    if (verbosity > 1) std::cout << "Optimizing q0_extent" << std::endl;
    while (eps1 > q0_extent_tolerance || eps2 > q0_extent_tolerance) {
      const double I1 = std::abs(LoopIntegrals::sum<double>(optimize_q0, 0, quadrature_1, q0_extent, 0.));
      const double I2 = std::abs(LoopIntegrals::sum<double>(optimize_q0, 0, quadrature_2, 2. * q0_extent, 0.));
      const double I3 = std::abs(LoopIntegrals::sum<double>(optimize_q0, 0, quadrature_3, 10. * q0_extent, 0.));

      if (std::abs((I2 - I1) / I1) > eps1 && std::abs((I2 - I3) / I2) > eps2)
        decrease_counter++;
      else
        decrease_counter = 0;
      if (decrease_counter > 2)
        throw std::runtime_error("Cannot reach requested precision for q0_extent - increase q0_quadrature_order or "
                                 "decrease q0_extent_tolerance. Last eps1: " +
                                 std::to_string(eps1) + " Last eps2: " + std::to_string(eps2) +
                                 " Last q0_extent: " + std::to_string(q0_extent) + " Last I1: " + std::to_string(I1) +
                                 " Last I2: " + std::to_string(I2) + " Last I3: " + std::to_string(I3));

      eps1 = std::abs((I2 - I1) / I1);
      eps2 = std::abs((I2 - I3) / I2);

      if (verbosity > 1)
        std::cout << "q0_extent: " << q0_extent << " I1: " << I1 << " I2: " << I2 << " I3: " << I3 << " eps1: " << eps1
                  << " eps2: " << eps2 << std::endl;

      if (eps1 > q0_extent_tolerance || eps2 > q0_extent_tolerance) q0_extent *= 1.15;
    }
    if (verbosity > 1) std::cout << "Optimizing q0_extent done.\n" << std::endl;
  }

  void FlowEquationsFiniteT::update_x0_extent()
  {
    // Optimize the extent of the momentum integral
    // we will integrate the optimize_x0 function over the intervals I1 = [0, x0_extent], I2 = [0, x0_extent * 2] and I3
    // = [0, x0_extent * 10] and then search for an x0_extent such that (I2 + I1) / I1 < x_extent_tolerance and (I2 +
    // I3) / I2 < x_extent_tolerance

    const QGauss<1> quadrature_1(quadrature_factor * x0_quadrature_order);
    const QGauss<1> quadrature_2(quadrature_factor * 2 * x0_quadrature_order);
    const QGauss<1> quadrature_3(quadrature_factor * 10 * x0_quadrature_order);
    x0_extent = 0.5 * sqrt(x_extent);
    double eps1 = 1., eps2 = 1.;
    uint decrease_counter = 0;
    if (verbosity > 1) std::cout << "Optimizing x0_extent" << std::endl;
    while (eps1 > x0_extent_tolerance || eps2 > x0_extent_tolerance) {
      const double I1 = std::abs(LoopIntegrals::sum<double>(optimize_x0, 0, quadrature_1, x0_extent * k, 0.));
      const double I2 = std::abs(LoopIntegrals::sum<double>(optimize_x0, 0, quadrature_2, 2. * x0_extent * k, 0.));
      const double I3 = std::abs(LoopIntegrals::sum<double>(optimize_x0, 0, quadrature_3, 10. * x0_extent * k, 0.));

      if (std::abs((I2 - I1) / I1) > eps1 && std::abs((I2 - I3) / I2) > eps2)
        decrease_counter++;
      else
        decrease_counter = 0;
      if (decrease_counter > 1)
        throw std::runtime_error("Cannot reach requested precision for x0_extent - increase x0_quadrature_order or "
                                 "decrease x0_extent_tolerance. Last eps1: " +
                                 std::to_string(eps1) + " Last eps2: " + std::to_string(eps2) +
                                 " Last x0_extent: " + std::to_string(x0_extent) + " Last I1: " + std::to_string(I1) +
                                 " Last I2: " + std::to_string(I2) + " Last I3: " + std::to_string(I3));
      eps1 = std::abs((I2 - I1) / I1);
      eps2 = std::abs((I2 - I3) / I2);

      if (verbosity > 1)
        std::cout << "x0_extent: " << x0_extent << " I1: " << I1 << " I2: " << I2 << " I3: " << I3 << " eps1: " << eps1
                  << " eps2: " << eps2 << std::endl;

      if (eps1 > x0_extent_tolerance || eps2 > x0_extent_tolerance) x0_extent *= 1.15;
    }
    if (verbosity > 1) std::cout << "Optimizing x0_extent done.\n" << std::endl;
  }

  void FlowEquationsFiniteT::set_jacobian_quadrature_factor(const double jacobian_quadrature_factor)
  {
    jac_x_quadrature_order = jacobian_quadrature_factor * x_quadrature_order;
    jac_x_quadrature = QGauss<1>(jac_x_quadrature_order);

    jac_angle_quadrature_order = jacobian_quadrature_factor * angle_quadrature_order;
    jac_angle_quadrature = QGauss<1>(jac_angle_quadrature_order);

    jac_x0_quadrature_order = jacobian_quadrature_factor * x0_quadrature_order;
    jac_x0_quadrature = QGauss<1>(jac_x0_quadrature_order);

    jac_q0_quadrature_order = jacobian_quadrature_factor * q0_quadrature_order;
    jac_q0_quadrature = QGauss<1>(jac_q0_quadrature_order);
  }
} // namespace DiFfRG