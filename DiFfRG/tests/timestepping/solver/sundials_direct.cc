#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <DiFfRG/timestepping/sundials/ida.hh>
#include <DiFfRG/timestepping/sundials/kinsol.hh>
#include <DiFfRG/timestepping/sundials/nvector.hh>

#include <cmath>
#include <stdexcept>
#include <string>

TEST_CASE("Direct SUNDIALS N_Vector operations", "[sundials][direct][nvector]")
{
  DiFfRG::sundials::Context context;
  dealii::Vector<double> x{1., -2., 3.};
  dealii::Vector<double> y{4., 5., -6.};
  dealii::Vector<double> z(3);

  DiFfRG::sundials::NVectorHandle nx(DiFfRG::sundials::create_view(x, context.get()));
  DiFfRG::sundials::NVectorHandle ny(DiFfRG::sundials::create_view(y, context.get()));
  DiFfRG::sundials::NVectorHandle nz(DiFfRG::sundials::create_view(z, context.get()));

  N_VLinearSum(2., nx.get(), -1., ny.get(), nz.get());

  REQUIRE(z[0] == Catch::Approx(-2.));
  REQUIRE(z[1] == Catch::Approx(-9.));
  REQUIRE(z[2] == Catch::Approx(12.));
  REQUIRE(N_VDotProd(nx.get(), ny.get()) == Catch::Approx(-24.));
}

TEST_CASE("Direct SUNDIALS IDA defaults match the migrated deal.II wrapper contract", "[sundials][direct][ida]")
{
  using AdditionalData = DiFfRG::sundials::IDA<dealii::Vector<double>>::AdditionalData;

  const AdditionalData data;

  REQUIRE(data.initial_step_size == Catch::Approx(1e-2));
  REQUIRE(data.output_period == Catch::Approx(1e-1));
  REQUIRE(data.minimum_step_size == Catch::Approx(1e-6));
  REQUIRE(data.maximum_non_linear_iterations == 10);
  REQUIRE(data.relative_tolerance == Catch::Approx(1e-5));
  REQUIRE(data.ignore_algebraic_terms_for_errors == true);
  REQUIRE(data.ic_type == AdditionalData::use_y_diff);
  REQUIRE(data.reset_type == AdditionalData::use_y_diff);
}

TEST_CASE("Direct SUNDIALS KINSOL solves a linear system", "[sundials][direct][kinsol]")
{
  dealii::FullMatrix<double> matrix(2, 2);
  matrix(0, 0) = 3.;
  matrix(0, 1) = 1.;
  matrix(1, 0) = 1.;
  matrix(1, 1) = 2.;

  dealii::FullMatrix<double> inverse(matrix);
  inverse.invert(matrix);

  dealii::Vector<double> rhs{1., 0.};
  dealii::Vector<double> iterate(2);

  DiFfRG::sundials::KINSOL<dealii::Vector<double>>::AdditionalData data;
  data.function_tolerance = 1e-14;
  data.maximum_non_linear_iterations = 10;
  data.maximum_newton_step = 1e10;

  DiFfRG::sundials::KINSOL<dealii::Vector<double>> solver(data);
  solver.reinit_vector = [](dealii::Vector<double> &v) { v.reinit(2); };
  solver.residual = [&](const dealii::Vector<double> &u, dealii::Vector<double> &res) {
    matrix.vmult(res, u);
    res -= rhs;
    return 0;
  };
  solver.setup_jacobian = [](const dealii::Vector<double> &, const dealii::Vector<double> &) { return 0; };
  solver.solve_with_jacobian = [&](const dealii::Vector<double> &src, dealii::Vector<double> &dst, double) {
    inverse.vmult(dst, src);
    return 0;
  };

  solver.solve(iterate);

  dealii::Vector<double> expected(2);
  inverse.vmult(expected, rhs);
  REQUIRE(iterate[0] == Catch::Approx(expected[0]).margin(1e-12));
  REQUIRE(iterate[1] == Catch::Approx(expected[1]).margin(1e-12));
}

TEST_CASE("Direct SUNDIALS IDA solves a harmonic oscillator", "[sundials][direct][ida]")
{
  constexpr double kappa = 1.;
  dealii::FullMatrix<double> matrix(2, 2);
  matrix(0, 1) = -1.;
  matrix(1, 0) = kappa * kappa;

  dealii::FullMatrix<double> jacobian(2, 2);
  dealii::FullMatrix<double> inverse(2, 2);

  dealii::Vector<double> y{0., kappa};
  dealii::Vector<double> y_dot{kappa, 0.};

  DiFfRG::sundials::IDA<dealii::Vector<double>>::AdditionalData data(0., 1., 1e-3, 1., 0., 5, 20, 0., 1e-10, 1e-10);
  DiFfRG::sundials::IDA<dealii::Vector<double>> solver(data);
  solver.reinit_vector = [](dealii::Vector<double> &v) { v.reinit(2); };
  solver.residual = [&](double, const dealii::Vector<double> &current_y, const dealii::Vector<double> &current_y_dot,
                        dealii::Vector<double> &res) {
    res = current_y_dot;
    matrix.vmult_add(res, current_y);
    return 0;
  };
  solver.setup_jacobian = [&](double, const dealii::Vector<double> &, const dealii::Vector<double> &, double alpha) {
    jacobian = matrix;
    jacobian(0, 0) += alpha;
    jacobian(1, 1) += alpha;
    inverse.invert(jacobian);
    return 0;
  };
  solver.solve_with_jacobian = [&](const dealii::Vector<double> &src, dealii::Vector<double> &dst, double) {
    inverse.vmult(dst, src);
    return 0;
  };

  solver.solve_dae(y, y_dot);

  REQUIRE(y[0] == Catch::Approx(std::sin(1.)).margin(1e-6));
  REQUIRE(y[1] == Catch::Approx(std::cos(1.)).margin(1e-6));
}

TEST_CASE("Direct SUNDIALS IDA rethrows callback exceptions during initial condition correction",
          "[sundials][direct][ida][exceptions]")
{
  dealii::Vector<double> y{1.};
  dealii::Vector<double> y_dot{0.};

  DiFfRG::sundials::IDA<dealii::Vector<double>> solver;
  solver.reinit_vector = [](dealii::Vector<double> &v) { v.reinit(1); };
  solver.residual = [](double, const dealii::Vector<double> &, const dealii::Vector<double> &,
                       dealii::Vector<double> &) -> int {
    throw std::runtime_error("initial-condition callback failed");
  };
  solver.setup_jacobian = [](double, const dealii::Vector<double> &, const dealii::Vector<double> &, double) { return 0; };
  solver.solve_with_jacobian = [](const dealii::Vector<double> &src, dealii::Vector<double> &dst, double) {
    dst = src;
    return 0;
  };

  try {
    solver.solve_dae(y, y_dot);
    FAIL("Expected callback exception to propagate");
  } catch (const std::runtime_error &error) {
    REQUIRE(std::string(error.what()).find("initial-condition callback failed") != std::string::npos);
  }
}
