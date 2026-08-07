#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/ScaledGMRES.hh>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <cmath>
#include <type_traits>

using namespace DiFfRG;
using namespace dealii;

namespace
{
  struct SparseSystem {
    SparsityPattern sparsity;
    SparseMatrix<double> matrix;
  };

  Vector<double> badly_scaled_expected_solution()
  {
    Vector<double> expected_solution(3);
    expected_solution[0] = 1.25;
    expected_solution[1] = -2.5e-4;
    expected_solution[2] = 3.e-3;
    return expected_solution;
  }

  template <typename MatrixType> Vector<double> matrix_vector_product(const MatrixType &matrix, const Vector<double> &x)
  {
    Vector<double> result(matrix.m());
    result = 0.;
    for (std::size_t row = 0; row < matrix.m(); ++row)
      for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
        result[row] += entry->value() * x[entry->column()];
    return result;
  }

  SparseSystem make_badly_scaled_matrix()
  {
    DynamicSparsityPattern dsp(3, 3);
    dsp.add(0, 0);
    dsp.add(0, 1);
    dsp.add(1, 0);
    dsp.add(1, 1);
    dsp.add(1, 2);
    dsp.add(2, 1);
    dsp.add(2, 2);

    SparseSystem system;
    system.sparsity.copy_from(dsp);

    system.matrix.reinit(system.sparsity);
    system.matrix.set(0, 0, 1.e-8);
    system.matrix.set(0, 1, 2.e4);
    system.matrix.set(1, 0, -3.e-5);
    system.matrix.set(1, 1, 4.);
    system.matrix.set(1, 2, 5.e5);
    system.matrix.set(2, 1, 6.e-6);
    system.matrix.set(2, 2, -7.e8);

    return system;
  }

  void build_badly_scaled_block_matrix(BlockSparsityPattern &sparsity, BlockSparseMatrix<double> &matrix)
  {
    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit(1, 1);
    dsp.block(0, 1).reinit(1, 2);
    dsp.block(1, 0).reinit(2, 1);
    dsp.block(1, 1).reinit(2, 2);

    dsp.block(0, 0).add(0, 0);
    dsp.block(0, 1).add(0, 0);
    dsp.block(1, 0).add(0, 0);
    dsp.block(1, 1).add(0, 0);
    dsp.block(1, 1).add(0, 1);
    dsp.block(1, 1).add(1, 0);
    dsp.block(1, 1).add(1, 1);
    dsp.collect_sizes();

    sparsity.copy_from(dsp);

    matrix.reinit(sparsity);
    matrix.set(0, 0, 1.e-8);
    matrix.set(0, 1, 2.e4);
    matrix.set(1, 0, -3.e-5);
    matrix.set(1, 1, 4.);
    matrix.set(1, 2, 5.e5);
    matrix.set(2, 1, 6.e-6);
    matrix.set(2, 2, -7.e8);
  }

  SparseSystem make_banded_scaled_matrix(const unsigned int n)
  {
    DynamicSparsityPattern dsp(n, n);
    for (unsigned int i = 0; i < n; ++i) {
      dsp.add(i, i);
      if (i > 0) dsp.add(i, i - 1);
      if (i + 1 < n) dsp.add(i, i + 1);
      if (i + 2 < n) dsp.add(i, i + 2);
      if (i > 2) dsp.add(i, i - 3);
    }

    SparseSystem system;
    system.sparsity.copy_from(dsp);

    system.matrix.reinit(system.sparsity);
    const std::array<double, 6> row_exponents = {-6., -3., -1., 1., 3., 6.};
    const std::array<double, 5> col_exponents = {5., -4., 2., -2., 0.};

    for (unsigned int i = 0; i < n; ++i) {
      const double row_scale = std::pow(10., row_exponents[i % row_exponents.size()]);
      const auto add_entry = [&](const unsigned int j, const double value) {
        const double col_scale = std::pow(10., col_exponents[j % col_exponents.size()]);
        system.matrix.set(i, j, row_scale * value * col_scale);
      };

      add_entry(i, 4.5 + 0.01 * double(i % 7));
      if (i > 0) add_entry(i - 1, -1.1);
      if (i + 1 < n) add_entry(i + 1, -0.9);
      if (i + 2 < n) add_entry(i + 2, 0.35);
      if (i > 2) add_entry(i - 3, -0.18);
    }

    return system;
  }

  template <typename MatrixType>
  void check_badly_scaled_solution(const MatrixType &matrix, const Vector<double> &rhs, const Vector<double> &solution)
  {
    const Vector<double> expected_solution = badly_scaled_expected_solution();
    for (unsigned int i = 0; i < solution.size(); ++i) {
      CHECK(std::isfinite(solution[i]));
      CHECK(solution[i] == Catch::Approx(expected_solution[i]).epsilon(1.e-8).margin(1.e-10));
    }

    const Vector<double> residual_check = matrix_vector_product(matrix, solution);
    for (unsigned int i = 0; i < residual_check.size(); ++i)
      CHECK(residual_check[i] == Catch::Approx(rhs[i]).epsilon(1.e-8).margin(1.e-8));
  }
} // namespace

static_assert(
    std::is_same_v<GMRES<SparseMatrix<double>, Vector<double>>,
                   GMRES<SparseMatrix<double>, Vector<double>, PreconditionJacobi<SparseMatrix<double>>>>);
static_assert(
    std::is_same_v<ScaledLinearSolver<SparseMatrix<double>, Vector<double>>,
                   ScaledLinearSolver<SparseMatrix<double>, Vector<double>,
                                      GMRES<SparseMatrix<double>, Vector<double>, PreconditionIdentity>>>);
static_assert(
    std::is_same_v<ScaledGMRES<SparseMatrix<double>, Vector<double>>,
                   ScaledGMRES<SparseMatrix<double>, Vector<double>, PreconditionIdentity>>);

TEST_CASE("ScaledLinearSolver defaults to scaled GMRES", "[timestepping][linear_solver][scaled_gmres]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  const auto &matrix = system.matrix;
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  ScaledLinearSolver<SparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CHECK(iterations >= 0);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledGMRES solves a badly scaled sparse system", "[timestepping][linear_solver][scaled_gmres]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  const auto &matrix = system.matrix;
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  GMRES<SparseMatrix<double>, Vector<double>> unscaled_solver;
  unscaled_solver.init(matrix);
  Vector<double> unscaled_solution(3);
  const int unscaled_iterations = unscaled_solver.solve(rhs, unscaled_solution, 1.e-12);

  ScaledGMRES<SparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CAPTURE(unscaled_iterations, iterations);
  CHECK(unscaled_iterations >= 0);
  CHECK(iterations >= 0);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledUMFPack solves a badly scaled sparse system", "[timestepping][linear_solver][scaled_umfpack]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  const auto &matrix = system.matrix;
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  ScaledUMFPack<SparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);
  CHECK(solver.invert());

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CHECK(iterations == -1);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledLinearSolver wraps UMFPack explicitly", "[timestepping][linear_solver][scaled_umfpack]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  const auto &matrix = system.matrix;
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  using DirectSolver = UMFPack<SparseMatrix<double>, Vector<double>>;
  ScaledLinearSolver<SparseMatrix<double>, Vector<double>, DirectSolver> solver;
  solver.init(matrix);
  CHECK(solver.invert());

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CHECK(iterations == -1);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledUMFPack solves a badly scaled block sparse system", "[timestepping][linear_solver][scaled_umfpack]")
{
  BlockSparsityPattern sparsity;
  BlockSparseMatrix<double> matrix;
  build_badly_scaled_block_matrix(sparsity, matrix);
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  ScaledUMFPack<BlockSparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);
  CHECK(solver.invert());

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CHECK(iterations == -1);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledGMRES preconditioner choice is explicit on a badly scaled sparse system",
          "[timestepping][linear_solver][scaled_gmres][preconditioner]")
{
  const SparseSystem badly_scaled_system = make_badly_scaled_matrix();
  ScaledGMRES<SparseMatrix<double>, Vector<double>, PreconditionIdentity> identity_solver;
  identity_solver.init(badly_scaled_system.matrix);
  const Vector<double> identity_rhs =
      matrix_vector_product(badly_scaled_system.matrix, badly_scaled_expected_solution());
  Vector<double> identity_solution(3);
  const int identity_iterations = identity_solver.solve(identity_rhs, identity_solution, 1.e-12);

  CAPTURE(identity_iterations);
  CHECK(identity_iterations >= 0);
  check_badly_scaled_solution(badly_scaled_system.matrix, identity_rhs, identity_solution);

  const SparseSystem banded_system = make_banded_scaled_matrix(120);
  const auto &banded_matrix = banded_system.matrix;

  Vector<double> expected_solution(banded_matrix.n());
  for (unsigned int i = 0; i < expected_solution.size(); ++i)
    expected_solution[i] = std::sin(0.13 * double(i + 1)) + 0.25 * std::cos(0.07 * double(i));

  const Vector<double> rhs = matrix_vector_product(banded_matrix, expected_solution);

  ScaledGMRES<SparseMatrix<double>, Vector<double>, PreconditionJacobi<SparseMatrix<double>>> jacobi_solver;
  jacobi_solver.init(banded_matrix);
  Vector<double> jacobi_solution(banded_matrix.n());
  const int jacobi_iterations = jacobi_solver.solve(rhs, jacobi_solution, 1.e-10);

  const Vector<double> jacobi_residual = matrix_vector_product(banded_matrix, jacobi_solution);
  CAPTURE(identity_iterations, jacobi_iterations);
  CHECK(jacobi_iterations >= 0);
  for (unsigned int i = 0; i < jacobi_solution.size(); ++i) {
    CHECK(std::isfinite(jacobi_solution[i]));
    CHECK(jacobi_residual[i] == Catch::Approx(rhs[i]).epsilon(1.e-8).margin(1.e-7));
  }
}
