#pragma once

// external libraries
#include <deal.II/lac/sparse_direct.h>

// DiFfRG
#include <DiFfRG/timestepping/linear_solver/abstract_linear_solver.hh>
#include <DiFfRG/timestepping/linear_solver/condition_estimate.hh>

namespace DiFfRG
{
  template <typename SparseMatrixType, typename VectorType>
  class UMFPack : public AbstractLinearSolver<SparseMatrixType, VectorType>
  {
  public:
    static constexpr bool performs_factorization = true;

    UMFPack() : matrix(nullptr) {}

    void init(const SparseMatrixType &matrix) { this->matrix = &matrix; }

    bool invert()
    {
      if (!matrix) throw std::runtime_error("UMFPack::invert: matrix not initialized");
      solver.initialize(*matrix);
      return true;
    }

    int solve(const VectorType &src, VectorType &dst, const double)
    {
      if (!matrix) throw std::runtime_error("UMFPack::solve: matrix not initialized");
      solver.vmult(dst, src);
      return -1;
    }

    void solve_transpose(const VectorType &src, VectorType &dst) const
    {
      if (!matrix) throw std::runtime_error("UMFPack::solve_transpose: matrix not initialized");
      dst = src;
      solver.solve(dst, true);
    }

    double estimate_rcond(const SparseMatrixType &input_matrix, const unsigned int max_iterations = 5) const
    {
      if (!matrix) throw std::runtime_error("UMFPack::estimate_rcond: matrix not initialized");
      const double one_norm = internal::matrix_one_norm(input_matrix);
      if (!(one_norm > 0.) || !std::isfinite(one_norm)) return std::numeric_limits<double>::quiet_NaN();

      const auto solve_direct = [&](const VectorType &src, VectorType &dst) { solver.vmult(dst, src); };
      const auto solve_direct_transpose = [&](const VectorType &src, VectorType &dst) {
        dst = src;
        solver.solve(dst, true);
      };
      const double inverse_one_norm = internal::estimate_inverse_one_norm<VectorType>(
          input_matrix.m(), solve_direct, solve_direct_transpose, max_iterations);
      if (!(inverse_one_norm > 0.) || !std::isfinite(inverse_one_norm)) return std::numeric_limits<double>::quiet_NaN();
      return std::clamp(1. / (one_norm * inverse_one_norm), 0., 1.);
    }

    double estimate_scaled_rcond(const SparseMatrixType &input_matrix, const unsigned int max_iterations = 5) const
    {
      if (!matrix) throw std::runtime_error("UMFPack::estimate_scaled_rcond: matrix not initialized");

      std::vector<double> row_scale, column_scale;
      double scaled_one_norm = 0.;
      internal::build_maximum_equilibration(input_matrix, row_scale, column_scale, scaled_one_norm);
      if (!(scaled_one_norm > 0.) || !std::isfinite(scaled_one_norm)) return std::numeric_limits<double>::quiet_NaN();

      const auto solve_scaled = [&](const VectorType &src, VectorType &dst) {
        VectorType rhs(src), solution(src);
        for (std::size_t i = 0; i < rhs.size(); ++i)
          rhs[i] /= row_scale[i];
        solver.vmult(solution, rhs);
        dst.reinit(solution);
        for (std::size_t i = 0; i < solution.size(); ++i)
          dst[i] = solution[i] / column_scale[i];
      };
      const auto solve_scaled_transpose = [&](const VectorType &src, VectorType &dst) {
        VectorType rhs(src), solution(src);
        for (std::size_t i = 0; i < rhs.size(); ++i)
          rhs[i] /= column_scale[i];
        solution = rhs;
        solver.solve(solution, true);
        dst.reinit(solution);
        for (std::size_t i = 0; i < solution.size(); ++i)
          dst[i] = solution[i] / row_scale[i];
      };

      const double inverse_one_norm = internal::estimate_inverse_one_norm<VectorType>(
          input_matrix.m(), solve_scaled, solve_scaled_transpose, max_iterations);
      if (!(inverse_one_norm > 0.) || !std::isfinite(inverse_one_norm)) return std::numeric_limits<double>::quiet_NaN();
      return std::clamp(1. / (scaled_one_norm * inverse_one_norm), 0., 1.);
    }

  private:
    const SparseMatrixType *matrix;
    dealii::SparseDirectUMFPACK solver;
  };
} // namespace DiFfRG
