#pragma once

// external libraries
#include <deal.II/lac/sparse_matrix.h>

namespace DiFfRG
{
  template <typename SparseMatrixType, typename VectorType> class AbstractLinearSolver
  {
  public:
    AbstractLinearSolver() {}

    void init(const SparseMatrixType &matrix)
    {
      (void)matrix;
      throw std::runtime_error("AbstractLinearSolver::init: not implemented");
    };

    bool invert()
    {
      throw std::runtime_error("AbstractLinearSolver::invert: not implemented");
      return false;
    };

    int solve(const VectorType &src, const VectorType &dst, const double tol)
    {
      (void)src;
      (void)dst;
      (void)tol;
      throw std::runtime_error("AbstractLinearSolver::solve: not implemented");
      return -1;
    };
  };
} // namespace DiFfRG