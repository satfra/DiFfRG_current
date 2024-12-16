#pragma once

// external libraries
#include <deal.II/lac/sparse_direct.h>

// DiFfRG
#include <DiFfRG/timestepping/linear_solver/abstract_linear_solver.hh>

namespace DiFfRG
{
  template <typename SparseMatrixType, typename VectorType>
  class UMFPack : public AbstractLinearSolver<SparseMatrixType, VectorType>
  {
  public:
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

  private:
    const SparseMatrixType *matrix;
    dealii::SparseDirectUMFPACK solver;
  };
} // namespace DiFfRG