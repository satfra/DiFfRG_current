#pragma once

// external libraries
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

// DiFfRG
#include <DiFfRG/timestepping/linear_solver/abstract_linear_solver.hh>

namespace DiFfRG
{
  template <typename SparseMatrixType, typename VectorType,
            typename PreconditionerType = dealii::PreconditionJacobi<SparseMatrixType>>
  class GMRES : public AbstractLinearSolver<SparseMatrixType, VectorType>
  {
  public:
    GMRES() : matrix(nullptr) {}

    void init(const SparseMatrixType &matrix)
    {
      this->matrix = &matrix;
      preconditioner.initialize(matrix, 1.0);
    }

    bool invert() { return false; }

    int solve(const VectorType &src, VectorType &dst, const double tol)
    {
      if (!matrix) throw std::runtime_error("GMRES::solve: matrix not initialized");
      dealii::SolverControl solver_control(std::max<std::size_t>(1000, src.size() / 10), tol);
      dealii::SolverGMRES<VectorType> solver(solver_control);

      try {
        solver.solve(*matrix, dst, src, preconditioner);
      } catch (std::exception &e) {
        std::cerr << "GMRES linear solver failed: " << e.what() << std::endl;
        throw;
      }

      int steps = solver_control.last_step();
      return steps;
    }

  private:
    const SparseMatrixType *matrix;
    PreconditionerType preconditioner;
  };
} // namespace DiFfRG
