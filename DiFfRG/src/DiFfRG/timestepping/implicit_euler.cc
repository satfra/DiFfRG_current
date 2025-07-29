// external libraries
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/implicit_euler.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/solver/kinsol.hh>
#include <DiFfRG/timestepping/solver/newton.hh>
#include <DiFfRG/timestepping/timestep_control/PI.hh>
#include <vector>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperImplicitEuler<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      AbstractFlowingVariables<NumberType> *initial_condition, double start, double stop)
  {
    this->data_out = this->get_data_out();
    this->adaptor = this->get_adaptor();

    // make some local copies of the initial condition which we use for stepping
    VectorType old_solution = initial_condition->spatial_data();
    VectorType solution = initial_condition->spatial_data();

    // newton algorithm
    Newton<VectorType> newton(impl.abs_tol, impl.rel_tol, 2e-1, 11, 21);

    // create time controller instance
    TC_PI tc(newton, 1, start, stop, impl.dt, impl.minimal_dt, impl.maximal_dt, output_dt);
    assembler->set_time(start);

    // create jacobian and solver for inverse jacobian
    SparseMatrixType jacobian(assembler->get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver;

    // all functions for assembly of the problem and linear solving
    newton.residual = [&](VectorType &res, const VectorType &u) {
      const auto now = std::chrono::high_resolution_clock::now();

      res = 0;
      assembler->residual(res, u, tc.get_dt(), 1);
      assembler->mass(res, old_solution, -1.);

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(tc.get_t(), "implicit residual", 1, ms_passed);
    };

    newton.update_jacobian = [&](const VectorType &u) {
      auto now = std::chrono::high_resolution_clock::now();

      jacobian = 0;
      assembler->jacobian(jacobian, u, tc.get_dt(), 1);
      linSolver.init(jacobian);

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(tc.get_t(), "jacobian construction", 1, ms_passed);
      now = std::chrono::high_resolution_clock::now();

      if (linSolver.invert()) {
        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(tc.get_t(), "jacobian inversion", 2, ms_passed);
      }
    };

    newton.lin_solve = [&](VectorType &Du, const VectorType &res) {
      const auto now = std::chrono::high_resolution_clock::now();
      const auto sol_iterations = linSolver.solve(res, Du, std::min(impl.abs_tol, impl.rel_tol * res.l2_norm()));
      if (sol_iterations >= 0) {
        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(tc.get_t(), "linear solver (" + std::to_string(sol_iterations) + " it)", 2, ms_passed);
      }
    };

    newton.reinit(solution);

    // saving and stepping helper functions
    auto save_data = [&](double t) {
      assembler->set_time(t);

      if (t > start)
        assembler->attach_data_output(*data_out, solution, newton.get_residual());
      else
        assembler->attach_data_output(*data_out, solution);

      data_out->flush(t);
    };
    auto dt_step = [&](double t, double dt) {
      assembler->set_time(t + dt);
      solution = old_solution;
      newton(solution);
      old_solution = solution;
    };

    // the actual time loop
    save_data(start);
    while (!tc.finished()) {
      if ((*adaptor)(tc.get_t(), old_solution)) solution = old_solution;
      tc.advance(dt_step, save_data);
    }

    initial_condition->spatial_data() = solution;
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, DiFfRG::GMRES>;

template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                DiFfRG::GMRES>;
template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                DiFfRG::GMRES>;
template class DiFfRG::TimeStepperImplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                DiFfRG::GMRES>;