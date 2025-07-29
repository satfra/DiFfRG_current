// external libraries
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/explicit_euler.hh>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperExplicitEuler<VectorType, SparseMatrixType, dim>::run(
      AbstractFlowingVariables<NumberType> *initial_condition, double start, double stop)
  {
    this->data_out = this->get_data_out();
    this->adaptor = this->get_adaptor();

    const SparseMatrixType &mass_matrix = assembler->get_mass_matrix();
    InverseSparseMatrixType inverse_mass_matrix;
    inverse_mass_matrix.initialize(mass_matrix);

    // Start out by saving step 0
    assembler->set_time(start);

    VectorType old_solution = initial_condition->spatial_data();
    VectorType solution = initial_condition->spatial_data();

    assembler->attach_data_output(*data_out, solution);

    data_out->flush(start);

    double last_save = start;
    double t = start;
    while (t < stop) {
      const auto now = std::chrono::high_resolution_clock::now();

      if ((*adaptor)(t, old_solution)) {
        solution = old_solution;
        inverse_mass_matrix.initialize(mass_matrix);
      }

      solution = 0.;
      assembler->set_time(t);
      assembler->residual(solution, old_solution, -expl.dt, 0.);
      inverse_mass_matrix.solve(solution);
      solution += old_solution;

      t += expl.dt;
      if ((t - last_save + 1e-4 * expl.dt) >= output_dt) {
        assembler->attach_data_output(*data_out, solution);

        data_out->flush(t);
        last_save = t;
      }
      old_solution = solution;

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "explicit residual", 1, ms_passed);
    }

    initial_condition->spatial_data() = solution;
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperExplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 1>;
template class DiFfRG::TimeStepperExplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 2>;
template class DiFfRG::TimeStepperExplicitEuler<dealii::Vector<double>, dealii::SparseMatrix<double>, 3>;

template class DiFfRG::TimeStepperExplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1>;
template class DiFfRG::TimeStepperExplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2>;
template class DiFfRG::TimeStepperExplicitEuler<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3>;