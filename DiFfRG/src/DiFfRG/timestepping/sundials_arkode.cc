// external libraries
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/sundials/arkode.h>

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/sundials_arkode.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      AbstractFlowingVariables<NumberType> *initial_condition, const double t_start, const double t_stop)
  {
    this->data_out = this->get_data_out();
    this->adaptor = this->get_adaptor();

    auto &full_data = initial_condition->data();
    if (full_data.n_blocks() != 1)
      run(full_data, t_start, t_stop);
    else
      run(full_data.block(0), t_start, t_stop);
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run_explicit(
      AbstractFlowingVariables<NumberType> *initial_condition, const double t_start, const double t_stop)
  {
    this->data_out = this->get_data_out();
    this->adaptor = this->get_adaptor();

    auto &full_data = initial_condition->data();
    if constexpr (dim == 0)
      run_explicit_vars(full_data.block(1), t_start, t_stop);
    else {
      if (full_data.n_blocks() != 1)
        run_explicit(full_data, t_start, t_stop);
      else
        run_explicit(full_data.block(0), t_start, t_stop);
    }
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run(VectorType &initial_data,
                                                                                        const double t_start,
                                                                                        const double t_stop)
  {
    auto y = initial_data;

    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    InverseSparseMatrixType inverse_mass_matrix;
    const auto &mass_matrix = assembler->get_mass_matrix();
    inverse_mass_matrix.initialize(mass_matrix);
    SparseMatrixType jacobian(assembler->get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver;

    double last_t = t_start;

    // Create a ARKode object with the right settings
    const auto add_data = typename SUNDIALS::ARKode<VectorType>::AdditionalData(
        t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5, 10, false, false, true, 3, impl.abs_tol, impl.rel_tol);
    SUNDIALS::ARKode<VectorType> ode_solver(add_data);

    ode_solver.implicit_function = [&](const double t, const VectorType &y, VectorType &ydot) {
      const auto now = std::chrono::high_resolution_clock::now();

      ydot = 0;
      assembler->set_time(t);
      assembler->residual(ydot, y, -1., y, 0.);

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "implicit residual", 1, ms_passed);

      last_t = t;
    };
    ode_solver.jacobian_preconditioner_setup = [&](const double t, const VectorType &y, const VectorType & /*fy*/,
                                                   const int /*jok*/, int & /*jcur*/, const double gamma) -> int {
      auto now = std::chrono::high_resolution_clock::now();

      jacobian = 0;
      assembler->set_time(t);
      assembler->jacobian(jacobian, y, gamma, y, 1., 1.);
      linSolver.init(jacobian);

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "jacobian construction", 1, ms_passed);
      now = std::chrono::high_resolution_clock::now();

      if (linSolver.invert()) {
        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(t, "jacobian inversion", 2, ms_passed);
      }

      last_t = t;

      return 0;
    };
    ode_solver.jacobian_preconditioner_solve = [&](const double, const VectorType &, const VectorType &,
                                                   const VectorType &, VectorType &, const double, const double,
                                                   const int) -> int {};
    ode_solver.jacobian_times_setup = [&](const double, const VectorType &, const VectorType &) -> int {};
    ode_solver.jacobian_times_vector = [&](const VectorType &v, VectorType &Jv, const double /*t*/,
                                           const VectorType & /*y*/,
                                           const VectorType & /*fy*/) -> int { jacobian.vmult(Jv, v); };
    ode_solver.solve_linearized_system = [&](SUNDIALS::SundialsOperator<VectorType> &,
                                             SUNDIALS::SundialsPreconditioner<VectorType> &, VectorType &x,
                                             const VectorType &b, double tol) -> int {
      const auto now = std::chrono::high_resolution_clock::now();
      const auto sol_iterations = linSolver.solve(b, x, tol);
      if (sol_iterations >= 0) {
        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(last_t, "linear solver (" + std::to_string(sol_iterations) + " it)", 2, ms_passed);
      }
    };
    ode_solver.mass_times_vector = [&](const double /*t*/, const VectorType &v, VectorType &Mv) {
      mass_matrix.vmult(Mv, v);
    };
    ode_solver.solve_mass = [&](SUNDIALS::SundialsOperator<VectorType> &,
                                SUNDIALS::SundialsPreconditioner<VectorType> &, VectorType &x, const VectorType &b,
                                double /*tol*/) { inverse_mass_matrix.vmult(x, b); };
    ode_solver.output_step = [&](const double t, const VectorType &sol, unsigned int /*step_number*/) {
      assembler->set_time(t);

      assembler->attach_data_output(*data_out, sol);
      data_out->flush(t);
    };

    ode_solver.solve_ode(y);

    initial_data = y;
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run_explicit(
      VectorType &initial_data, const double t_start, const double t_stop)
  {
    auto y = initial_data;

    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    InverseSparseMatrixType inverse_mass_matrix;
    const auto &mass_matrix = assembler->get_mass_matrix();
    inverse_mass_matrix.initialize(mass_matrix);

    // Create a SUNDIALS IDA object with the right settings
    const auto add_data = typename SUNDIALS::ARKode<VectorType>::AdditionalData(
        t_start, t_stop, expl.dt, output_dt, expl.minimal_dt, 5, 10, false, false, true, 3, expl.abs_tol, expl.rel_tol);
    SUNDIALS::ARKode<VectorType> ode_solver(add_data);

    ode_solver.explicit_function = [&](const double t, const VectorType &y, VectorType &ydot) {
      const auto now = std::chrono::high_resolution_clock::now();

      ydot = 0;
      assembler->set_time(t);
      assembler->residual(ydot, y, -1., y, 0.);

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "explicit residual", 1, ms_passed);
    };
    ode_solver.mass_times_vector = [&](const double /*t*/, const VectorType &v, VectorType &Mv) {
      mass_matrix.vmult(Mv, v);
    };
    ode_solver.solve_mass = [&](SUNDIALS::SundialsOperator<VectorType> &,
                                SUNDIALS::SundialsPreconditioner<VectorType> &, VectorType &x, const VectorType &b,
                                double /*tol*/) { inverse_mass_matrix.vmult(x, b); };
    ode_solver.output_step = [&](const double t, const VectorType &sol, unsigned int /*step_number*/) {
      assembler->set_time(t);

      assembler->attach_data_output(*data_out, sol);
      data_out->flush(t);
    };

    ode_solver.solve_ode(y);

    initial_data = y;
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run(BlockVectorType &initial_data,
                                                                                        const double t_start,
                                                                                        const double t_stop)
  {
    auto y = initial_data;

    if (y.n_blocks() != 2) throw std::runtime_error("TimeStepperSUNDIALS_ARKode_vars::run: y must have two blocks!");
    if (y.block(1).size() == 0)
      throw std::runtime_error(
          "TimeStepperSUNDIALS_ARKode_vars::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    InverseSparseMatrixType inverse_mass_matrix;
    const auto &mass_matrix = assembler->get_mass_matrix();
    inverse_mass_matrix.initialize(mass_matrix);
    SparseMatrixType jacobian(assembler->get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver;

    double last_t = t_start;

    // Create a SUNDIALS_ARKode object with the right settings
    const auto add_data = typename SUNDIALS::ARKode<BlockVectorType>::AdditionalData(
        t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5, 10, false, false, true, 3, impl.abs_tol, impl.rel_tol);
    SUNDIALS::ARKode<BlockVectorType> ode_solver(add_data);

    ode_solver.explicit_function = [&](const double t, const BlockVectorType &y, BlockVectorType &ydot) {
      const auto now = std::chrono::high_resolution_clock::now();

      ydot = 0;
      assembler->set_time(t);
      assembler->residual_variables(ydot.block(1), y.block(1), y.block(0));
      ydot.block(1) *= -1.;

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "explicit residual", 1, ms_passed);
    };
    ode_solver.implicit_function = [&](const double t, const BlockVectorType &y, BlockVectorType &ydot) {
      const auto now = std::chrono::high_resolution_clock::now();

      ydot = 0;
      assembler->set_time(t);
      assembler->residual(ydot.block(0), y.block(0), -1., y.block(0), 0., y.block(1));

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "implicit residual", 1, ms_passed);
    };
    ode_solver.jacobian_preconditioner_setup = [&](const double t, const BlockVectorType &y,
                                                   const BlockVectorType & /*fy*/, const int /*jok*/, int & /*jcur*/,
                                                   const double gamma) {
      auto now = std::chrono::high_resolution_clock::now();

      jacobian = 0;
      assembler->set_time(t);
      assembler->jacobian(jacobian, y.block(0), gamma, y.block(0), 1., 1., y.block(1));
      linSolver.init(jacobian);

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "jacobian construction", 1, ms_passed);
      now = std::chrono::high_resolution_clock::now();

      if (linSolver.invert()) {
        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(t, "jacobian inversion", 2, ms_passed);
      }

      last_t = t;
    };
    ode_solver.jacobian_preconditioner_solve = [&](const double, const BlockVectorType &, const BlockVectorType &,
                                                   const BlockVectorType &, BlockVectorType &, const double,
                                                   const double, const int) {};
    ode_solver.jacobian_times_setup = [&](const double, const BlockVectorType &, const BlockVectorType &) {};
    ode_solver.jacobian_times_vector = [&](const BlockVectorType &v, BlockVectorType &Jv, const double /*t*/,
                                           const BlockVectorType & /*y*/, const BlockVectorType & /*fy*/) {
      jacobian.vmult(Jv.block(0), v.block(0));
      Jv.block(1) = v.block(1);
    };
    ode_solver.solve_linearized_system = [&](SUNDIALS::SundialsOperator<BlockVectorType> &,
                                             SUNDIALS::SundialsPreconditioner<BlockVectorType> &, BlockVectorType &x,
                                             const BlockVectorType &b, double tol) {
      const auto now = std::chrono::high_resolution_clock::now();
      const auto sol_iterations = linSolver.solve(b.block(0), x.block(0), tol);
      if (sol_iterations >= 0) {
        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(last_t, "linear solver (" + std::to_string(sol_iterations) + " it)", 2, ms_passed);
      }
      x.block(1) = b.block(1);
    };
    ode_solver.mass_times_vector = [&](const double /*t*/, const BlockVectorType &v, BlockVectorType &Mv) {
      mass_matrix.vmult(Mv.block(0), v.block(0));
      Mv.block(1) = v.block(1);
    };
    ode_solver.solve_mass = [&](SUNDIALS::SundialsOperator<BlockVectorType> &,
                                SUNDIALS::SundialsPreconditioner<BlockVectorType> &, BlockVectorType &x,
                                const BlockVectorType &b, double /*tol*/) {
      inverse_mass_matrix.vmult(x.block(0), b.block(0));
      x.block(1) = b.block(1);
    };
    ode_solver.output_step = [&](const double t, const BlockVectorType &sol, unsigned int /*step_number*/) {
      assembler->set_time(t);

      assembler->attach_data_output(*data_out, sol.block(0), sol.block(1));
      data_out->flush(t);
    };

    ode_solver.solve_ode(y);

    initial_data = y;
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run_explicit(
      BlockVectorType &initial_data, const double t_start, const double t_stop)
  {
    auto y = initial_data;

    if (y.n_blocks() != 2) throw std::runtime_error("TimeStepperSUNDIALS_ARKode_vars::run: y must have two blocks!");
    if (y.block(1).size() == 0)
      throw std::runtime_error(
          "TimeStepperSUNDIALS_ARKode_vars::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    InverseSparseMatrixType inverse_mass_matrix;
    const auto &mass_matrix = assembler->get_mass_matrix();
    inverse_mass_matrix.initialize(mass_matrix);

    // Create a SUNDIALS IDA object with the right settings
    const auto add_data = typename SUNDIALS::ARKode<BlockVectorType>::AdditionalData(
        t_start, t_stop, expl.dt, output_dt, expl.minimal_dt, 5, 10, false, false, true, 3, expl.abs_tol, expl.rel_tol);
    SUNDIALS::ARKode<BlockVectorType> ode_solver(add_data);

    ode_solver.explicit_function = [&](const double t, const BlockVectorType &y, BlockVectorType &ydot) {
      const auto now = std::chrono::high_resolution_clock::now();

      ydot = 0;
      assembler->set_time(t);
      assembler->residual_variables(ydot.block(1), y.block(1), y.block(0));
      ydot.block(1) *= -1.;
      assembler->residual(ydot.block(0), y.block(0), -1., y.block(0), 0., y.block(1));

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "explicit residual", 1, ms_passed);
    };
    ode_solver.mass_times_vector = [&](const double /*t*/, const BlockVectorType &v, BlockVectorType &Mv) {
      mass_matrix.vmult(Mv.block(0), v.block(0));
      Mv.block(1) = v.block(1);
    };
    ode_solver.solve_mass = [&](SUNDIALS::SundialsOperator<BlockVectorType> &,
                                SUNDIALS::SundialsPreconditioner<BlockVectorType> &, BlockVectorType &x,
                                const BlockVectorType &b, double /*tol*/) {
      inverse_mass_matrix.vmult(x.block(0), b.block(0));
      x.block(1) = b.block(1);
    };
    ode_solver.output_step = [&](const double t, const BlockVectorType &sol, unsigned int /*step_number*/) {
      assembler->set_time(t);

      assembler->attach_data_output(*data_out, sol.block(0), sol.block(1));
      data_out->flush(t);
    };

    ode_solver.solve_ode(y);

    initial_data = y;
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run_explicit_vars(
      VectorType &initial_data, const double t_start, const double t_stop)
  {
    auto y = initial_data;

    // Create a SUNDIALS IDA object with the right settings
    const auto add_data = typename SUNDIALS::ARKode<VectorType>::AdditionalData(
        t_start, t_stop, expl.dt, output_dt, expl.minimal_dt, 5, 10, false, false, true, 3, expl.abs_tol, expl.rel_tol);
    SUNDIALS::ARKode<VectorType> ode_solver(add_data);

    ode_solver.explicit_function = [&](const double t, const VectorType &y, VectorType &ydot) {
      const auto now = std::chrono::high_resolution_clock::now();

      ydot = 0;
      assembler->set_time(t);
      assembler->residual_variables(ydot, y, y);
      ydot *= -1.;

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "explicit residual", 1, ms_passed);
    };
    ode_solver.output_step = [&](const double t, const VectorType &sol, unsigned int /*step_number*/) {
      assembler->set_time(t);

      assembler->attach_data_output(*data_out, Vector<double>(), sol);
      data_out->flush(t);
    };

    ode_solver.solve_ode(y);

    initial_data = y;
  }
}; // namespace DiFfRG

template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                                  DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                  DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                  DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                  DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                                  DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                  DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                  DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                  DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                                  DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                  DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                  DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                  DiFfRG::GMRES>;

template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                                  DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                  DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                  DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                  DiFfRG::GMRES>;