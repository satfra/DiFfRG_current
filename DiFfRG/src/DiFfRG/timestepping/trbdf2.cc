// external libraries
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/solver/kinsol.hh>
#include <DiFfRG/timestepping/solver/newton.hh>
#include <DiFfRG/timestepping/timestep_control/PI.hh>
#include <DiFfRG/timestepping/trbdf2.hh>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  uint TimeStepperTRBDF2<VectorType, SparseMatrixType, dim, LinearSolver>::get_jacobians()
  {
    return ptr_newton_TR->get_jacobians() + ptr_newton_BDF2->get_jacobians();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  double TimeStepperTRBDF2<VectorType, SparseMatrixType, dim, LinearSolver>::get_error()
  {
    return std::sqrt(powr<2>(ptr_newton_TR->get_error()) + powr<2>(ptr_newton_BDF2->get_error()));
  }
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperTRBDF2<VectorType, SparseMatrixType, dim, LinearSolver>::set_ignore_nonconv(bool x)
  {
    ptr_newton_TR->set_ignore_nonconv(x);
    ptr_newton_BDF2->set_ignore_nonconv(x);
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperTRBDF2<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      AbstractFlowingVariables<NumberType> *initial_condition, double start, double stop)
  {
    std::shared_ptr<DataOutput<dim, VectorType>> data_out_expl;
    DataOutput<dim, VectorType> *data_out = nullptr;
    if(this->data_out == nullptr) {
      data_out_expl = std::make_shared<DataOutput<dim, VectorType>>(json);
      data_out = data_out_expl.get();
    }
    else
      data_out = this->data_out;

    const double gamma = 2. - std::sqrt(2.);

    // make some local copies of the initial condition which we use for stepping
    VectorType u_n = initial_condition->spatial_data();
    VectorType u_npgamma = initial_condition->spatial_data();
    VectorType u_np1 = initial_condition->spatial_data();

    // newton algorithm
    ptr_newton_TR = std::make_shared<Newton<VectorType>>(impl.abs_tol, impl.rel_tol, 2e-1, 11, 21);
    ptr_newton_BDF2 = std::make_shared<Newton<VectorType>>(impl.abs_tol, impl.rel_tol, 2e-1, 11, 21);
    auto &newton_TR = *ptr_newton_TR;
    auto &newton_BDF2 = *ptr_newton_BDF2;

    // create time controller instance
    TC_PI tc(*this, 2, start, stop, impl.dt, impl.minimal_dt, impl.maximal_dt, output_dt);
    assembler->set_time(0.);

    // initialize jacobians and inverse jacobians
    SparseMatrixType jacobian_TR(assembler->get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver_TR;
    SparseMatrixType jacobian_BDF2(assembler->get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver_BDF2;

    // all functions for assembly of the problem and linear solving
    newton_TR.residual = [&](VectorType &res, const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      res = 0;

      assembler->set_time(t);
      assembler->residual(res, u_n, gamma * dt / 2., -1.);

      assembler->set_time(t + dt * gamma);
      assembler->residual(res, u, gamma * dt / 2., 1.);
    };
    newton_TR.update_jacobian = [&](const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      jacobian_TR = 0;

      assembler->set_time(t + dt * gamma);
      assembler->jacobian(jacobian_TR, u, gamma * dt / 2., 1.);

      if (verbosity > 0) std::cout << "jacobian TR: " << std::endl;

      linSolver_TR.init(jacobian_TR);
      linSolver_TR.invert();
    };
    newton_TR.lin_solve = [&](VectorType &Du, const VectorType &res) {
      linSolver_TR.solve(res, Du, std::min(impl.abs_tol, impl.rel_tol * res.l2_norm()));
    };
    newton_TR.reinit(u_n);

    newton_BDF2.residual = [&](VectorType &res, const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      res = 0;

      assembler->set_time(t);
      assembler->mass(res, u_n, powr<2>(1. - gamma) / gamma);

      assembler->set_time(t + dt * gamma);
      assembler->mass(res, u_npgamma, -1. / gamma);

      assembler->set_time(t + dt);
      assembler->residual(res, u, (1. - gamma) * dt, (2. - gamma));
    };
    newton_BDF2.update_jacobian = [&](const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      jacobian_BDF2 = 0;

      assembler->set_time(t + dt);
      assembler->jacobian(jacobian_BDF2, u, (1. - gamma) * dt, (2. - gamma));

      if (verbosity > 0) std::cout << "jacobian BDF2: " << std::endl;

      linSolver_BDF2.init(jacobian_BDF2);
      linSolver_BDF2.invert();
    };
    newton_BDF2.lin_solve = [&](VectorType &Du, const VectorType &res) {
      linSolver_BDF2.solve(res, Du, std::min(impl.abs_tol, impl.rel_tol * res.l2_norm()));
    };
    newton_BDF2.reinit(u_n);

    // saving and stepping helper functions
    auto save_data = [&](double t) {
      assembler->set_time(t);
      assembler->attach_data_output(*data_out, u_n);
      data_out->flush(t);
    };
    auto dt_step = [&](double /*t*/, double /*dt*/) {
      u_npgamma = u_n;
      newton_TR(u_npgamma);
      u_np1 = u_npgamma;
      newton_BDF2(u_np1);
      u_n = u_np1;
    };

    // the actual time loop
    save_data(0.);
    while (!tc.finished()) {
      if ((*adaptor)(tc.get_t(), u_n)) {
        u_np1 = u_n;
        u_npgamma = u_n;
        jacobian_TR.reinit(assembler->get_sparsity_pattern_jacobian());
        newton_TR.reinit(u_n);
        jacobian_BDF2.reinit(assembler->get_sparsity_pattern_jacobian());
        newton_BDF2.reinit(u_n);
      }
      if (verbosity > 0) std::cout << "t = " << tc.get_t() << std::endl;
      tc.advance(dt_step, save_data);
    }

    initial_condition->spatial_data() = u_n;
  }

} // namespace DiFfRG

template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1, DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2, DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3, DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, DiFfRG::GMRES>;

template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperTRBDF2<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3, DiFfRG::GMRES>;