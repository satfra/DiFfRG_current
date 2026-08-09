#pragma once

// standard library
#include <sstream>

// external libraries
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/output_session.hh>

namespace DiFfRG
{
  namespace Variables
  {
    using namespace dealii;

    template <typename... T> auto fe_tie(T &&...t)
    {
      return named_tuple<std::tuple<T &...>, StringSet<"variables">>(std::tie(t...));
    }

    /**
     * @brief The basic assembler that can be used for any standard CG scheme with flux and source.
     *
     * @tparam Model The model class which contains the physical equations.
     */
    template <typename Model_> class Assembler : public AbstractAssembler<Vector<double>, SparseMatrix<double>, 0>
    {
      constexpr static int nothing = 0;

    public:
      using Model = Model_;
      using NumberType = double;
      using VectorType = Vector<double>;

      using Components = typename Model_::Components;
      static constexpr uint dim = 0;

      [[deprecated("Pass output.log_port() or an intentional LogPort{}")]] Assembler(
          Model &model, DiFfRG::internal::LegacyDefaultLogPortArgument<Model, ConfigTree> config)
          : Assembler(model, config.value(), DiFfRG::internal::legacy_default_log_port<Model>())
      {
      }

      Assembler(Model &model, const ConfigTree &config, LogPort log_port)
          : model(model), log_port(std::move(log_port)),
            mesh_workers(config.get_uint("/discretization/mesh_workers", 8))
      {
        if (mesh_workers == 0) mesh_workers = std::max(1u, dealii::MultithreadInfo::n_threads() / 2);
        static_assert(Components::count_fe_functions() == 0, "The pure variable assembler cannot handle FE functions!");
        reinit();
      }

      virtual void reinit_vector(VectorType &vec) const override { vec.reinit(0); }

      virtual IndexSet get_differential_indices() const override { return IndexSet(); }

      virtual void attach_data_output(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                                      const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                      const VectorType &residual = VectorType()) override
      {
        (void)dt_solution;
        (void)residual;
        readouts(data_out, solution, variables);
      }

      virtual void reinit() override {}

      virtual void set_time(double t) override { model.set_time(t); }

      virtual const SparsityPattern &get_sparsity_pattern_jacobian() const override
      {
        return sparsity_pattern_jacobian;
      }
      virtual const SparseMatrix<NumberType> &get_mass_matrix() const override { return mass_matrix; }

      virtual void residual_variables(VectorType &residual, const VectorType &variables, const VectorType &) override
      {
        Timer timer;
        model.dt_variables(residual, fe_tie(variables));
        Kokkos::fence();
        timings_residual.push_back(timer.wall_time());
      };

      virtual void jacobian_variables(FullMatrix<NumberType> &jacobian, const VectorType &variables,
                                      const VectorType &) override
      {
        Timer timer;
        model.template jacobian_variables<0>(jacobian, fe_tie(variables));
        Kokkos::fence();
        timings_jacobian.push_back(timer.wall_time());
      };

      void readouts(OutputFrame<dim, VectorType> &data_out, const VectorType &, const VectorType &variables) const
      {
        auto helper = [&](auto &&...args) {
          if constexpr (sizeof...(args) == 3) {
            auto &&[id, EoMfun, outputter] = std::forward_as_tuple(std::forward<decltype(args)>(args)...);
            data_out.register_readout(id);
            (void)EoMfun;
            outputter(data_out, Point<0>(), fe_tie(variables));
          } else {
            DiFfRG::internal::validate_readout_helper_arity<decltype(args)...>();
          }
        };
        model.readouts_multiple(helper, data_out);
      }

      virtual void mass(VectorType &, const VectorType &, const VectorType &, NumberType) override {}

      virtual void residual(VectorType &, const VectorType &, NumberType, const VectorType &, NumberType,
                            const VectorType &variables = VectorType()) override
      {
        (void)variables;
      }

      virtual void jacobian_mass(SparseMatrix<NumberType> &, const VectorType &, const VectorType &, NumberType,
                                 NumberType) override
      {
      }

      virtual void jacobian(SparseMatrix<NumberType> &, const VectorType &, NumberType, const VectorType &, NumberType,
                            NumberType, const VectorType &variables = VectorType()) override
      {
        (void)variables;
      }

      template <typename String>
        requires std::convertible_to<String, std::string>
      [[deprecated("Construct the assembler with output.log_port() and call log() instead")]] void log(String &&) const
      {
        DiFfRG::internal::reject_named_assembler_log<String>();
      }

      void log() const
      {
        std::stringstream ss;
        ss << "Variable Assembler: " << std::endl;
        ss << "        Residual: " << average_time_residual_assembly() * 1000 << "ms (" << num_residuals() << ")"
           << std::endl;
        ss << "        Jacobian: " << average_time_jacobian_assembly() * 1000 << "ms (" << num_jacobians() << ")"
           << std::endl;
        log_port.info(ss.str());
      }

      double average_time_residual_assembly()
      {
        double t = 0.;
        double n = timings_residual.size();
        for (const auto &t_ : timings_residual)
          t += t_ / n;
        return t;
      }
      uint num_residuals() const { return timings_residual.size(); }

      double average_time_jacobian_assembly()
      {
        double t = 0.;
        double n = timings_jacobian.size();
        for (const auto &t_ : timings_jacobian)
          t += t_ / n;
        return t;
      }
      uint num_jacobians() const { return timings_jacobian.size(); }

    private:
      Model &model;
      LogPort log_port;

      /// Number of cells kept in flight in the MeshWorker assembly pipeline (its queue length).
      /// This is not a thread count - see /discretization/threads for that.
      uint mesh_workers;

      SparsityPattern sparsity_pattern_mass;
      SparsityPattern sparsity_pattern_jacobian;
      SparseMatrix<NumberType> mass_matrix;

      std::vector<double> timings_residual;
      std::vector<double> timings_jacobian;
    };
  } // namespace Variables
} // namespace DiFfRG
