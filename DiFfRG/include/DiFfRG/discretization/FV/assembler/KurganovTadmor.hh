#pragma once

// external libraries

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      template <typename Discretization_, typename Model_>
      class Assembler : public AbstractAssembler<typename Discretization_::VectorType,
                                                 typename Discretization_::SparseMatrixType, Discretization_::dim>
      {
      protected:
        template <typename... T> auto fv_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, "variables">(std::tie(t...));
        }

        template <typename... T> static constexpr auto v_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, "variables", "extractors">(std::tie(t...));
        }

        template <typename... T> static constexpr auto e_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, "fe_functions", "fe_derivatives", "fe_hessians", "extractors",
                             "variables">(std::tie(t...));
        }

      public:
        using Discretization = Discretization_;
        using Model = Model_;
        using NumberType = typename Discretization::NumberType;
        using VectorType = typename Discretization::VectorType;

        using Components = typename Discretization::Components;
        static constexpr uint dim = Discretization::dim;

        Assembler(Discretization &discretization, Model &model, const JSONValue &json)
            : discretization(discretization), model(model), mapping(discretization.get_mapping()),
              triangulation(discretization.get_triangulation()), json(json),
              threads(json.get_uint("/discretization/threads")),
              batch_size(json.get_uint("/discretization/batch_size")),
              EoM_abs_tol(json.get_double("/discretization/EoM_abs_tol")),
              EoM_max_iter(json.get_uint("/discretization/EoM_max_iter"))
        {
          if (this->threads == 0) this->threads = dealii::MultithreadInfo::n_threads() / 2;
          spdlog::get("log")->info("FV: Using {} threads for assembly.", threads);
        }

        virtual void reinit_vector(VectorType &vec) const override { vec.reinit(0); }

        virtual IndexSet get_differential_indices() const override { return IndexSet(); }

        virtual void attach_data_output(DataOutput<dim, VectorType> &data_out, const VectorType &solution,
                                        const VectorType &variables, const VectorType &dt_solution = VectorType(),
                                        const VectorType &residual = VectorType())
        {
          (void)dt_solution;
          (void)residual;
          (void)solution;
          (void)variables;
          (void)data_out;
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
          model.dt_variables(residual, fv_tie(variables));
        };

        virtual void jacobian_variables(FullMatrix<NumberType> &jacobian, const VectorType &variables,
                                        const VectorType &) override
        {
          model.template jacobian_variables<0>(jacobian, fv_tie(variables));
        };

        void readouts(DataOutput<dim, VectorType> &data_out, const VectorType &, const VectorType &variables) const
        {
          auto helper = [&](auto EoMfun, auto outputter) {
            (void)EoMfun;
            outputter(data_out, Point<0>(), fv_tie(variables));
          };
          model.template readouts_multiple(helper, data_out);
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

        virtual void jacobian(SparseMatrix<NumberType> &, const VectorType &, NumberType, const VectorType &,
                              NumberType, NumberType, const VectorType &variables = VectorType()) override
        {
          (void)variables;
        }

      protected:
        Discretization &discretization;
        Model &model;
        const Mapping<dim> &mapping;
        const Triangulation<dim> &triangulation;

        const JSONValue &json;

        uint threads;
        const uint batch_size;

        mutable typename Triangulation<dim>::cell_iterator EoM_cell;
        typename Triangulation<dim>::cell_iterator old_EoM_cell;
        const double EoM_abs_tol;
        const uint EoM_max_iter;

        SparsityPattern sparsity_pattern_mass;
        SparsityPattern sparsity_pattern_jacobian;
        SparseMatrix<NumberType> mass_matrix;
      };
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG