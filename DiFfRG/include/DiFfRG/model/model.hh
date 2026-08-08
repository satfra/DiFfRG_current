#pragma once

// external libraries
#include "DiFfRG/common/config_tree.hh"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// standard library
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

// DiFfRG
#include <DiFfRG/discretization/common/affine_constraint_metadata.hh>
#include <DiFfRG/model/ad.hh>
#include <DiFfRG/model/component_descriptor.hh>
#include <DiFfRG/model/fv_boundaries.hh>
#include <DiFfRG/model/numflux.hh>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief This namespace contains all default implementations and definitions needed for numerical models.
   *
   */
  namespace def
  {
    /**
     * @brief The abstract interface for any numerical model.
     * Most methods have a standard implementation, which can be overwritten if needed.
     * To see how the models are used, refer to the DiFfRG::AbstractAssembler class and the Numerical Models guide
     * in the documentation.
     *
     * @tparam Model The model which implements this interface. (CRTP)
     * @tparam Components_ The components of the model, this must be a DiFfRG::ComponentDescriptor.
     */
    template <typename Model, typename Components_> class AbstractModel
    {
      Model &asImp()
      {
        static_assert(
            std::is_base_of_v<AbstractModel<Model, Components_>, Model>,
            "AbstractModel<Model, Components>: Model must inherit from AbstractModel<Model, Components> (CRTP). "
            "Check that your model class passes itself as the first template argument.");
        return static_cast<Model &>(*this);
      }
      const Model &asImp() const
      {
        static_assert(
            std::is_base_of_v<AbstractModel<Model, Components_>, Model>,
            "AbstractModel<Model, Components>: Model must inherit from AbstractModel<Model, Components> (CRTP). "
            "Check that your model class passes itself as the first template argument.");
        return static_cast<const Model &>(*this);
      }

    protected:
      Components_ m_components;
      auto &components() { return m_components; }

    public:
      const auto &get_components() const { return m_components; }
      using Components = Components_;
      /**
       * @name Spatial discretization
       */
      //@{

      /**
       * @brief This method implements the initial condition for the FE functions.
       *
       * @note No standard implementation is given, this method has to be reimplemented whenever one uses FE functions.
       *
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param u_i the field values \f$u_i(x)\f$ at the point `x`.
       * This method should fill this argument with the desired initial condition.
       */
      template <int dim, typename Vector> void initial_condition(const Point<dim> &x, Vector &u_i) const = delete;

      /**
       * @brief
       * The mass function \f$m_i(\partial_t u_j, u_j, x)\f$ is implemented in this method.
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model. It is however necessary to leave at least the NumberType, Vector,
       * and Vector_dot template parameters, as these can differ between calls (e.g. when doing automatic
       * differentiation).
       *
       * @note The standard implementation of this method simply sets \f$m_i = \partial_t u_i\f$.
       *
       * @param m_i the resulting mass function \f$m_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param u_i the field values \f$u_i(x)\f$ at the point `x`.
       * @param dt_u_i the time derivative of the field values \f$\partial_t u_i(x)\f$ at the point `x`.
       */
      template <int dim, typename NumberType, typename Vector, typename Vector_dot, size_t n_fe_functions>
      void mass([[maybe_unused]] std::array<NumberType, n_fe_functions> &m_i, [[maybe_unused]] const Point<dim> &x,
                [[maybe_unused]] const Vector &u_i, const Vector_dot &dt_u_i) const
      {
        for (uint i = 0; i < n_fe_functions; ++i)
          m_i[i] = dt_u_i[i];
      }

      /**
       * @brief If not using a DAE, the mass matrix \f$m_{ij}(x)\f$ is implemented in this method.
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model. It is however necessary to leave at least the NumberType, Vector,
       * and Vector_dot template parameters, as these can differ between calls (e.g. when doing automatic
       * differentiation).
       *
       * @note The standard implementation of this method simply sets \f$m_{ij} = \delta_{ij}\f$.
       *
       * @param m_ij the resulting mass matrix \f$m_{ij}\f$, with \f$N_f\f$ components in each dimension.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       */
      template <int dim, typename NumberType, size_t n_fe_functions>
      void mass(std::array<std::array<NumberType, n_fe_functions>, n_fe_functions> &m_ij,
                [[maybe_unused]] const Point<dim> &x) const
      {
        for (uint i = 0; i < n_fe_functions; ++i)
          for (uint j = 0; j < n_fe_functions; ++j)
            m_ij[i][j] = 0.;
        for (uint i = 0; i < n_fe_functions; ++i)
          m_ij[i][i] = 1.;
      }

      /**
       * @brief The flux function \f$F_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x)\f$ is implemented by this
       * method.
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model. It is however necessary to leave at least the NumberType, Vector,
       * and Vector_dot template parameters, as these can differ between calls (e.g. when doing automatic
       * differentiation).
       *
       * @note The standard implementation of this method simply sets \f$F_i = 0\f$.
       *
       * @note The meaning of this method depends on the discretization. For CG/DG/LDG it is the full flux of the
       * conservation law. For the Kurganov Tadmor scheme it is the *advection* flux, i.e. the hyperbolic part which
       * determines the numerical face flux and the wave speeds; the parabolic part is implemented separately in
       * `diffusion_flux`. In the Kurganov Tadmor case the assembler evaluates this callback separately for the minus
       * and plus face traces, and the derivatives are the face gradients produced by the active advection
       * reconstructor; they are distinct from the corrected gradients handed to `diffusion_flux`.
       *
       * @param F_i the resulting flux function \f$F_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param sol a `std::tuple<...>` which contains
       * 1. the array u_j
       * 2. the array of arrays \f$\partial_x u_j\f$
       * 3. the array of arrays of arrays \f$\partial_x^2 u_j\f$ (CG/DG/LDG only)
       * 4. the array of extractors \f$e_b\f$ (CG/DG/LDG only)
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief If the Kurganov Tadmor Scheme is used, this is the implementation of the diffusion (parabolic) part of
       * the face flux. \f$D_i(u_j, \partial_x u_j, \partial_x^3 u_j, x)\f$
       *
       * @remarks The assembler evaluates this callback separately for the minus and plus face traces, using the
       * corrected gradients of the diffusion reconstructor; these are distinct from the advection face gradients
       * handed to `flux`. This method is never called by the CG/DG/LDG discretizations.
       *
       * @note The standard implementation of this method simply sets \f$D_i = 0\f$.
       *
       * @note Sign convention: the face flux is \f$(H + D)\cdot n\f$, i.e. the advection numerical flux \f$H\f$ (built
       * from `flux`) and the diffusion flux \f$D\f$ (from this method) are SUMMED. Both methods therefore return the
       * physical flux with the same sign - exactly the conservation-law convention used by CG / LLFFlux. A diffusion
       * flux \f$f_{diff}\f$ must be a DECREASING function of the gradient (\f$\partial f_{diff} / \partial (\partial
       * u) < 0\f$) for forward diffusion, e.g. \f$f_{diff} = -\nu\, \partial u\f$ for the heat/viscous term.
       *
       * @param F_i the resulting diffusion flux \f$D_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param sol a `std::tuple<...>` which contains
       * 1. the array u_j
       * 2. the array of arrays \f$\partial_x u_j\f$
       * 3. the array of arrays of arrays \f$\partial_x^3 u_j\f$
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void diffusion_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                          [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief The source function \f$s_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x)\f$ is implemented by
       * this method.
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the
       * types are consistent with the rest of the model. It is however necessary to leave at least the NumberType,
       * Vector, and Vector_dot template parameters, as these can differ between calls (e.g. when doing automatic
       * differentiation).
       *
       * @note The standard implementation of this method simply sets \f$s_i = 0\f$.
       *
       * @param s_i the resulting source function \f$s_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param sol a `std::tuple<...>` which contains
       * 1. the array u_j
       * 2. the array of arrays \f$\partial_x u_j\f$
       * 3. the array of arrays of arrays \f$\partial_x^2 u_j\f$
       * 4. the array of extractors \f$e_b\f$
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void source([[maybe_unused]] std::array<NumberType, n_fe_functions> &s_i, [[maybe_unused]] const Point<dim> &x,
                  [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief A method to find out which components of the mass function are differential when using a DAE.
       *
       * @note The standard implementation of this method tests whether the mass function changes when changing the time
       * derivative of one component slightly. For highly complicated models, this method might not be able to set all
       * differential components correctly.
       *
       * @return std::vector<bool> with `true` for differential components and `false` for algebraic components.
       */
      template <uint dim> std::vector<bool> differential_components() const
      {
        std::vector<bool> differential_components(Model::Components::count_fe_functions(), false);

        // First we need two reference solutions u_i and dt_u_i, which we then both fill with 1.s
        std::array<double, Model::Components::count_fe_functions()> u_i{{}};
        std::array<double, Model::Components::count_fe_functions()> dt_u_i{{}};
        for (uint i = 0; i < Model::Components::count_fe_functions(); ++i) {
          u_i[i] = 1.;
          dt_u_i[i] = 1.;
        }
        // Set the point to be at 1. in all directions
        Point<dim> x;
        for (uint i = 0; i < dim; ++i)
          x[i] = 1.;
        // Get the mass function m_i
        std::array<double, Model::Components::count_fe_functions()> m_i{{}};
        asImp().mass(m_i, x, u_i, dt_u_i);

        // Now we check which components are differential by changing dt_u_i slightly and checking whether the mass
        // function changes.
        for (uint i = 0; i < Model::Components::count_fe_functions(); ++i) {
          dt_u_i[i] = 1. + 1e-1;
          std::array<double, Model::Components::count_fe_functions()> m_i_new{{}};
          asImp().mass(m_i_new, x, u_i, dt_u_i);
          dt_u_i[i] = 1.;
          for (uint j = 0; j < Model::Components::count_fe_functions(); ++j)
            if (!is_close(m_i[j], m_i_new[j])) differential_components[j] = true;
        }

        return differential_components;
      }

      //@}
      /**
       * @name Other variables
       */
      //@{

      template <typename Vector> void initial_condition_variables([[maybe_unused]] Vector &v_a) const
      {
        // Just to avoid warnings
      }

      template <typename Vector, typename Solution>
      void dt_variables([[maybe_unused]] Vector &r_a, [[maybe_unused]] const Solution &sol) const
      {
        // Just to avoid warnings
      }

      //@}
      /**
       * @name Extractors
       */
      //@{

      template <int dim, typename Vector, typename Solutions>
      void extract([[maybe_unused]] Vector &result, [[maybe_unused]] const Point<dim> &x,
                   [[maybe_unused]] const Solutions &sol) const
      {
      }

      //@}
      /**
       * @name LDG equations
       */
      //@{

      /**
       * @brief The LDG flux function \f$F^{LDG}_i(u_j, x),\,i>0\f$ is implemented by this method.
       *
       * The assembler constructs the i-th LDG function l_i from the i-1-th level
       * as \f[l_i = \partial_x F^{LDG}_i(l_{i-1}, x) + s^{LDG}_i(l_{i-1}, x)\f]
       * Here, \f$l_0\f$ is the solution itself (with all its components).
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model.
       *
       * @note The standard implementation of this method simply sets \f$F^{LDG}_i = 0\f$.
       *
       * @tparam dependent the index \f$i\f$ of the dependent variable \f$l_i\f$ which is constructed from the previous
       * level \f$l_{i-1}\f$.
       *
       * @param F the resulting LDG flux function \f$F^{LDG}_i\f$, with n_fe_functions_dep components.
       * This method should fill this argument with the desired structure of the flow equation.
       *
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param u the field values of \f$l_j(x)\f$ at the point `x`.
       *
       */
      template <uint dependent, int dim, typename NumberType, typename Vector, size_t n_fe_functions_dep>
      void ldg_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions_dep> &F,
                    [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Vector &u) const
      {
      }

      /**
       * @brief The LDG source function \f$s^{LDG}_i(u_j, x),\,i>0\f$ is implemented by this method.
       *
       * The assembler constructs the i-th LDG function l_i from the i-1-th level
       * as \f[l_i = \partial_x F^{LDG}_i(l_{i-1}, x) + s^{LDG}_i(l_{i-1}, x)\f]
       * Here, \f$l_0\f$ is the solution itself (with all its components).
       *
       * @remarks Note, that the precise template structure is not important, the only important thing is that the types
       * are consistent with the rest of the model.
       *
       * @note The standard implementation of this method simply sets \f$s^{LDG}_i = 0\f$.
       *
       * @tparam dependent the index \f$i\f$ of the dependent variable \f$l_i\f$ which is constructed from the previous
       * level \f$l_{i-1}\f$.
       *
       * @param s the resulting LDG source function \f$s^{LDG}_i\f$, with n_fe_functions_dep components.
       * This method should fill this argument with the desired structure of the flow equation.
       *
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param u the field values of \f$l_j(x)\f$ at the point `x`.
       *
       */
      template <uint dependent, int dim, typename NumberType, typename Vector, size_t n_fe_functions_dep>
      void ldg_source([[maybe_unused]] std::array<NumberType, n_fe_functions_dep> &s,
                      [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Vector &u) const
      {
      }

      template <int dim, typename NumberType, typename Solutions_s, typename Solutions_n>
      void face_indicator([[maybe_unused]] std::array<NumberType, 2> &indicator,
                          [[maybe_unused]] const Tensor<1, dim> &normal, [[maybe_unused]] const Point<dim> &p,
                          [[maybe_unused]] const Solutions_s &sol_s, [[maybe_unused]] const Solutions_n &sol_n) const
      {
      }

      template <int dim, typename NumberType, typename Solution>
      void cell_indicator([[maybe_unused]] NumberType &indicator, [[maybe_unused]] const Point<dim> &p,
                          [[maybe_unused]] const Solution &sol) const
      {
      }

      template <int dim, typename Vector>
      std::array<double, dim> EoM([[maybe_unused]] const Point<dim> &x, const Vector &u) const
      {
        return std::array<double, dim>{{u[0]}};
      }

      template <int dim, typename Vector> Point<dim> EoM_postprocess(const Point<dim> &EoM, const Vector &) const
      {
        return EoM;
      }

      template <typename FUN, typename DataOut> void readouts_multiple(FUN &helper, DataOut &) const
      {
        helper("primary", [&](const auto &x, const auto &u_i) { return asImp().EoM(x, u_i); }, // chiral EoM
               [&](auto &output, const auto &x, const auto &sol) { asImp().readouts(output, x, sol); });
      }

      template <int dim, typename DataOut, typename Solutions>
      void readouts([[maybe_unused]] DataOut &output, [[maybe_unused]] const Point<dim> &x,
                    [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief Add affine constraints to the FE/DG system before sparsity patterns and operators are rebuilt.
       *
       * Boundary-only constraints should use `apply_boundary_affine_constraints(constraints, context)` and inspect
       * `context.template boundary<"u">()`. Constraints that may need interior support points should use
       * `apply_affine_constraints(constraints, context)` and inspect `context.template support<"u">()`.
       *
       * The origin helpers use `x[0]` as their signed origin coordinate in one-dimensional domains. In
       * multidimensional domains, models using these helpers must provide
       * `Model::OriginConstraintCoordinate<component_name>::signed_coordinate(point)` to define the zero level set
       * that should be constrained for each named component.
       */
      template <typename Constraints, typename Context>
      void affine_constraints(Constraints &constraints, const Context &context) const
      {
        if constexpr (requires(const Model &model, Constraints &constraint_matrix,
                               const Context &affine_constraint_context) {
                        model.apply_boundary_affine_constraints(constraint_matrix, affine_constraint_context);
                      })
          asImp().apply_boundary_affine_constraints(constraints, context);
        if constexpr (requires(const Model &model, Constraints &constraint_matrix,
                               const Context &affine_constraint_context) {
                        model.apply_affine_constraints(constraint_matrix, affine_constraint_context);
                      })
          asImp().apply_affine_constraints(constraints, context);
      }

    };

    namespace internal
    {
      template <typename> inline constexpr bool dependent_false_v = false;

      template <FixedString component_name, typename Model, int dim>
      double origin_constraint_coordinate(const Point<dim> &point)
      {
        if constexpr (dim == 1) {
          return point[0];
        } else {
          if constexpr (requires {
                          Model::template OriginConstraintCoordinate<component_name>::signed_coordinate(point);
                        }) {
            return Model::template OriginConstraintCoordinate<component_name>::signed_coordinate(point);
          } else {
            static_assert(dependent_false_v<Model>,
                          "Multidimensional origin affine-constraint helpers require Model::"
                          "OriginConstraintCoordinate<component_name>::signed_coordinate(point).");
            return 0.0;
          }
        }
      }

      template <FixedString component_name, typename Model, typename Context>
      std::vector<types::global_dof_index> select_origin_candidates([[maybe_unused]] const Context &context,
                                                                    const auto &view)
      {
        constexpr int dim = Context::dimension;
        static_assert(dim == 1 || dim == 2,
                      "Origin affine-constraint helpers currently support only one- and two-dimensional domains.");
        static_assert(Context::template component_size<component_name>() == 1,
                      "Origin affine-constraint helpers require a scalar FE-function component.");

        double best_abs_coordinate = std::numeric_limits<double>::infinity();
        bool has_non_negative_best = false;

        for (uint i = 0; i < view.dofs.n_elements(); ++i) {
          const double coordinate = origin_constraint_coordinate<component_name, Model>(view.points[i]);
          const double abs_coordinate = std::abs(coordinate);
          if (abs_coordinate < best_abs_coordinate) {
            best_abs_coordinate = abs_coordinate;
            has_non_negative_best = coordinate >= 0.0;
          } else if (abs_coordinate == best_abs_coordinate && coordinate >= 0.0) {
            has_non_negative_best = true;
          }
        }

        std::vector<types::global_dof_index> candidates;
        for (uint i = 0; i < view.dofs.n_elements(); ++i) {
          const double coordinate = origin_constraint_coordinate<component_name, Model>(view.points[i]);
          const double abs_coordinate = std::abs(coordinate);
          if (abs_coordinate != best_abs_coordinate) continue;
          if ((coordinate >= 0.0) != has_non_negative_best) continue;
          candidates.push_back(view.dofs.nth_index_in_set(i));
        }

        return candidates;
      }

      template <FixedString component_name, typename Context>
      std::optional<types::global_dof_index> select_origin_candidate(const Context &context, const auto &view)
      {
        constexpr int dim = Context::dimension;
        static_assert(dim == 1, "select_origin_candidate supports only one-dimensional domains; use "
                                "select_origin_candidates for multi-dimensional domains.");

        const auto candidates = select_origin_candidates<component_name, void>(context, view);
        if (candidates.empty()) return std::nullopt;
        return candidates.front();
      }
    } // namespace internal

    /**
     * @brief Constrain the boundary dofs of a named scalar FE-function component nearest its origin coordinate to zero.
     *
     * In one dimension, the origin coordinate is `x[0]`. In multidimensional domains, `Model` must provide an
     * `OriginConstraintCoordinate<component_name>` policy whose `signed_coordinate(point)` method defines the zero
     * level set. All boundary dofs on the nearest discrete zero level set are constrained, with symmetric ties
     * resolved toward the non-negative side.
     */
    template <FixedString component_name, typename Model> class ConstrainOriginBoundaryPointToZero
    {
    public:
      template <typename Constraints, typename Context>
      void apply_boundary_affine_constraints(Constraints &constraints, const Context &context) const
      {
        const auto candidates =
            internal::select_origin_candidates<component_name, Model>(context,
                                                                      context.template boundary<component_name>());
        for (const auto dof : candidates) {
          constraints.add_line(dof);
          constraints.set_inhomogeneity(dof, 0.0);
        }
      }
    };

    /**
     * @brief Constrain the support dofs of a named scalar FE-function component nearest its origin coordinate to zero.
     *
     * This is useful for cell-centered DG0/FV layouts where `sigma = 0` is not itself a boundary support point.
     * In multidimensional domains, `Model::OriginConstraintCoordinate<component_name>::signed_coordinate(point)`
     * defines the zero level set to constrain.
     */
    template <FixedString component_name, typename Model> class ConstrainOriginSupportPointToZero
    {
    public:
      template <typename Constraints, typename Context>
      void apply_affine_constraints(Constraints &constraints, const Context &context) const
      {
        const auto candidates =
            internal::select_origin_candidates<component_name, Model>(context,
                                                                      context.template support<component_name>());
        for (const auto dof : candidates) {
          constraints.add_line(dof);
          constraints.set_inhomogeneity(dof, 0.0);
        }
      }
    };

    template <typename Model> class NoAffineConstraints
    {
    };

    class Time
    {
    public:
      void set_time(double t);
      const double &get_time() const;

    protected:
      double t;
    };

    /**
     * @brief The fRG class is used to keep track of the RG time and the cutoff scale.
     */
    class fRG
    {
    public:
      /**
       * @brief Construct a new fRG object from a given initial cutoff scale
       *
       * @param Lambda the initial cutoff scale of the fRG
       */
      fRG(double Lambda);

      /**
       * @brief Construct a new fRG object from a given ConfigTree object
       *
       * @param json the JSON object containing the initial cutoff scale at "/physical/Lambda"
       */
      fRG(const ConfigTree &json);

      /**
       * @brief Set the time of the fRG object, updating the cutoff scale and its powers
       *
       * @param t the time to set
       */
      void set_time(double t);

      /**
       * @brief Get the time of the fRG object
       *
       * @return const double& the time of the fRG object
       */
      const double &get_time() const;

    protected:
      const double Lambda;
      double t = 0., k = 0., k2 = 0., k3 = 0., k4 = 0., k5 = 0., k6 = 0.;
      bool time_initialized = false;
    };
  } // namespace def
} // namespace DiFfRG
