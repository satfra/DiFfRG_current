#pragma once

// external libraries
#include "DiFfRG/common/json.hh"
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// DiFfRG
#include <DiFfRG/model/ad.hh>
#include <DiFfRG/model/component_descriptor.hh>
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
     * To see how the models are used, refer to the DiFfRG::AbstractAssembler class and the [guide](#HowTo).
     *
     * @tparam Model The model which implements this interface. (CRTP)
     * @tparam Components_ The components of the model, this must be a DiFfRG::ComponentDescriptor.
     */
    template <typename Model, typename Components_> class AbstractModel
    {
      Model &asImp() { return static_cast<Model &>(*this); }
      const Model &asImp() const { return static_cast<const Model &>(*this); }

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
       * @param F_i the resulting flux function \f$F_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param sol a `std::tuple<...>` which contains
       * 1. the array u_j
       * 2. the array of arrays \f$\partial_x u_j\f$
       * 3. the array of arrays of arrays \f$\partial_x^2 u_j\f$
       * 4. the array of extractors \f$e_b\f$
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol) const
      {
      }

      /**
       * @brief If the Kurganov Tadmor Scheme is used, this is the implementation of the advection flux. \f$F_i(u_j, x)$
       *
       * @remarks although the design of DiFfRG allows you to have a \f$\partial_x u$ dependent advection flux, the
       * implementation is not designed to handle such cases.
       *
       * @note The standard implementation of this method simply sets \f$F_i = 0\f$.
       *
       * @param F_i the resulting flux function \f$F_i\f$, with \f$N_f\f$ components.
       * This method should fill this argument with the desired structure of the flow equation.
       * @param x a d-dimensional dealii::Point<dim> representing field coordinates.
       * @param sol a `std::tuple<...>` which contains
       * 1. the array u_j
       * 2. the array of arrays \f$\partial_x u_j\f$
       * 3. the array of arrays of arrays \f$\partial_x^2 u_j\f$
       * 4. the array of extractors \f$e_b\f$
       */
      template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
      void KurganovTadmor_advection_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                                         [[maybe_unused]] const Point<dim> &x,
                                         [[maybe_unused]] const Solutions &sol) const
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
        helper([&](const auto &x, const auto &u_i) { return asImp().EoM(x, u_i); }, // chiral EoM
               [&](auto &output, const auto &x, const auto &sol) { asImp().readouts(output, x, sol); });
      }

      template <int dim, typename DataOut, typename Solutions>
      void readouts([[maybe_unused]] DataOut &output, [[maybe_unused]] const Point<dim> &x,
                    [[maybe_unused]] const Solutions &sol) const
      {
      }

      template <int dim, typename Constraints>
      void affine_constraints([[maybe_unused]] Constraints &constraints,
                              [[maybe_unused]] const std::vector<IndexSet> &component_boundary_dofs,
                              [[maybe_unused]] const std::vector<std::vector<Point<dim>>> &component_boundary_points)
      {
      }

      /**
       * @brief Applies boundary conditions for all boundary faces of a cell simultaneously.
       *
       * This allows the model to use existing internal neighbors to better reconstruct ghost states
       * (e.g. using the opposite neighbor for linear extrapolation).
       *
       * @param u_neighbors [in/out] Solution values of neighbors. Internals are pre-filled; boundaries must be filled
       * here.
       * @param x_neighbors [in/out] Positions of neighbors.
       * @param boundary_ids [in] Boundary IDs for boundary faces (undefined for internal).
       * @param face_centers [in] Centers of boundary faces (undefined for internal).
       * @param u_cell [in] Current cell solution.
       * @param x_cell [in] Current cell center.
       */
      template <int dim, typename NumberType, size_t n_components, size_t n_faces>
      void
      apply_boundary_conditions([[maybe_unused]] std::array<std::array<NumberType, n_components>, n_faces> &u_neighbors,
                                [[maybe_unused]] std::array<Point<dim>, n_faces> &x_neighbors,
                                [[maybe_unused]] const std::array<types::boundary_id, n_faces> &boundary_ids,
                                [[maybe_unused]] const std::array<Point<dim>, n_faces> &face_centers,
                                [[maybe_unused]] const std::array<NumberType, n_components> &u_cell,
                                [[maybe_unused]] const Point<dim> &x_cell) const
      {
      }

      /**
       * @brief Computes the gradient of the ghost cell at a boundary face.
       *
       * In a second-order Kurganov-Tadmor scheme, the numerical flux at interior faces uses
       * linearly reconstructed states from both sides. The reconstruction requires each cell's
       * gradient, which is computed from its neighbors via a minmod limiter. At a boundary, the
       * ghost cell has no real neighbors-of-neighbors, so this method lets the model provide
       * the ghost gradient directly.
       *
       * Common patterns:
       * - **Default (smooth extension):** ghost gradient = interior cell gradient. Preserves
       *   second-order accuracy for smooth solutions and outflow-type boundaries.
       * - **Dirichlet:** If the boundary value is known analytically, compute the gradient
       *   from the analytical solution.
       * - **Reflecting (wall):** Negate the component of the gradient normal to the boundary.
       * - **Zero-gradient (Neumann):** Set the normal component of the ghost gradient to zero.
       *
       * @param ghost_gradient [out] The gradient of the ghost cell to be filled.
       * @param boundary_id The boundary ID of the face.
       * @param normal The outward unit normal at the boundary face.
       * @param x_face The position of the boundary face (quadrature point).
       * @param u_ghost The ghost cell value (as set by apply_boundary_conditions).
       * @param x_ghost The ghost cell position (as set by apply_boundary_conditions).
       * @param u_cell The interior cell value.
       * @param x_cell The interior cell center.
       * @param cell_gradient The minmod-limited gradient of the interior cell.
       */
      template <int dim, typename NumberType, size_t n_components>
      void boundary_ghost_gradient(
          [[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_components> &ghost_gradient,
          [[maybe_unused]] const types::boundary_id boundary_id, [[maybe_unused]] const Tensor<1, dim> &normal,
          [[maybe_unused]] const Point<dim> &x_face,
          [[maybe_unused]] const std::array<NumberType, n_components> &u_ghost,
          [[maybe_unused]] const Point<dim> &x_ghost,
          [[maybe_unused]] const std::array<NumberType, n_components> &u_cell,
          [[maybe_unused]] const Point<dim> &x_cell,
          [[maybe_unused]] const std::array<Tensor<1, dim, NumberType>, n_components> &cell_gradient) const
      {
        // Default: mirror the interior cell's gradient (smooth extension)
        ghost_gradient = cell_gradient;
      }

      //@}
      // ...existing code...
      //@}
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
       * @brief Construct a new fRG object from a given JSONValue object
       *
       * @param json the JSON object containing the initial cutoff scale at "/physical/Lambda"
       */
      fRG(const JSONValue &json);

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
      double t, k, k2, k3, k4, k5, k6;
    };
  } // namespace def
} // namespace DiFfRG
