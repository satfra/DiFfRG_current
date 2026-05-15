#pragma once

// external libraries
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>
#include <DiFfRG/model/model.hh>

namespace DiFfRG
{
  namespace Testing
  {
    using namespace dealii;

    struct PhysicalParameters {
      std::array<double, 3> initial_x0{{0., 0., 0.}};
      std::array<double, 3> initial_x1{{0., 0., 0.}};
      std::array<double, 3> initial_x2{{0., 0., 0.}};
      std::array<double, 3> initial_x3{{0., 0., 0.}};
    };

    template <uint components> struct compFactory {
      using value = ComponentDescriptor<FEFunctionDescriptor<>>;
    };

    template <> struct compFactory<1> {
      using value = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>;
    };

    template <> struct compFactory<2> {
      using value = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>;
    };

    template <> struct compFactory<3> {
      using value = ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">, Scalar<"w">>>;
    };

    template <uint dim, uint components = 1>
    class ModelConstant
        : public def::AbstractModel<ModelConstant<dim, components>, typename compFactory<components>::value>,
          public def::Time,                                           // this handles time
          public def::NoNumFlux<ModelConstant<dim, components>>,      // use no numflux
          public def::FlowBoundaries<ModelConstant<dim, components>>, // use Inflow/Outflow boundaries
          public def::AD<ModelConstant<dim, components>>              // define all jacobians per AD
    {
    public:
      const PhysicalParameters prm;

      ModelConstant(PhysicalParameters prm) : prm(prm) {}
      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        for (uint c = 0; c < components; ++c)
          values[c] = prm.initial_x0[c] + prm.initial_x1[c] * pos[c];
      }

      template <typename Vector> std::array<double, dim> EoM(const Point<dim> &x, const Vector &u) const
      {
        // Just to avoid warnings
        (void)x;
        if constexpr (dim == 1)
          return std::array<double, dim>{{u[0]}};
        else if constexpr (dim == 2) {
          return std::array<double, dim>{{u[0], u[1]}};
        } else if constexpr (dim == 3)
          return std::array<double, dim>{{u[0], u[1], u[2]}};
        else
          throw std::runtime_error("Only 1, 2, and 3 dimensions are supported.");
      }

      double solution(const Point<dim> &pos) const { return prm.initial_x0[0] + prm.initial_x1[0] * pos[0]; }
    };

    template <uint dim>
    class LDGModelConstant
        : public def::AbstractModel<LDGModelConstant<dim>,
                                    ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<>,
                                                        ExtractorDescriptor<>, FEFunctionDescriptor<Scalar<"l">>>>,
          public def::Time,                                  // this handles time
          public def::NoNumFlux<LDGModelConstant<dim>>,      // use no numflux
          public def::FlowBoundaries<LDGModelConstant<dim>>, // use Inflow/Outflow boundaries
          public def::AD<LDGModelConstant<dim>>              // define all jacobians per AD
    {
    public:
      const PhysicalParameters prm;

      LDGModelConstant(PhysicalParameters prm) : prm(prm) {}
      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }
      double solution(const Point<dim> &pos) const { return prm.initial_x0[0] + prm.initial_x1[0] * pos[0]; }
      template <uint dependent, int mdim, typename NumberType, typename Solutions_s, typename Solutions_n>
      void ldg_numflux(std::array<Tensor<1, mdim, NumberType>, 1> &, const Tensor<1, mdim> &, const Point<mdim> &,
                       const Solutions_s &, const Solutions_n &) const
      {
      }
    };

    template <uint dim>
    class ModelExp : public def::AbstractModel<ModelExp<dim>, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
                     public def::Time,                          // this handles time
                     public def::NoNumFlux<ModelExp<dim>>,      // use no numflux
                     public def::FlowBoundaries<ModelExp<dim>>, // use Inflow/Outflow boundaries
                     public def::AD<ModelExp<dim>>              // define all jacobians per AD
    {
    protected:
      const PhysicalParameters prm;

    public:
      ModelExp(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      double solution(const Point<dim> &pos) const
      {
        return std::exp(t) * (prm.initial_x0[0] + prm.initial_x1[0] * pos[0]);
      }

      template <typename NT, typename Solution>
      void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        s_i[0] = -fe_functions[0];
      }
    };

    template <uint dim>
    class ModelBurgers
        : public def::AbstractModel<ModelBurgers<dim>, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
          public def::Time,                              // this handles time
          public def::LLFFlux<ModelBurgers<dim>>,        // use LL numflux
          public def::FlowBoundaries<ModelBurgers<dim>>, // use Inflow/Outflow boundaries
          public def::AD<ModelBurgers<dim>>              // define all jacobians per AD
    {
    protected:
      const PhysicalParameters prm;

    public:
      ModelBurgers(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      double solution(const Point<dim> &pos) const
      {
        return (prm.initial_x0[0] + prm.initial_x1[0] * pos[0]) / (prm.initial_x1[0] * t + 1.);
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
      }
    };

    // A coupled FEM + explicit-variable model used to exercise the hybrid IDA + explicit
    // timesteppers (TimeStepperSUNDIALS_IDA_Boost{ABM,RK}).
    //
    // The FEM function u obeys the logistic ODE  du/dt = K u (1 - u)  at every spatial
    // point, i.e. u(t) = 1 / (1 + exp(-K (t - t_mid))).  For a large K this is a sharp
    // transition centred at t_mid: u stays near 0 for a long time and then rises steeply
    // to 1.  The implicit IDA controller grows its step in the flat region and is then
    // forced to reject and retry steps as it crosses the transition.
    //
    // The explicit variable v obeys  dv/dt = -v u(t)^2 , i.e. it is coupled to the FEM
    // solution.  Using  u^2 = u - (1/K) du/dt  and  \int u dt = (1/K) ln(1 + e^{K(t-t_mid)}),
    // its exact solution is
    //     v(t) = v0 exp( -[ Iu(t) - Iu(0) - (u(t) - u(0)) / K ] ),
    //     Iu(t) = (1/K) ln(1 + e^{K (t - t_mid)}).
    //
    // If a rejected IDA step is allowed to leak into the explicit-variable buffer, the
    // variable is integrated along a trajectory that IDA never accepted and v(t_final)
    // deviates from the exact value above.  That is the regression this model guards
    // against.
    template <uint dim>
    class ModelHybridRollback
        : public def::AbstractModel<
              ModelHybridRollback<dim>,
              ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>, VariableDescriptor<Scalar<"v">>,
                                  ExtractorDescriptor<Scalar<"u_eom">>>>,
          public def::Time,                                  // this handles time
          public def::NoNumFlux<ModelHybridRollback<dim>>,   // pure source term, no spatial coupling
          public def::FlowBoundaries<ModelHybridRollback<dim>>,
          public def::AD<ModelHybridRollback<dim>>           // define all jacobians per AD
    {
    public:
      static constexpr double K = 30.;     // sharpness of the FEM transition
      static constexpr double t_mid = 0.5; // location of the FEM transition
      static constexpr double v0 = 1.;     // initial value of the explicit variable

      const PhysicalParameters prm;
      ModelHybridRollback(PhysicalParameters prm) : prm(prm) {}

      static double u_exact(double tt) { return 1. / (1. + std::exp(-K * (tt - t_mid))); }
      // antiderivative of u_exact
      static double Iu(double tt) { return std::log1p(std::exp(K * (tt - t_mid))) / K; }

      template <typename Vector> void initial_condition(const Point<dim> & /*pos*/, Vector &values) const
      {
        values[0] = u_exact(0.);
      }
      template <typename Vector> void initial_condition_variables(Vector &values) const { values[0] = v0; }

      // du/dt = K u (1 - u);  the residual convention is u_dot = -source.
      template <typename NT, typename Solution>
      void source(std::array<NT, 1> &s_i, const Point<dim> & /*p*/, const Solution &sol) const
      {
        const auto u = get<0>(sol)[0];
        s_i[0] = -K * u * (1. - u);
      }

      // u is spatially constant, so the extraction location is irrelevant; a fixed, well
      // defined root at x = 0.5 simply keeps the EoM search well-posed.
      template <int d, typename Vector> std::array<double, 1> EoM(const Point<d> &x, const Vector & /*u*/) const
      {
        return {{x[0] - 0.5}};
      }

      // hand the FEM solution at the EoM point to the explicit-variable residual
      template <typename NT, typename Solution>
      void extract(std::array<NT, 1> &extractors, const Point<dim> & /*x*/, const Solution &sol) const
      {
        extractors[0] = get<"fe_functions">(sol)[0];
      }

      // dv/dt = -v u^2;  the residual convention is v_dot = -r_a.
      template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const
      {
        const auto u = get<"extractors">(sol)[0];
        const auto v = get<"variables">(sol)[0];
        r_a[0] = v * u * u;
      }

      // exact FEM solution (validated by the standard test harness, block 0)
      double solution(const Point<dim> & /*pos*/) const { return u_exact(t); }
      // exact explicit-variable solution (validated by the dedicated test, block 1)
      double variable_solution() const
      {
        // \int_0^t u^2 dt' = (Iu(t) - Iu(0)) - (u(t) - u(0)) / K
        const double int_u2 = (Iu(t) - Iu(0.)) - (u_exact(t) - u_exact(0.)) / K;
        return v0 * std::exp(-int_u2);
      }
    };
  } // namespace Testing
} // namespace DiFfRG