#pragma once

// external libraries
#include <array>
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

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        return {prm.initial_x0[0] + prm.initial_x1[0] * pos[0]};
      }
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
      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        return prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }
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

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        return {std::exp(t) * (prm.initial_x0[0] + prm.initial_x1[0] * pos[0])};
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

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        return {(prm.initial_x0[0] + prm.initial_x1[0] * pos[0]) / (prm.initial_x1[0] * t + 1.)};
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
      }
    };

    template <uint dim>
    class ModelBurgersKT
        : public def::AbstractModel<ModelBurgersKT<dim>, ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
          public def::Time,                                // this handles time
          public def::LLFFlux<ModelBurgersKT<dim>>,        // use LL numflux
          public def::FlowBoundaries<ModelBurgersKT<dim>>, // use Inflow/Outflow boundaries
          public def::FVDefaultBoundaries<ModelBurgersKT<dim>>,
          public def::AD<ModelBurgersKT<dim>>              // define all jacobians per AD
    {
    protected:
      const PhysicalParameters prm;

    public:
      ModelBurgersKT(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        return {(prm.initial_x0[0] + prm.initial_x1[0] * pos[0]) / (prm.initial_x1[0] * t + 1.)};
      }

      template <typename NT, typename Solution>
      void KurganovTadmor_advection_flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/,
                                         const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
      }

      template <int mdim, typename NumberType, size_t n_components, size_t n_faces>
      void apply_boundary_conditions(std::array<std::array<NumberType, n_components>, n_faces> &u_neighbors,
                                     std::array<Point<mdim>, n_faces> &x_neighbors,
                                     const std::array<types::boundary_id, n_faces> &boundary_ids,
                                     const std::array<Point<mdim>, n_faces> &face_centers,
                                     const std::array<NumberType, n_components> & /*u_cell*/,
                                     const Point<mdim> & /*x_cell*/) const
      {
        for (size_t f = 0; f < n_faces; ++f)
          if (boundary_ids[f] != numbers::internal_face_boundary_id) {
            x_neighbors[f] = face_centers[f];
            u_neighbors[f][0] =
                (prm.initial_x0[0] + prm.initial_x1[0] * x_neighbors[f][0]) / (prm.initial_x1[0] * t + 1.);
          }
      }
    };

    template <uint dim>
    class ModelBurgersTravelingWaveKT
        : public def::AbstractModel<ModelBurgersTravelingWaveKT<dim>,
                                    ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">>>>,
          public def::Time,
          public def::LLFFlux<ModelBurgersTravelingWaveKT<dim>>,
          public def::FlowBoundaries<ModelBurgersTravelingWaveKT<dim>>,
          public def::FVDefaultBoundaries<ModelBurgersTravelingWaveKT<dim>>,
          public def::AD<ModelBurgersTravelingWaveKT<dim>>
    {
    protected:
      static constexpr double nu = 10.0;
      static constexpr double f_plus = 2.0;
      static constexpr double f_minus = 0.0;
      static constexpr double c = (f_plus + f_minus) / 2.0;

    public:
      ModelBurgersTravelingWaveKT(PhysicalParameters) {}

      /// Steadily propagating traveling wave: u(x,0) = 2 / (1 + exp(x / nu))
      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        values[0] = f_plus / (1.0 + std::exp(pos[0] / nu));
      }

      /// Exact solution: u(x,t) = 2 / (1 + exp((x - c*t) / nu))
      std::array<double, 1> solution(const Point<dim> &pos) const
      {
        return {f_plus / (1.0 + std::exp((pos[0] - c * t) / nu))};
      }

      template <typename NT, typename Solution>
      void KurganovTadmor_advection_flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/,
                                         const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 1> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_derivatives = get<1>(sol);
        F_i[0] = nu * fe_derivatives[0];
      }

      template <int mdim, typename NumberType, size_t n_comp, size_t n_faces_bc>
      void apply_boundary_conditions(std::array<std::array<NumberType, n_comp>, n_faces_bc> &u_neighbors,
                                     std::array<Point<mdim>, n_faces_bc> &x_neighbors,
                                     const std::array<types::boundary_id, n_faces_bc> &boundary_ids,
                                     const std::array<Point<mdim>, n_faces_bc> &face_centers,
                                     const std::array<NumberType, n_comp> & /*u_cell*/,
                                     const Point<mdim> & /*x_cell*/) const
      {
        for (size_t f = 0; f < n_faces_bc; ++f)
          if (boundary_ids[f] != numbers::internal_face_boundary_id) {
            x_neighbors[f] = face_centers[f];
            u_neighbors[f][0] = f_plus / (1.0 + std::exp((x_neighbors[f][0] - c * t) / nu));
          }
      }
    };

    template <uint dim>
    class ModelTwoComponentBurgersKT
        : public def::AbstractModel<ModelTwoComponentBurgersKT<dim>,
                                    ComponentDescriptor<FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>>>,
          public def::Time,
          public def::LLFFlux<ModelTwoComponentBurgersKT<dim>>,
          public def::FlowBoundaries<ModelTwoComponentBurgersKT<dim>>,
          public def::FVDefaultBoundaries<ModelTwoComponentBurgersKT<dim>>,
          public def::AD<ModelTwoComponentBurgersKT<dim>>
    {
    protected:
      const PhysicalParameters prm;
      static constexpr double nu = 10.0;
      static constexpr double f_plus = 2.0;
      static constexpr double f_minus = 0.0;
      static constexpr double c = (f_plus + f_minus) / 2.0;

    public:
      ModelTwoComponentBurgersKT(PhysicalParameters prm) : prm(prm) {}

      template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
      {
        // Component 0 (u): BurgersTravelingWaveKT initial condition
        values[0] = f_plus / (1.0 + std::exp(pos[0] / nu));
        // Component 1 (v): BurgersKT initial condition
        values[1] = prm.initial_x0[0] + prm.initial_x1[0] * pos[0];
      }

      /// Component 0 (u): exact solution for traveling wave
      /// Component 1 (v): exact solution for Burgers
      std::array<double, 2> solution(const Point<dim> &pos) const
      {
        const double u_sol = f_plus / (1.0 + std::exp((pos[0] - c * t) / nu));
        const double v_sol = (prm.initial_x0[0] + prm.initial_x1[0] * pos[0]) / (prm.initial_x1[0] * t + 1.);
        return {u_sol, v_sol};
      }

      template <typename NT, typename Solution>
      void KurganovTadmor_advection_flux(std::array<Tensor<1, dim, NT>, 2> &F_i, const Point<dim> & /*pos*/,
                                         const Solution &sol) const
      {
        const auto &fe_functions = get<0>(sol);
        // Component 0 (u): traveling wave advection flux
        F_i[0][0] = 0.5 * powr<2>(fe_functions[0]);
        // Component 1 (v): Burgers advection flux
        F_i[1][0] = 0.5 * powr<2>(fe_functions[1]);
      }

      template <typename NT, typename Solution>
      void flux(std::array<Tensor<1, dim, NT>, 2> &F_i, const Point<dim> & /*pos*/, const Solution &sol) const
      {
        const auto &fe_derivatives = get<1>(sol);
        // Component 0 (u): traveling wave diffusion flux
        F_i[0] = nu * fe_derivatives[0];
        // Component 1 (v): no diffusion for BurgersKT
        F_i[1] = 0.0;
      }

      template <int mdim, typename NumberType, size_t n_components, size_t n_faces>
      void apply_boundary_conditions(std::array<std::array<NumberType, n_components>, n_faces> &u_neighbors,
                                     std::array<Point<mdim>, n_faces> &x_neighbors,
                                     const std::array<types::boundary_id, n_faces> &boundary_ids,
                                     const std::array<Point<mdim>, n_faces> &face_centers,
                                     const std::array<NumberType, n_components> & /*u_cell*/,
                                     const Point<mdim> & /*x_cell*/) const
      {
        for (size_t f = 0; f < n_faces; ++f)
          if (boundary_ids[f] != numbers::internal_face_boundary_id) {
            x_neighbors[f] = face_centers[f];
            // Component 0 (u): traveling wave boundary condition
            u_neighbors[f][0] = f_plus / (1.0 + std::exp((x_neighbors[f][0] - c * t) / nu));
            // Component 1 (v): Burgers boundary condition
            u_neighbors[f][1] =
                (prm.initial_x0[0] + prm.initial_x1[0] * x_neighbors[f][0]) / (prm.initial_x1[0] * t + 1.);
          }
      }
    };

  } // namespace Testing
} // namespace DiFfRG
