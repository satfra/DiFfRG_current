#pragma once

#include "AAAA/ZA4.hh"

#include "def.hh"
#include <DiFfRG/physics/flow_equations.hh>

template <template <typename, typename> class INT> class YangMillsFlowEquations : public FlowEquations
{
public:
  YangMillsFlowEquations(const JSONValue &json)
      : FlowEquations(json,
                      [](double x) { return powr<-1>(x + __REGULATOR__::RB(1., x)) * __REGULATOR__::RBdot(1., x); }),
        grid_size_int{{x_quadrature_order}}, grid_sizes_angle_int{{x_quadrature_order, 2 * angle_quadrature_order}},
        grid_sizes_3D_int{{x_quadrature_order, angle_quadrature_order, angle_quadrature_order}},
        grid_sizes_4D_int{{x_quadrature_order, angle_quadrature_order, angle_quadrature_order, angle_quadrature_order}},
        ZA4_integrator(quadrature_provider, grid_sizes_4D_int, x_extent, json)
  {
  }

private:
  const std::array<uint, 1> grid_size_int;
  const std::array<uint, 2> grid_sizes_angle_int;
  const std::array<uint, 3> grid_sizes_3D_int;
  const std::array<uint, 4> grid_sizes_4D_int;

public:
  ::DiFfRG::QuadratureProvider quadrature_provider;
  ::DiFfRG::Flows::ZA4_integrator<INT> ZA4_integrator;
};
