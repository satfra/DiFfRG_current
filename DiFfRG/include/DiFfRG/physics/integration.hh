#pragma once

#include <DiFfRG/common/quadrature/quadrature_provider.hh>

#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

#include <DiFfRG/physics/integration/vacuum/integrator_p2.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_1ang.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_4D_2ang.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_4D_3ang.hh>

#include <DiFfRG/physics/integration/finiteT/integrator_fT.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_1ang.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_4D_2ang.hh>

#include <DiFfRG/physics/integration/lattice/integrator_lat_1D.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat_2D.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat_3D.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat_4D.hh>

namespace DiFfRG
{
  template <typename T>
  concept has_set_k = requires(T t, double k) { t.set_k(k); };

  template <typename T>
  concept has_integrator_AD = requires(T t) { t.integrator_AD; };
} // namespace DiFfRG