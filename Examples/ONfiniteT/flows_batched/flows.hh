#pragma once

#include "DiFfRG/common/utils.hh"
#include "DiFfRG/physics/integration.hh"
#include "./V/V.hh"
#include "./V_GPU/V_GPU.hh"
#include "./V_GPU_f/V_GPU_f.hh"

class ONFiniteTBatchedFlows
{
  public:
  ONFiniteTBatchedFlows(const DiFfRG::ConfigTree& config)
  ;

  void set_k(const double k)
  ;

  void set_T(const double T)
  ;

  void set_typical_E(const double E)
  ;

  void set_x_extent(const double x_extent)
  ;

  DiFfRG::QuadratureProvider quadrature_provider;

  V_integrator V;

  V_GPU_integrator V_GPU;

  V_GPU_f_integrator V_GPU_f;
};