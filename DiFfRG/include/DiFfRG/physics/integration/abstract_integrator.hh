#pragma once

// DiFfRG
#include "distribution.hh"
#include <DiFfRG/physics/integration/distribution.hh>

namespace DiFfRG
{
  class AbstractIntegrator
  {
  public:
    void set_node_distribution(const NodeDistribution &distribution);

    const NodeDistribution &get_node_distribution() const;

    void set_load_balancer(IntegrationLoadBalancer &load_balancer);

  protected:
    NodeDistribution node_distribution;
  };
} // namespace DiFfRG