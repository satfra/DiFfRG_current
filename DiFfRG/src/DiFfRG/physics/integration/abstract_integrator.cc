#include <DiFfRG/physics/integration/abstract_integrator.hh>

namespace DiFfRG
{
  void AbstractIntegrator::set_node_distribution(const NodeDistribution &distribution)
  {
    node_distribution = distribution;
  }

  const NodeDistribution &AbstractIntegrator::get_node_distribution() const { return node_distribution; }

  void AbstractIntegrator::set_load_balancer(IntegrationLoadBalancer &load_balancer)
  {
    load_balancer.register_integrator(*this);
  }
} // namespace DiFfRG