#include <DiFfRG/physics/integration/distribution.hh>

#include <numeric>

namespace DiFfRG
{
  NodeDistribution::NodeDistribution(MPI_Comm mpi_comm, const std::vector<uint> &sizes, const std::vector<uint> &nodes)
      : mpi_comm(mpi_comm), sizes(sizes), nodes(nodes)
  {
    total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
  }

  IntegrationLoadBalancer::IntegrationLoadBalancer(MPI_Comm mpi_comm) : mpi_comm(mpi_comm) {}

  void IntegrationLoadBalancer::balance() {}

} // namespace DiFfRG