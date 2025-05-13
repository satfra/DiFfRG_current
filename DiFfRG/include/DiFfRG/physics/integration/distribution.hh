#pragma once

#include <DiFfRG/common/mpi.hh>

namespace DiFfRG
{
  struct NodeDistribution {
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    uint total_size = 0;
    std::vector<uint> sizes;
    std::vector<uint> nodes;

    NodeDistribution() = default;
    NodeDistribution(MPI_Comm &mpi_comm, const std::vector<uint> &sizes, const std::vector<uint> &nodes);
  };

  class IntegrationLoadBalancer
  {
  public:
    IntegrationLoadBalancer(MPI_Comm &mpi_comm);

    void balance();

    template <typename INT> void register_integrator(INT &integrator) { integrators.push_back(&integrator); }

  private:
    MPI_Comm mpi_comm;

    std::vector<void *> integrators;
    std::vector<NodeDistribution> node_distributions;
  };
} // namespace DiFfRG