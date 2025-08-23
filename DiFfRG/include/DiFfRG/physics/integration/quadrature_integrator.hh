#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/common/tbb.hh>
#include <DiFfRG/common/tuples.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/integration/abstract_integrator.hh>

namespace DiFfRG
{
  /**
   * @brief This class performs numerical integration over a d-dimensional hypercube using quadrature rules.
   *
   * @tparam dim The dimension of the hypercube, which can be between 1 and 5.
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, which must provide the static methods `kernel` and `constant`
   * @tparam ExecutionSpace can be any execution space, e.g. GPU_exec, TBB_exec, or OpenMP_exec.
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim > 0)
  class QuadratureIntegrator : public AbstractIntegrator
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    /**
     * @brief Execution space to be used for the integration, e.g. GPU_exec, TBB_exec, or OpenMP_exec.
     */
    using execution_space = ExecutionSpace;

    QuadratureIntegrator(QuadratureProvider &quadrature_provider, const std::array<size_t, dim> &_grid_size,
                         const std::array<ctype, dim> &grid_min, const std::array<ctype, dim> &grid_max,
                         const std::array<QuadratureType, dim> &quadrature_type)
        : quadrature_provider(quadrature_provider)
    {
      for (size_t i = 0; i < dim; ++i) {
        grid_size[i] = _grid_size[i];

        nodes[i] = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[i],
                                                                                                    quadrature_type[i]);
        weights[i] = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
            grid_size[i], quadrature_type[i]);
      }
      set_grid_extents(grid_min, grid_max);
    }

    void set_grid_extents(const std::array<ctype, dim> &grid_min, const std::array<ctype, dim> &grid_max)
    {
      for (size_t i = 0; i < dim; ++i) {
        grid_extents[0][i] = grid_min[i];
        grid_extents[1][i] = grid_max[i];

        grid_start[i] = grid_extents[0][i];
        grid_scale[i] = (grid_extents[1][i] - grid_extents[0][i]);
      }
    }

    template <typename... T>
      requires is_valid_kernel<NT, KERNEL, ctype, dim, T...>
    void get(NT &dest, const T &...t) const
    {
      // create an execution space
      ExecutionSpace space;

      Kokkos::View<NT, typename ExecutionSpace::memory_space> result(Kokkos::view_alloc(space, "result"));

      get(space, result, t...);
      space.fence();

      auto result_host = Kokkos::create_mirror_view(result);
      Kokkos::deep_copy(space, result_host, result);
      dest = result_host();
    }

    template <typename OT, typename... T>
      requires(!std::is_same_v<OT, NT> && is_valid_kernel<NT, KERNEL, ctype, dim, T...>)
    void get(OT &dest, const T &...t) const
    {
      ExecutionSpace space;
      get(space, dest, t...);
    }

    template <typename OT, typename... T>
      requires(!std::is_same_v<OT, NT> && is_valid_kernel<NT, KERNEL, ctype, dim, T...>)
    void get(ExecutionSpace &space, OT &dest, const T &...t) const
    {
      const auto args = device::make_tuple(t...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      auto functor = KOKKOS_LAMBDA(const device::array<size_t, dim> &idx, NT &update)
      {
        device::array<ctype, dim> x;
        ctype weight = 1;
        bool is_first = true;
        for (size_t i = 0; i < dim; ++i) {
          x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
          weight *= w[i][idx[i]] * scale[i];
          is_first &= idx[i] == 0;
        }
        device::apply([&](const auto &...iargs) { update += weight * KERNEL::kernel(iargs...); },
                      device::tuple_cat(x, args));
        device::apply([&](const auto &...iargs) { update += is_first ? KERNEL::constant(iargs...) : NT(0); }, args);
      };

      Kokkos::parallel_reduce("QuadratureIntegral_" + std::to_string(dim) + "D", // name of the kernel
                              makeKokkosNDRange<dim, ExecutionSpace>(space, {0}, grid_size),
                              KokkosNDLambdaWrapperReduction<dim, decltype(functor)>(functor), dest);
    }

    template <typename view_type, typename Coordinates, typename... Args>
    void map(ExecutionSpace &space, const view_type integral_view, const Coordinates &coordinates, const Args &...args)
    {
      for (size_t k = 0; k < integral_view.size(); ++k) {
        // make subview
        auto subview = Kokkos::subview(integral_view, k);
        // get the position for the current index
        const auto idx = coordinates.from_linear_index(k);
        const auto pos = coordinates.forward(idx);
        // make a tuple of all arguments
        const auto l_args = device::tuple_cat(pos, device::tie(args...));
        // enqueue a get operation for the current position
        device::apply([&](const auto &...iargs) { get(space, subview, iargs...); }, l_args);
      }
    }

    template <typename Coordinates, typename... Args>
    auto map(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      // Take care of MPI distribution
      const auto &node_distribution = AbstractIntegrator::node_distribution;
      if (node_distribution.mpi_comm != MPI_COMM_NULL && node_distribution.total_size > 0) {
        auto mpi_comm = node_distribution.mpi_comm;
        const auto &nodes = node_distribution.nodes;
        const auto &sizes = node_distribution.sizes;

        // Check if the rank is contained in nodes
        const size_t m_rank = DiFfRG::MPI::rank(mpi_comm);
        // If not, return an empty execution space
        if (std::find(nodes.begin(), nodes.end(), m_rank) == nodes.end()) return ExecutionSpace();

        // Get the size of the current rank
        const size_t rank_size = sizes[m_rank];
        // Offset is the sum of all previous ranks
        const size_t offset = std::accumulate(sizes.begin(), sizes.begin() + m_rank, 0);

        // Create a SubCoordinates object
        const auto sub_coordinates = SubCoordinates(coordinates, offset, rank_size);
        // Offset the destination pointer
        NT *dest_offset = dest + offset;

        return map_dist(dest_offset, sub_coordinates, args...);
      }

      return map_dist(dest, coordinates, args...);
    }

    template <typename Coordinates, typename... Args>
    auto map_dist(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      // create an execution space
      ExecutionSpace space;

      // create unmanaged host view for dest
      auto dest_view = Kokkos::View<NT *, CPU_memory, Kokkos::MemoryUnmanaged>(dest, coordinates.size());

      // create device view for dest
      auto dest_device_view = Kokkos::View<NT *, ExecutionSpace>(
          Kokkos::view_alloc(space, "MapIntegrators_device_view"), coordinates.size());

      // run the map function
      map(space, dest_device_view, coordinates, args...);

      // copy the result from device to host
      auto dest_host_view = Kokkos::create_mirror_view(space, dest_device_view);
      Kokkos::deep_copy(space, dest_host_view, dest_device_view);

      // copy the result from the mirror to the requested destination
      Kokkos::deep_copy(space, dest_view, dest_host_view);

      return space;
    }

  protected:
    device::array<size_t, dim> grid_size;

    QuadratureProvider &quadrature_provider;
    device::array<device::array<ctype, dim>, 2> grid_extents;
    device::array<ctype, dim> grid_start;
    device::array<ctype, dim> grid_scale;

    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, dim> nodes;
    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, dim> weights;
  };

  template <int dim, typename NT, typename KERNEL>
  class QuadratureIntegrator<dim, NT, KERNEL, TBB_exec> : public QuadratureIntegrator<dim, NT, KERNEL, Threads_exec>
  {
    using Base = QuadratureIntegrator<dim, NT, KERNEL, Threads_exec>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or
     * possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = TBB_exec;

    QuadratureIntegrator(QuadratureProvider &quadrature_provider, const std::array<size_t, dim> _grid_size,
                         std::array<ctype, dim> grid_min, std::array<ctype, dim> grid_max,
                         const std::array<QuadratureType, dim> quadrature_type)
        : Base(quadrature_provider, _grid_size, grid_min, grid_max, quadrature_type)
    {
    }

    template <typename... T>
      requires is_valid_kernel<NT, KERNEL, ctype, dim, T...>
    void get(NT &dest, const T &...t) const
    {
      const auto args = device::tie(t...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      auto functor = [&](const device::array<size_t, dim> &idx) {
        device::array<ctype, dim> x;
        ctype weight = 1;
        bool is_first = true;
        for (size_t i = 0; i < dim; ++i) {
          x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
          weight *= w[i][idx[i]] * scale[i];
          is_first &= idx[i] == 0;
        }
        return device::apply([&](const auto &...iargs) { return weight * KERNEL::kernel(iargs...); },
                             device::tuple_cat(x, args));
      };

      dest = KERNEL::constant(t...) + TBBReduction<dim, NT, decltype(functor)>(grid_size, functor);
    }

    template <typename Coordinates, typename... Args>
    void map(execution_space &, NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      const auto m_args = device::tie(args...);

      tbb::parallel_for(tbb::blocked_range<uint>(0, coordinates.size()), [&](const tbb::blocked_range<uint> &r) {
        for (uint idx = r.begin(); idx != r.end(); ++idx) {
          const auto dis_idx = coordinates.from_linear_index(idx);
          const auto pos = coordinates.forward(dis_idx);
          // make a tuple of all arguments
          const auto full_args = device::tuple_cat(pos, m_args);
          device::apply([&](const auto &...iargs) { get(dest[idx], iargs...); }, full_args);
        }
      });
    }

    template <typename Coordinates, typename... Args>
    auto map(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      auto space = execution_space();

      // Take care of MPI distribution
      const auto &node_distribution = AbstractIntegrator::node_distribution;
      if (node_distribution.mpi_comm != MPI_COMM_NULL && node_distribution.total_size > 0) {
        auto mpi_comm = node_distribution.mpi_comm;
        const auto &nodes = node_distribution.nodes;
        const auto &sizes = node_distribution.sizes;

        // Check if the rank is contained in nodes
        const size_t m_rank = DiFfRG::MPI::rank(mpi_comm);
        // If not, return an empty execution space
        if (std::find(nodes.begin(), nodes.end(), m_rank) == nodes.end()) return execution_space();

        // Get the size of the current rank
        const size_t rank_size = sizes[m_rank];
        // Offset is the sum of all previous ranks
        const size_t offset = std::accumulate(sizes.begin(), sizes.begin() + m_rank, 0);

        // Create a SubCoordinates object
        const auto sub_coordinates = SubCoordinates(coordinates, offset, rank_size);
        // Offset the destination pointer
        NT *dest_offset = dest + offset;

        map(space, dest_offset, sub_coordinates, args...);
      } else
        map(space, dest, coordinates, args...);
      return space;
    }

  protected:
    using Base::grid_extents;
    using Base::grid_scale;
    using Base::grid_size;
    using Base::grid_start;
    using Base::quadrature_provider;

    using Base::nodes;
    using Base::weights;
  };
} // namespace DiFfRG
