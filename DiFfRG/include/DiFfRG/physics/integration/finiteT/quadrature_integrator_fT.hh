#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/common/tbb.hh>
#include <DiFfRG/common/tuples.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/abstract_integrator.hh>

namespace DiFfRG
{
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim > 0)
  class QuadratureIntegrator_fT : public AbstractIntegrator
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

    /**
     * @brief Spatial dimension of the integration problem.
     *
     */
    static constexpr int sdim = dim - 1;

    QuadratureIntegrator_fT(QuadratureProvider &quadrature_provider, const std::array<size_t, sdim> _grid_size,
                            std::array<ctype, sdim> grid_min, std::array<ctype, sdim> grid_max,
                            const std::array<QuadratureType, sdim> quadrature_type, const ctype T = 1,
                            const ctype typical_E = 1)
        : quadrature_provider(quadrature_provider), T(T), typical_E(typical_E)
    {
      for (int d = 0; d < sdim; ++d)
        grid_size[d] = _grid_size[d];
      matsubara_nodes =
          quadrature_provider.template matsubara_nodes<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      matsubara_weights =
          quadrature_provider.template matsubara_weights<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      for (int i = 0; i < sdim; ++i) {
        nodes[i] = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[i],
                                                                                                    quadrature_type[i]);
        weights[i] = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
            grid_size[i], quadrature_type[i]);
      }
      set_grid_extents(grid_min, grid_max);
      grid_size[dim - 1] = matsubara_nodes.size();
    }

    void set_grid_extents(const std::array<ctype, sdim> grid_min, const std::array<ctype, sdim> grid_max)
    {
      for (int d = 0; d < sdim; ++d) {
        grid_extents[0][d] = grid_min[d];
        grid_extents[1][d] = grid_max[d];
      }
      for (int i = 0; i < sdim; ++i) {
        grid_start[i] = grid_extents[0][i];
        grid_scale[i] = (grid_extents[1][i] - grid_extents[0][i]);
      }
    }

    void set_T(const ctype T)
    {
      this->T = T;
      matsubara_nodes =
          quadrature_provider.template matsubara_nodes<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      matsubara_weights =
          quadrature_provider.template matsubara_weights<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      grid_size[dim - 1] = matsubara_nodes.size();
    }

    void set_typical_E(const ctype typical_E)
    {
      if (is_close(this->typical_E, typical_E, 1e-4 * T + std::numeric_limits<ctype>::epsilon() * 10)) return;

      this->typical_E = typical_E;
      matsubara_nodes =
          quadrature_provider.template matsubara_nodes<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      matsubara_weights =
          quadrature_provider.template matsubara_weights<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      grid_size[dim - 1] = matsubara_nodes.size();
    }

    template <typename... T> void get(NT &dest, const T &...t) const
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
      requires(!std::is_same_v<OT, NT>)
    void get(OT &dest, const T &...t) const
    {
      ExecutionSpace space;
      get(space, dest, t...);
    }

    template <typename OT, typename... Args>
      requires(!std::is_same_v<OT, NT>)
    void get(ExecutionSpace &space, OT &dest, const Args &...t) const
    {
      const auto args = device::make_tuple(t...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      const auto &m_T = T;

      auto functor = KOKKOS_LAMBDA(const device::array<size_t, dim> &idx, NT &update)
      {
        device::array<ctype, sdim> x;
        ctype weight = 1;
        bool is_first = true;
        for (int i = 0; i < sdim; ++i) {
          x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
          weight *= w[i][idx[i]] * scale[i];
          is_first &= idx[i] == 0;
        }
        is_first &= idx[dim - 1] == 0;
        const ctype xt = m_n[idx[dim - 1]];
        const ctype wt = m_w[idx[dim - 1]];
        device::apply(
            [&](const auto &...iargs) {
              device::apply(
                  [&](const auto &...posargs) {
                    update +=
                        weight *
                        (
                            // positive and negative Matsubara frequencies
                            wt * (KERNEL::kernel(posargs..., xt, iargs...) + KERNEL::kernel(posargs..., -xt, iargs...))
                            // The zero mode (once per matsubara sum)
                            + (idx[dim - 1] != 0 ? (ctype)0 : m_T * KERNEL::kernel(posargs..., (ctype)0, iargs...)));
                  },
                  x);
            },
            args);
        device::apply([&](const auto &...iargs) { update += is_first ? KERNEL::constant(iargs...) : NT(0); }, args);
      };

      Kokkos::parallel_reduce("QuadratureIntegral_fT_" + std::to_string(dim) + "D", // name of the kernel
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
    QuadratureProvider &quadrature_provider;
    device::array<device::array<ctype, sdim>, 2> grid_extents;
    device::array<ctype, sdim> grid_start;
    device::array<ctype, sdim> grid_scale;

    device::array<size_t, dim> grid_size;

    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, sdim> nodes;
    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, sdim> weights;

    ctype T, typical_E;

    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> matsubara_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> matsubara_weights;
  };

  template <int dim, typename NT, typename KERNEL>
  class QuadratureIntegrator_fT<dim, NT, KERNEL, TBB_exec>
      : public QuadratureIntegrator_fT<dim, NT, KERNEL, Threads_exec>
  {
    using Base = QuadratureIntegrator_fT<dim, NT, KERNEL, Threads_exec>;

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or
     * possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = TBB_exec;

    static constexpr int sdim = dim - 1; // spatial dimension

    QuadratureIntegrator_fT(QuadratureProvider &quadrature_provider, const std::array<size_t, sdim> _grid_size,
                            std::array<ctype, sdim> grid_min, std::array<ctype, sdim> grid_max,
                            const std::array<QuadratureType, sdim> quadrature_type, const ctype T = 1,
                            const ctype typical_E = 1)
        : Base(quadrature_provider, _grid_size, grid_min, grid_max, quadrature_type, T, typical_E)
    {
    }

    template <typename... Args>
      requires is_valid_kernel<NT, KERNEL, ctype, dim, Args...>
    void get(NT &dest, const Args &...t) const
    {
      const auto args = device::tie(t...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      const auto &m_T = T;

      auto functor = [&](const device::array<size_t, dim> &idx) {
        device::array<ctype, sdim> x;
        ctype weight = 1;
        for (int i = 0; i < sdim; ++i) {
          x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
          weight *= w[i][idx[i]] * scale[i];
        }
        const ctype xt = m_n[idx[dim - 1]];
        const ctype wt = m_w[idx[dim - 1]];
        NT update{};
        device::apply(
            [&](const auto &...iargs) {
              device::apply(
                  [&](const auto &...posargs) {
                    update +=
                        weight *
                        (
                            // positive and negative Matsubara frequencies
                            wt * (KERNEL::kernel(posargs..., xt, iargs...) + KERNEL::kernel(posargs..., -xt, iargs...))
                            // The zero mode (once per matsubara sum)
                            + (idx[dim - 1] != 0 ? (ctype)0 : m_T * KERNEL::kernel(posargs..., (ctype)0, iargs...)));
                  },
                  x);
            },
            args);
        return update;
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

    using Base::matsubara_nodes;
    using Base::matsubara_weights;
    using Base::nodes;
    using Base::weights;

    using Base::T;
    using Base::typical_E;
  };

} // namespace DiFfRG
