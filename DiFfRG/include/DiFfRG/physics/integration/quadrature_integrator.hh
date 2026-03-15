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
  /**
   * @brief This class performs numerical integration over a d-dimensional hypercube using quadrature rules.
   *
   * @tparam dim The dimension of the hypercube, which can be between 1 and 5.
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, which must provide the static methods `kernel` and `constant`
   * @tparam ExecutionSpace can be any execution space, e.g. GPU_exec, TBB_exec.
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
     * @brief Execution space to be used for the integration, e.g. GPU_exec, TBB_exec.
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

      if (!m_result_views_initialized) {
        m_result_view = Kokkos::View<NT, typename ExecutionSpace::memory_space>("result");
        m_result_host = Kokkos::create_mirror_view(m_result_view);
        m_result_views_initialized = true;
      }
      get(space, m_result_view, t...);
      Kokkos::deep_copy(space, m_result_host, m_result_view);
      space.fence();
      dest = m_result_host();
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
      device::array<size_t, 1 + dim> extents;
      extents[0] = integral_view.size();
      for (int i = 0; i < dim; ++i)
        extents[1 + i] = grid_size[i];

      // Reuse cached view if large enough, otherwise reallocate (grow-only)
      {
        bool needs_realloc = false;
        for (size_t i = 0; i < 1 + dim; ++i)
          needs_realloc |= (extents[i] > m_cache_extents[i]);
        if (needs_realloc) {
          for (size_t i = 0; i < 1 + dim; ++i)
            m_cache_extents[i] = std::max(m_cache_extents[i], extents[i]);
          m_cache = makeKokkosNDView<1 + dim, NT, ExecutionSpace>("cache", m_cache_extents);
        }
      }
      // Create a Restrict-tagged alias of the cache for no-alias optimization
      const auto cache = KokkosNDViewRestrict<1 + dim, NT, ExecutionSpace>(m_cache);

      const auto m_args = device::make_tuple(args...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
      {
        // make subview
        auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

        // get the position for the current index
        const auto idx_v = coordinates.from_linear_index(idx[0]);
        const auto pos = coordinates.forward(idx_v);

        device::array<ctype, dim> x;
        ctype weight = 1;
        for (int i = 0; i < dim; ++i) {
          x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
          weight *= w[i][idx[1 + i]] * scale[i];
        }

        // make a tuple of all arguments
        const auto full_args = device::tuple_cat(x, pos, m_args);

        device::apply([&](const auto &...iargs) { subview() = weight * KERNEL::kernel(iargs...); }, full_args);
      };

      Kokkos::parallel_for(makeKokkosNDRange<1 + dim, ExecutionSpace>(space, {0}, extents),
                           KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));

      using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
      // reduction with vector lanes for warp-level parallelism
      constexpr int vector_width = 32;
      Kokkos::parallel_for(
          Kokkos::TeamPolicy(space, integral_view.size(), Kokkos::AUTO, vector_width),
          KOKKOS_CLASS_LAMBDA(const TeamType &team) {
            // get the current (continuous) index
            const uint k = team.league_rank();

            if (k > integral_view.size()) return;

            // no-ops to capture
            (void)cache;
            (void)grid_size;

            // Flatten grid_size into total element count for thread+vector splitting
            size_t total_elements = 1;
            for (int d = 0; d < dim; ++d)
              total_elements *= grid_size[d];

            // Pre-compute stride array for index decomposition (avoids division/modulo in inner loop)
            device::array<size_t, dim> strides;
            strides[dim - 1] = 1;
            for (int d = dim - 2; d >= 0; --d)
              strides[d] = strides[d + 1] * grid_size[d + 1];

            NT res{};
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, (total_elements + vector_width - 1) / vector_width),
                [&](const size_t outer, NT &team_update) {
                  NT vec_sum{};
                  Kokkos::parallel_reduce(
                      Kokkos::ThreadVectorRange(team, vector_width),
                      [&](const size_t inner, NT &vec_update) {
                        const size_t flat = outer * vector_width + inner;
                        if (flat < total_elements) {
                          // Convert flat index back to multi-dimensional using pre-computed strides
                          device::array<size_t, dim> ridx;
                          size_t remainder = flat;
                          for (int d = 0; d < dim; ++d) {
                            ridx[d] = remainder / strides[d];
                            remainder -= ridx[d] * strides[d];
                          }
                          device::apply([&](const auto &...iargs) { vec_update += cache(k, iargs...); }, ridx);
                        }
                      },
                      vec_sum);
                  team_update += vec_sum;
                },
                res);

            // add the constant value (skip coordinate computation if kernel has no constant)
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              const auto idx = coordinates.from_linear_index(k);
              const auto pos = coordinates.forward(idx);
              const auto full_args = device::tuple_cat(pos, m_args);
              integral_view(k) =
                  res + device::apply([&](const auto &...iargs) { return KERNEL::constant(iargs...); }, full_args);
            });
          });
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
      // create unmanaged host view for dest
      auto dest_view = Kokkos::View<NT *, CPU_memory, Kokkos::MemoryUnmanaged>(dest, coordinates.size());

      // Reuse cached device view if large enough, otherwise reallocate (grow-only)
      if (m_dest_device_size < coordinates.size()) {
        m_dest_device = Kokkos::View<NT *, ExecutionSpace>(Kokkos::view_alloc(space, "MapIntegrators_device_view"),
                                                           coordinates.size());
        m_dest_device_size = coordinates.size();
      }
      auto dest_device_view =
          Kokkos::View<NT *, ExecutionSpace>(m_dest_device, Kokkos::make_pair(size_t(0), coordinates.size()));

      // run the map function
      map(space, dest_device_view, coordinates, args...);

      // copy the result from device to the unmanaged host view
      Kokkos::deep_copy(space, dest_view, dest_device_view);

      return space;
    }

  protected:
    device::array<size_t, dim> grid_size;

    ExecutionSpace space;
    QuadratureProvider &quadrature_provider;
    device::array<device::array<ctype, dim>, 2> grid_extents;
    device::array<ctype, dim> grid_start;
    device::array<ctype, dim> grid_scale;

    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, dim> nodes;
    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, dim> weights;

    // Persistent view caches to avoid per-call GPU memory allocation
    mutable KokkosNDView<1 + dim, NT, ExecutionSpace> m_cache;
    mutable device::array<size_t, 1 + dim> m_cache_extents{};
    mutable Kokkos::View<NT *, ExecutionSpace> m_dest_device;
    mutable size_t m_dest_device_size = 0;
    mutable Kokkos::View<NT, typename ExecutionSpace::memory_space> m_result_view;
    mutable typename Kokkos::View<NT, typename ExecutionSpace::memory_space>::host_mirror_type m_result_host;
    mutable bool m_result_views_initialized = false;
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
