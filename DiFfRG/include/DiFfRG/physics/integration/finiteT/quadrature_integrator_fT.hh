#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/common/tuples.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/integration/abstract_integrator.hh>

// standard libraries
#include <array>

// external libraries
#include <tbb/tbb.h>

namespace DiFfRG
{
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
  class QuadratureIntegrator_fT : public AbstractIntegrator
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    QuadratureIntegrator_fT(QuadratureProvider &quadrature_provider, const std::array<uint, dim> grid_size,
                            std::array<ctype, dim> grid_min, std::array<ctype, dim> grid_max,
                            const std::array<QuadratureType, dim> quadrature_type, const ctype T = 1,
                            const ctype typical_E = 1)
        : grid_size(grid_size), quadrature_provider(quadrature_provider), T(T), typical_E(typical_E)
    {
      matsubara_nodes =
          quadrature_provider.template matsubara_nodes<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      matsubara_weights =
          quadrature_provider.template matsubara_weights<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      for (int i = 0; i < dim; ++i) {
        nodes[i] = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[i],
                                                                                                    quadrature_type[i]);
        weights[i] = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
            grid_size[i], quadrature_type[i]);
      }
      set_grid_extents(grid_min, grid_max);
    }

    void set_grid_extents(const std::array<ctype, dim> grid_min, const std::array<ctype, dim> grid_max)
    {
      grid_extents = {grid_min, grid_max};
      for (int i = 0; i < dim; ++i) {
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
    }

    void set_typical_E(const ctype typical_E)
    {
      if (is_close(this->typical_E, typical_E, 1e-4 * T + std::numeric_limits<ctype>::epsilon() * 10)) return;

      this->typical_E = typical_E;
      matsubara_nodes =
          quadrature_provider.template matsubara_nodes<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
      matsubara_weights =
          quadrature_provider.template matsubara_weights<ctype, typename ExecutionSpace::memory_space>(T, typical_E);
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
      const auto args = std::make_tuple(t...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      const auto &m_T = T;

      if constexpr (dim == 0) {
        std::cout << "QuadratureIntegral_0D_fT" << std::endl;
        Kokkos::parallel_reduce(
            "QuadratureIntegral_0D_fT", // name of the kernel
            Kokkos::RangePolicy<ExecutionSpace>(space, 0, m_n.size()),
            KOKKOS_LAMBDA(const uint idxt, NT &update) {
              const ctype xt = m_n[idxt];
              const NT result = std::apply(
                  [&](const auto &...iargs) {
                    return
                        // positive and negative Matsubara frequencies
                        m_w[idxt] * (KERNEL::kernel(xt, iargs...) + KERNEL::kernel(-xt, iargs...))
                        // The zero mode (once per spatial integral)
                        + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel((ctype)0, iargs...));
                  },
                  args);
              update += result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim == 1) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_1D_fT", // name of the kernel
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<dim + 1>>(space, {0, 0}, {n[0].size(), m_n.size()}),
            KOKKOS_LAMBDA(const uint idx0, const uint idxt, NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype xt = m_n[idxt];
              const ctype weight = w[0][idx0] * scale[0];
              const NT result = std::apply(
                  [&](const auto &...iargs) {
                    return
                        // positive and negative Matsubara frequencies
                        m_w[idxt] * (KERNEL::kernel(x0, xt, iargs...) + KERNEL::kernel(x0, -xt, iargs...))
                        // The zero mode (once per spatial integral)
                        + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, (ctype)0, iargs...));
                  },
                  args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim == 2) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_2D_fT", // name of the kernel
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<dim + 1>>(space, {0, 0, 0},
                                                                         {n[0].size(), n[1].size(), m_n.size()}),
            KOKKOS_LAMBDA(const uint idx0, const uint idx1, const uint idxt, NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
              const ctype xt = m_n[idxt];
              const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1];
              const NT result = std::apply(
                  [&](const auto &...iargs) {
                    return
                        // positive and negative Matsubara frequencies
                        m_w[idxt] * (KERNEL::kernel(x0, x1, xt, iargs...) + KERNEL::kernel(x0, x1, -xt, iargs...))
                        // The zero mode (once per spatial integral)
                        + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, (ctype)0, iargs...));
                  },
                  args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim == 3) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_3D_fT", // name of the kernel
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<dim + 1>>(
                space, {0, 0, 0, 0}, {n[0].size(), n[1].size(), n[2].size(), m_n.size()}),
            KOKKOS_LAMBDA(const uint idx0, const uint idx1, const uint idx2, const uint idxt, NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
              const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
              const ctype xt = m_n[idxt];
              const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2];
              const NT result = std::apply(
                  [&](const auto &...iargs) {
                    return
                        // positive and negative Matsubara frequencies
                        m_w[idxt] *
                            (KERNEL::kernel(x0, x1, x2, xt, iargs...) + KERNEL::kernel(x0, x1, x2, -xt, iargs...))
                        // The zero mode (once per spatial integral)
                        + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, x2, (ctype)0, iargs...));
                  },
                  args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim == 4) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_4D_fT", // name of the kernel
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<dim + 1>>(
                space, {0, 0, 0, 0, 0}, {n[0].size(), n[1].size(), n[2].size(), n[3].size(), m_n.size()}),
            KOKKOS_LAMBDA(const uint idx0, const uint idx1, const uint idx2, const uint idx3, const uint idxt,
                          NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
              const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
              const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
              const ctype xt = m_n[idxt];
              const ctype weight =
                  w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2] * w[3][idx3] * scale[3];
              const NT result = std::apply(
                  [&](const auto &...iargs) {
                    return
                        // positive and negative Matsubara frequencies
                        m_w[idxt] * (KERNEL::kernel(x0, x1, x2, x3, xt, iargs...) +
                                     KERNEL::kernel(x0, x1, x2, x3, -xt, iargs...))
                        // The zero mode (once per spatial integral)
                        + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, x2, x3, (ctype)0, iargs...));
                  },
                  args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim > 4) {
        // make a compiler error
        static_assert(dim <= 4, "QuadratureIntegrator_fT only supports integrals from 1 to 4 (spatial) dimensions");
      }
    }

    template <typename view_type, typename Coordinates, typename... Args>
    void map(ExecutionSpace &space, const view_type integral_view, const Coordinates &coordinates, const Args &...args)
    {
      const auto m_args = std::make_tuple(args...);

      using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
      using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
      using Scratch = typename ExecutionSpace::scratch_memory_space;

      const auto &n = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      const auto &m_T = T;

      Kokkos::parallel_for(
          Kokkos::TeamPolicy(space, integral_view.size(), Kokkos::AUTO), KOKKOS_LAMBDA(const TeamType &team) {
            // get the current (continuous) index
            const uint k = team.league_rank();
            // make subview
            auto subview = Kokkos::subview(integral_view, k);
            // get the position for the current index
            const auto idx = coordinates.from_continuous_index(k);
            const auto pos = coordinates.forward(idx);
            // make a tuple of all arguments
            const auto full_args = std::tuple_cat(pos, m_args);

            // no-ops to capture
            (void)start;
            (void)scale;
            (void)m_n;
            (void)m_w;
            (void)n;
            (void)w;
            (void)m_T;

            NT res = 0;
            if constexpr (dim == 0) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(team, m_n.size()), // range of the kernel
                  [&](const uint idxt, NT &update) {
                    const ctype xt = m_n[idxt];
                    const NT result = std::apply(
                        [&](const auto &...iargs) {
                          return
                              // positive and negative Matsubara frequencies
                              m_w[idxt] * (KERNEL::kernel(xt, iargs...) + KERNEL::kernel(-xt, iargs...))
                              // The zero mode (once per spatial integral)
                              + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel((ctype)0, iargs...));
                        },
                        full_args);
                    update += result;
                  },
                  res);
            } else if constexpr (dim == 1) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadMDRange(team, n[0].size(), m_n.size()), // range of the kernel
                  [&](const uint idx0, const uint idxt, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype xt = m_n[idxt];
                    const ctype weight = w[0][idx0] * scale[0];
                    const NT result = std::apply(
                        [&](const auto &...iargs) {
                          return
                              // positive and negative Matsubara frequencies
                              m_w[idxt] * (KERNEL::kernel(x0, xt, iargs...) + KERNEL::kernel(x0, -xt, iargs...))
                              // The zero mode (once per spatial integral)
                              + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, (ctype)0, iargs...));
                        },
                        full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim == 2) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadMDRange(team, n[0].size(), n[1].size(), m_n.size()), // range of the kernel
                  [&](const uint idx0, const uint idx1, const uint idxt, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                    const ctype xt = m_n[idxt];
                    const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1];
                    const NT result = std::apply(
                        [&](const auto &...iargs) {
                          return
                              // positive and negative Matsubara frequencies
                              m_w[idxt] * (KERNEL::kernel(x0, x1, xt, iargs...) + KERNEL::kernel(x0, x1, -xt, iargs...))
                              // The zero mode (once per spatial integral)
                              + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, (ctype)0, iargs...));
                        },
                        full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim == 3) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadMDRange(team, n[0].size(), n[1].size(), n[2].size(),
                                            m_n.size()), // range of the kernel
                  [&](const uint idx0, const uint idx1, const uint idx2, const uint idxt, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                    const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                    const ctype xt = m_n[idxt];
                    const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2];
                    const NT result = std::apply(
                        [&](const auto &...iargs) {
                          return
                              // positive and negative Matsubara frequencies
                              m_w[idxt] *
                                  (KERNEL::kernel(x0, x1, x2, xt, iargs...) + KERNEL::kernel(x0, x1, x2, -xt, iargs...))
                              // The zero mode (once per spatial integral)
                              + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, x2, (ctype)0, iargs...));
                        },
                        full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim == 4) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadMDRange(team, n[0].size(), n[1].size(), n[2].size(), n[3].size(),
                                            m_n.size()), // range of the kernel
                  [&](const uint idx0, const uint idx1, const uint idx2, const uint idx3, const uint idxt, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                    const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                    const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
                    const ctype xt = m_n[idxt];
                    const ctype weight =
                        w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2] * w[3][idx3] * scale[3];
                    const NT result = std::apply(
                        [&](const auto &...iargs) {
                          return
                              // positive and negative Matsubara frequencies
                              m_w[idxt] * (KERNEL::kernel(x0, x1, x2, x3, xt, iargs...) +
                                           KERNEL::kernel(x0, x1, x2, x3, -xt, iargs...))
                              // The zero mode (once per spatial integral)
                              + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, x2, x3, (ctype)0, iargs...));
                        },
                        full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim > 4) {
              // make a compiler error
              static_assert(dim <= 4,
                            "QuadratureIntegrator_fT only supports integrals from 1 to 4 (spatial) dimensions");
            }

            // add the constant value
            team.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
              subview() = res + std::apply([&](const auto &...iargs) { return KERNEL::constant(iargs...); }, full_args);
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
        const uint m_rank = DiFfRG::MPI::rank(mpi_comm);
        // If not, return an empty execution space
        if (std::find(nodes.begin(), nodes.end(), m_rank) == nodes.end()) return ExecutionSpace();

        // Get the size of the current rank
        const uint rank_size = sizes[m_rank];
        // Offset is the sum of all previous ranks
        const uint offset = std::accumulate(sizes.begin(), sizes.begin() + m_rank, 0);

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

    const std::array<uint, dim> grid_size;

  private:
    QuadratureProvider &quadrature_provider;
    std::array<std::array<ctype, dim>, 2> grid_extents;
    std::array<ctype, dim> grid_start;
    std::array<ctype, dim> grid_scale;

    std::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, dim> nodes;
    std::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, dim> weights;

    ctype T, typical_E;

    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> matsubara_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> matsubara_weights;
  };

  template <int dim, typename NT, typename KERNEL>
  class QuadratureIntegrator_fT<dim, NT, KERNEL, TBB_exec> : public AbstractIntegrator
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = TBB_exec;

    QuadratureIntegrator_fT(QuadratureProvider &quadrature_provider, const std::array<uint, dim> grid_size,
                            std::array<ctype, dim> grid_min, std::array<ctype, dim> grid_max,
                            const std::array<QuadratureType, dim> quadrature_type, const ctype T = 1,
                            const ctype typical_E = 1)
        : grid_size(grid_size), quadrature_provider(quadrature_provider), T(T), typical_E(typical_E)
    {
      matsubara_nodes =
          quadrature_provider.template matsubara_nodes<ctype, typename execution_space::memory_space>(T, typical_E);
      matsubara_weights =
          quadrature_provider.template matsubara_weights<ctype, typename execution_space::memory_space>(T, typical_E);
      for (int i = 0; i < dim; ++i) {
        nodes[i] = quadrature_provider.template nodes<ctype, typename execution_space::memory_space>(
            grid_size[i], quadrature_type[i]);
        weights[i] = quadrature_provider.template weights<ctype, typename execution_space::memory_space>(
            grid_size[i], quadrature_type[i]);
      }
      set_grid_extents(grid_min, grid_max);
    }

    void set_grid_extents(const std::array<ctype, dim> grid_min, const std::array<ctype, dim> grid_max)
    {
      grid_extents = {grid_min, grid_max};
      for (int i = 0; i < dim; ++i) {
        grid_start[i] = grid_extents[0][i];
        grid_scale[i] = (grid_extents[1][i] - grid_extents[0][i]);
      }
    }

    void set_T(const ctype T)
    {
      this->T = T;
      matsubara_nodes =
          quadrature_provider.template matsubara_nodes<ctype, typename execution_space::memory_space>(T, typical_E);
      matsubara_weights =
          quadrature_provider.template matsubara_weights<ctype, typename execution_space::memory_space>(T, typical_E);
    }

    void set_typical_E(const ctype typical_E)
    {
      if (is_close(this->typical_E, typical_E, 1e-4 * T + std::numeric_limits<ctype>::epsilon() * 10)) return;

      this->typical_E = typical_E;
      matsubara_nodes =
          quadrature_provider.template matsubara_nodes<ctype, typename execution_space::memory_space>(T, typical_E);
      matsubara_weights =
          quadrature_provider.template matsubara_weights<ctype, typename execution_space::memory_space>(T, typical_E);
    }

    template <typename... Args> void get(NT &dest, const Args &...t) const
    {
      const auto &n = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      const auto &m_T = T;

      if constexpr (dim == 0) {
        dest = KERNEL::constant(t...) + tbb::parallel_reduce(
                                            tbb::blocked_range<uint>(0, m_n.size()), NT(0),
                                            [&](const tbb::blocked_range<uint> &r, NT value) -> NT {
                                              for (uint idxt = r.begin(); idxt != r.end(); ++idxt) {
                                                const ctype xt = m_n[idxt];
                                                const NT result =
                                                    // positive and negative Matsubara frequencies
                                                    m_w[idxt] * (KERNEL::kernel(xt, t...) + KERNEL::kernel(-xt, t...))
                                                    // The zero mode (once per spatial integral)
                                                    + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel((ctype)0, t...));
                                                value += result;
                                              }
                                              return value;
                                            },
                                            [&](NT x, NT y) -> NT { return x + y; });
      } else if constexpr (dim == 1) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range2d<uint, uint>(0, n[0].size(), 0, m_n.size()), NT(0),
                   [&](const tbb::blocked_range2d<uint, uint> &r, NT value) -> NT {
                     for (uint idx0 = r.rows().begin(); idx0 != r.rows().end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       const ctype weight = w[0][idx0] * scale[0];
                       for (uint idxt = r.cols().begin(); idxt != r.cols().end(); ++idxt) {
                         const ctype xt = m_n[idxt];
                         const NT result =
                             // positive and negative Matsubara frequencies
                             m_w[idxt] * (KERNEL::kernel(x0, xt, t...) + KERNEL::kernel(x0, -xt, t...))
                             // The zero mode (once per spatial integral)
                             + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, (ctype)0, t...));
                         value += weight * result;
                       }
                     }
                     return value;
                   },
                   [&](NT x, NT y) -> NT { return x + y; });
      } else if constexpr (dim == 2) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range3d<uint, uint, uint>(0, n[0].size(), 0, n[1].size(), 0, m_n.size()), NT(0),
                   [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                     for (uint idx0 = r.pages().begin(); idx0 != r.pages().end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       for (uint idx1 = r.rows().begin(); idx1 != r.rows().end(); ++idx1) {
                         const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                         const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1];
                         for (uint idxt = r.cols().begin(); idxt != r.cols().end(); ++idxt) {
                           const ctype xt = m_n[idxt];
                           const NT result =
                               // positive and negative Matsubara frequencies
                               m_w[idxt] * (KERNEL::kernel(x0, x1, xt, t...) + KERNEL::kernel(x0, x1, -xt, t...))
                               // The zero mode (once per spatial integral)
                               + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, (ctype)0, t...));
                           value += weight * result;
                         }
                       }
                     }
                     return value;
                   },
                   [&](NT x, NT y) -> NT { return x + y; });
      } else if constexpr (dim == 3) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range3d<uint, uint, uint>(0, n[0].size(), 0, n[1].size(), 0, n[2].size()), NT(0),
                   [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                     for (uint idx0 = r.pages().begin(); idx0 != r.pages().end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       for (uint idx1 = r.rows().begin(); idx1 != r.rows().end(); ++idx1) {
                         const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                         for (uint idx2 = r.cols().begin(); idx2 != r.cols().end(); ++idx2) {
                           const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                           value += tbb::parallel_reduce(
                               tbb::blocked_range<uint>(0, m_n.size()), NT(0),
                               [&](const tbb::blocked_range<uint> &r2, NT value2) -> NT {
                                 for (uint idxt = r2.begin(); idxt != r2.end(); ++idxt) {
                                   const ctype xt = m_n[idxt];
                                   const ctype weight =
                                       w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2];
                                   const NT result =
                                       // positive and negative Matsubara frequencies
                                       m_w[idxt] * (KERNEL::kernel(x0, x1, x2, xt, t...) +
                                                    KERNEL::kernel(x0, x1, x2, -xt, t...))
                                       // The zero mode (once per spatial integral)
                                       + (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, x2, (ctype)0, t...));
                                   value2 += weight * result;
                                 }
                                 return value2;
                               },
                               [&](NT x, NT y) -> NT { return x + y; });
                         }
                       }
                     }
                     return value;
                   },
                   [&](NT x, NT y) -> NT { return x + y; });
      } else if constexpr (dim == 4) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range3d<uint, uint, uint>(0, n[0].size(), 0, n[1].size(), 0, n[2].size()), NT(0),
                   [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                     for (uint idx0 = r.pages().begin(); idx0 != r.pages().end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       for (uint idx1 = r.rows().begin(); idx1 != r.rows().end(); ++idx1) {
                         const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                         for (uint idx2 = r.cols().begin(); idx2 != r.cols().end(); ++idx2) {
                           const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                           value += tbb::parallel_reduce(
                               tbb::blocked_range2d<uint, uint>(0, n[3].size(), 0, m_n.size()), NT(0),
                               [&](const tbb::blocked_range2d<uint, uint> &r2, NT value2) -> NT {
                                 for (uint idx3 = r2.rows().begin(); idx3 != r2.rows().end(); ++idx3) {
                                   const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
                                   for (uint idxt = r2.cols().begin(); idxt != r2.cols().end(); ++idxt) {
                                     const ctype xt = m_n[idxt];
                                     const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] *
                                                          scale[2] * w[3][idx3] * scale[3];
                                     const NT result =
                                         // positive and negative Matsubara frequencies
                                         m_w[idxt] * (KERNEL::kernel(x0, x1, x2, x3, xt, t...) +
                                                      KERNEL::kernel(x0, x1, x2, x3, -xt, t...))
                                         // The zero mode (once per spatial integral)
                                         +
                                         (idxt != 0 ? (ctype)0 : m_T * KERNEL::kernel(x0, x1, x2, x3, (ctype)0, t...));
                                     value2 += weight * result;
                                   }
                                 }
                                 return value2;
                               },
                               [&](NT x, NT y) -> NT { return x + y; });
                         }
                       }
                     }
                     return value;
                   },
                   [&](NT x, NT y) -> NT { return x + y; });
      } else if constexpr (dim > 4) {
        // make a compiler error
        static_assert(dim <= 4, "QuadratureIntegrator_fT only supports integrals from 1 to 4 (spatial) dimensions");
      }
    }

    template <typename Coordinates, typename... Args>
    void map(execution_space &space, NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      const auto m_args = std::tie(args...);

      tbb::parallel_for(tbb::blocked_range<uint>(0, coordinates.size()), [&](const tbb::blocked_range<uint> &r) {
        for (uint idx = r.begin(); idx != r.end(); ++idx) {
          const auto dis_idx = coordinates.from_continuous_index(idx);
          const auto pos = coordinates.forward(dis_idx);
          // make a tuple of all arguments
          const auto full_args = std::tuple_cat(pos, m_args);
          std::apply([&](const auto &...iargs) { get(dest[idx], iargs...); }, full_args);
        }
      });
    }

    template <typename Coordinates, typename... Args>
    auto map(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      const auto space = execution_space();

      // Take care of MPI distribution
      const auto &node_distribution = AbstractIntegrator::node_distribution;
      if (node_distribution.mpi_comm != MPI_COMM_NULL && node_distribution.total_size > 0) {
        auto mpi_comm = node_distribution.mpi_comm;
        const auto &nodes = node_distribution.nodes;
        const auto &sizes = node_distribution.sizes;

        // Check if the rank is contained in nodes
        const uint m_rank = DiFfRG::MPI::rank(mpi_comm);
        // If not, return an empty execution space
        if (std::find(nodes.begin(), nodes.end(), m_rank) == nodes.end()) return execution_space();

        // Get the size of the current rank
        const uint rank_size = sizes[m_rank];
        // Offset is the sum of all previous ranks
        const uint offset = std::accumulate(sizes.begin(), sizes.begin() + m_rank, 0);

        // Create a SubCoordinates object
        const auto sub_coordinates = SubCoordinates(coordinates, offset, rank_size);
        // Offset the destination pointer
        NT *dest_offset = dest + offset;

        map_impl(space, dest_offset, sub_coordinates, args...);
      } else
        map_impl(space, dest, coordinates, args...);
      return space;
    }

    const std::array<uint, dim> grid_size;

  private:
    QuadratureProvider &quadrature_provider;
    std::array<std::array<ctype, dim>, 2> grid_extents;
    std::array<ctype, dim> grid_start;
    std::array<ctype, dim> grid_scale;

    std::array<Kokkos::View<const ctype *, typename execution_space::memory_space>, dim> nodes;
    std::array<Kokkos::View<const ctype *, typename execution_space::memory_space>, dim> weights;

    ctype T, typical_E;

    Kokkos::View<const ctype *, typename execution_space::memory_space> matsubara_nodes;
    Kokkos::View<const ctype *, typename execution_space::memory_space> matsubara_weights;
  };

} // namespace DiFfRG
