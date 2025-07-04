#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>
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
  /**
   * @brief This class performs numerical integration over a d-dimensional hypercube using quadrature rules.
   *
   * @tparam dim The dimension of the hypercube, which can be between 1 and 5.
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, which must provide the static methods `kernel` and `constant`
   * @tparam ExecutionSpace can be any execution space, e.g. GPU_exec, TBB_exec, or OpenMP_exec.
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
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

    QuadratureIntegrator(QuadratureProvider &quadrature_provider, const std::array<uint, dim> grid_size,
                         std::array<ctype, dim> grid_min, std::array<ctype, dim> grid_max,
                         const std::array<QuadratureType, dim> quadrature_type)
        : grid_size(grid_size), quadrature_provider(quadrature_provider)
    {
      for (uint i = 0; i < dim; ++i) {
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
      for (uint i = 0; i < dim; ++i) {
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
      const auto args = std::make_tuple(t...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      if constexpr (dim == 1) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_2D", // name of the kernel
            Kokkos::RangePolicy<ExecutionSpace>(space, 0, n[0].size()),
            KOKKOS_LAMBDA(const uint idx0, NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype weight = w[0][idx0] * scale[0];
              const NT result = std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, iargs...); }, args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim == 2) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_2D", // name of the kernel
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<dim>>(space, {0, 0}, {n[0].size(), n[1].size()}),
            KOKKOS_LAMBDA(const uint idx0, const uint idx1, NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
              const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1];
              const NT result =
                  std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, x1, iargs...); }, args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim == 3) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_3D", // name of the kernel
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<dim>>(space, {0, 0, 0},
                                                                     {n[0].size(), n[1].size(), n[2].size()}),
            KOKKOS_LAMBDA(const uint idx0, const uint idx1, const uint idx2, NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
              const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
              const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2];
              const NT result =
                  std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, iargs...); }, args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim == 4) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_4D", // name of the kernel
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<dim>>(
                space, {0, 0, 0, 0}, {n[0].size(), n[1].size(), n[2].size(), n[3].size()}),
            KOKKOS_LAMBDA(const uint idx0, const uint idx1, const uint idx2, const uint idx3, NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
              const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
              const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
              const ctype weight =
                  w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2] * w[3][idx3] * scale[3];
              const NT result =
                  std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, x3, iargs...); }, args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim == 5) {
        Kokkos::parallel_reduce(
            "QuadratureIntegral_5D", // name of the kernel
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<dim>>(
                space, {0, 0, 0, 0, 0}, {n[0].size(), n[1].size(), n[2].size(), n[3].size(), n[4].size()}),
            KOKKOS_LAMBDA(const uint idx0, const uint idx1, const uint idx2, const uint idx3, const uint idx4,
                          NT &update) {
              const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
              const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
              const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
              const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
              const ctype x4 = Kokkos::fma(scale[4], n[4][idx4], start[4]);
              const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2] * w[3][idx3] *
                                   scale[3] * w[4][idx4] * scale[4];
              const NT result =
                  std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, x3, x4, iargs...); }, args);
              update += weight * result;
            },
            SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
      } else if constexpr (dim > 5) {
        // make a compiler error
        static_assert(dim <= 5, "QuadratureIntegrator only supports integrals from 1 to 5 dimensions");
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
      const auto &start = grid_start;
      const auto &scale = grid_scale;

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
            (void)n;
            (void)w;

            NT res = 0;
            if constexpr (dim == 1) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(team, 0, n[0].size()), // range of the kernel
                  [&](const uint idx0, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype weight = w[0][idx0] * scale[0];
                    const NT result =
                        std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, iargs...); }, full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim == 2) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadMDRange(team, n[0].size(), n[1].size()), // range of the kernel
                  [&](const uint idx0, const uint idx1, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                    const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1];
                    const NT result =
                        std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, x1, iargs...); }, full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim == 3) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadMDRange(team, n[0].size(), n[1].size(), n[2].size()), // range of the kernel
                  [&](const uint idx0, const uint idx1, const uint idx2, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                    const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                    const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2];
                    const NT result = std::apply(
                        [&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, iargs...); }, full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim == 4) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadMDRange(team, n[0].size(), n[1].size(), n[2].size(),
                                            n[3].size()), // range of the kernel
                  [&](const uint idx0, const uint idx1, const uint idx2, const uint idx3, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                    const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                    const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
                    const ctype weight =
                        w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2] * w[3][idx3] * scale[3];
                    const NT result = std::apply(
                        [&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, x3, iargs...); }, full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim == 5) {
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadMDRange(team, n[0].size(), n[1].size(), n[2].size(), n[3].size(),
                                            n[4].size()), // range of the kernel
                  [&](const uint idx0, const uint idx1, const uint idx2, const uint idx3, const uint idx4, NT &update) {
                    const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                    const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                    const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                    const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
                    const ctype x4 = Kokkos::fma(scale[4], n[4][idx4], start[4]);
                    const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2] *
                                         w[3][idx3] * scale[3] * w[4][idx4] * scale[4];
                    const NT result = std::apply(
                        [&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, x3, x4, iargs...); }, full_args);
                    update += weight * result;
                  },
                  res);
            } else if constexpr (dim > 5) {
              // make a compiler error
              static_assert(dim <= 5, "QuadratureIntegrator only supports integrals from 1 to 5 dimensions");
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
  };

  template <int dim, typename NT, typename KERNEL>
  class QuadratureIntegrator<dim, NT, KERNEL, TBB_exec> : public AbstractIntegrator
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or
     * possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = TBB_exec;

    QuadratureIntegrator(QuadratureProvider &quadrature_provider, const std::array<uint, dim> grid_size,
                         std::array<ctype, dim> grid_min, std::array<ctype, dim> grid_max,
                         const std::array<QuadratureType, dim> quadrature_type)
        : grid_size(grid_size), quadrature_provider(quadrature_provider)
    {
      for (uint i = 0; i < dim; ++i) {
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
      for (uint i = 0; i < dim; ++i) {
        grid_start[i] = grid_extents[0][i];
        grid_scale[i] = (grid_extents[1][i] - grid_extents[0][i]);
      }
    }

    template <typename... T>
      requires is_valid_kernel<NT, KERNEL, ctype, dim, T...>
    void get(NT &dest, const T &...t) const
    {
      const auto args = std::tie(t...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      if constexpr (dim == 1) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range<uint>(0, n[0].size()), NT(0),
                   [&](const tbb::blocked_range<uint> &r, NT value) -> NT {
                     for (uint idx0 = r.begin(); idx0 != r.end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       const ctype weight = w[0][idx0] * scale[0];
                       const NT result =
                           std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, iargs...); }, args);
                       value += weight * result;
                     }
                     return value;
                   },
                   [&](NT x, NT y) -> NT { return x + y; });
      } else if constexpr (dim == 2) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range2d<uint, uint>(0, n[0].size(), 0, n[1].size()), NT(0.),
                   [&](const tbb::blocked_range2d<uint, uint> &r, NT value) -> NT {
                     for (uint idx0 = r.rows().begin(); idx0 != r.rows().end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       for (uint idx1 = r.cols().begin(); idx1 != r.cols().end(); ++idx1) {
                         const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                         const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1];
                         const NT result =
                             std::apply([&](const auto &...iargs) { return KERNEL::kernel(x0, x1, iargs...); }, args);
                         value += weight * result;
                       }
                     }
                     return value;
                   },
                   [&](NT x, NT y) -> NT { return x + y; });
      } else if constexpr (dim == 3) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range3d<uint, uint, uint>(0, n[0].size(), 0, n[1].size(), 0, n[2].size()), NT(0.),
                   [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                     for (uint idx0 = r.pages().begin(); idx0 != r.pages().end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       for (uint idx1 = r.rows().begin(); idx1 != r.rows().end(); ++idx1) {
                         const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                         for (uint idx2 = r.cols().begin(); idx2 != r.cols().end(); ++idx2) {
                           const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                           const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] * scale[2];
                           const NT result = std::apply(
                               [&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, iargs...); }, args);
                           value += weight * result;
                         }
                       }
                     }
                     return value;
                   },
                   [&](NT x, NT y) -> NT { return x + y; });
      } else if constexpr (dim == 4) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range3d<uint, uint, uint>(0, n[0].size(), 0, n[1].size(), 0, n[2].size()), NT(0.),
                   [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                     for (uint idx0 = r.pages().begin(); idx0 != r.pages().end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       for (uint idx1 = r.rows().begin(); idx1 != r.rows().end(); ++idx1) {
                         const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                         for (uint idx2 = r.cols().begin(); idx2 != r.cols().end(); ++idx2) {
                           const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                           value += tbb::parallel_reduce(
                               tbb::blocked_range<uint>(0, n[3].size()), NT(0.),
                               [&](const tbb::blocked_range<uint> &r2, NT value2) -> NT {
                                 for (uint idx3 = r2.begin(); idx3 != r2.end(); ++idx3) {
                                   const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
                                   const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] *
                                                        scale[2] * w[3][idx3] * scale[3];
                                   const NT result = std::apply(
                                       [&](const auto &...iargs) { return KERNEL::kernel(x0, x1, x2, x3, iargs...); },
                                       args);
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
      } else if constexpr (dim == 5) {
        dest = KERNEL::constant(t...) +
               tbb::parallel_reduce(
                   tbb::blocked_range3d<uint, uint, uint>(0, n[0].size(), 0, n[1].size(), 0, n[2].size()), NT(0.),
                   [&](const tbb::blocked_range3d<uint, uint, uint> &r, NT value) -> NT {
                     for (uint idx0 = r.pages().begin(); idx0 != r.pages().end(); ++idx0) {
                       const ctype x0 = Kokkos::fma(scale[0], n[0][idx0], start[0]);
                       for (uint idx1 = r.rows().begin(); idx1 != r.rows().end(); ++idx1) {
                         const ctype x1 = Kokkos::fma(scale[1], n[1][idx1], start[1]);
                         for (uint idx2 = r.cols().begin(); idx2 != r.cols().end(); ++idx2) {
                           const ctype x2 = Kokkos::fma(scale[2], n[2][idx2], start[2]);
                           value += tbb::parallel_reduce(
                               tbb::blocked_range2d<uint, uint>(0, n[3].size(), 0, n[4].size()), NT(0.),
                               [&](const tbb::blocked_range2d<uint, uint> &r2, NT value2) -> NT {
                                 for (uint idx3 = r2.rows().begin(); idx3 != r2.rows().end(); ++idx3) {
                                   const ctype x3 = Kokkos::fma(scale[3], n[3][idx3], start[3]);
                                   for (uint idx4 = r2.cols().begin(); idx4 != r2.cols().end(); ++idx4) {
                                     const ctype x4 = Kokkos::fma(scale[4], n[4][idx4], start[4]);
                                     const ctype weight = w[0][idx0] * scale[0] * w[1][idx1] * scale[1] * w[2][idx2] *
                                                          scale[2] * w[3][idx3] * scale[3] * w[4][idx4] * scale[4];
                                     const NT result = std::apply(
                                         [&](const auto &...iargs) {
                                           return KERNEL::kernel(x0, x1, x2, x3, x4, iargs...);
                                         },
                                         args);
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
      } else if constexpr (dim > 5) {
        // make a compiler error
        static_assert(dim <= 5, "QuadratureIntegrator only supports integrals "
                                "from 1 to 4 dimensions");
      }
    }

    template <typename Coordinates, typename... Args>
    void map(execution_space &, NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      const auto m_args = std::tie(args...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

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

        map(space, dest_offset, sub_coordinates, args...);
      } else
        map(space, dest, coordinates, args...);
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
  };

} // namespace DiFfRG
