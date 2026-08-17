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
#include <DiFfRG/physics/integration/map_completion.hh>

// std
#include <cstdio>
#include <string>

namespace DiFfRG
{
  /**
   * @brief Whether a coordinates type carries enough identity for QuadratureIntegrator::map() to
   * cache its forward()-transformed positions in a device view (one forward() per grid point
   * instead of per thread). Namespace-scope on purpose: it is referenced inside extended device
   * lambdas, where nvcc mishandles function-local constexpr variables.
   */
  template <typename Coordinates>
  inline constexpr bool has_cacheable_positions_v = requires(const Coordinates &c) {
    c.to_string();
    c.forward(c.from_linear_index(size_t(0)));
  };

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
        : space(quadrature_provider.template next_execution_space<ExecutionSpace>()),
          quadrature_provider(quadrature_provider)
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
                              make_kokkos_nd_range<dim, ExecutionSpace>(space, {0}, grid_size),
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
          m_cache = make_kokkos_nd_view<1 + dim, NT, ExecutionSpace>("cache", m_cache_extents);
        }
      }
      // Create a Restrict-tagged alias of the cache for no-alias optimization
      const auto cache = KokkosNDViewRestrict<1 + dim, NT, ExecutionSpace>(m_cache);

      const auto m_args = device::make_tuple(args...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      // The external position is a function of idx[0] alone: at most integral_view.size() distinct
      // values per launch, while the functor below runs size * prod(grid_size) threads. For the
      // logarithmic coordinate classes forward() is a fp64 expm1/sinh+exp, so recomputing it per
      // thread wastes one transcendental per thread. Precompute the positions once per coordinate
      // system into a device view; the fill runs the very same forward() on the same device, so the
      // cached values are bit-identical to what the per-thread computation produced.
      //
      // Coordinates without a to_string() identity keep the per-thread computation. SubCoordinates
      // (the MPI path) is *not* one of them -- it has its own to_string(), and the key construction
      // below was written to separate two windows into the same base grid. The only cost on that
      // path is that a schedule which moves a slice boundary re-keys and re-runs the ~5 us fill.
      constexpr size_t cdim = Coordinates::dim;
      // Compile-time, so the un-taken branch in the device functors below is dead code and the
      // per-thread forward() (a fp64 expm1/sinh+exp on the log grids) actually leaves the kernel.
      if constexpr (has_cacheable_positions_v<Coordinates>) {
        // The key must identify the positions, not just the coordinate parameters: SubCoordinates
        // inherits its base's to_string(), so two windows into the same base grid would otherwise
        // collide. Appending the exact (hex-formatted) first and last positions separates them.
        std::string key = coordinates.to_string() + "|" + std::to_string(integral_view.size());
        {
          char buf[64];
          const auto first = coordinates.forward(coordinates.from_linear_index(size_t(0)));
          const auto last = coordinates.forward(coordinates.from_linear_index(integral_view.size() - 1));
          for (size_t d = 0; d < cdim; ++d) {
            std::snprintf(buf, sizeof(buf), "|%la|%la", double(first[d]), double(last[d]));
            key += buf;
          }
        }
        const size_t need = integral_view.size() * cdim;
        if (m_positions_key != key || m_positions.extent(0) < need) {
          if (m_positions.extent(0) < need)
            m_positions = Kokkos::View<ctype *, typename ExecutionSpace::memory_space>(
                Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "QuadratureIntegrator_positions"), need);
          const auto pos_fill = m_positions;
          const auto coords = coordinates;
          Kokkos::parallel_for(
              "QuadratureIntegrator_fill_positions",
              Kokkos::RangePolicy<ExecutionSpace>(space, 0, integral_view.size()), KOKKOS_LAMBDA(const size_t i) {
                const auto p = coords.forward(coords.from_linear_index(i));
                for (size_t d = 0; d < cdim; ++d)
                  pos_fill(i * cdim + d) = p[d];
              });
          m_positions_key = key;
        }
      }
      const auto pos_view = m_positions;
      using pos_ctype = typename Coordinates::ctype;
      // Runtime copy for the team lambda below: its constant() evaluation runs once per team, so
      // keeping the transform code there costs nothing, and a plain branch avoids nvcc's fragile
      // handling of if-constexpr inside extended class lambdas.
      const bool pos_cached_rt = has_cacheable_positions_v<Coordinates>;

      // Two complete functors, selected by a HOST-level if constexpr. nvcc's extended-lambda
      // transformation miscompiles `if constexpr` INSIDE the lambda body here (wrong or
      // unregistered kernel stubs: cudaErrorInvalidResourceHandle, or a silently zero RHS), while
      // a lambda DEFINED inside an if-constexpr block is fine (the position fill above). So the
      // branch lives out here and each lambda body is branch-free.
      //
      // No explicit tile: Kokkos' own heuristic is used. An explicit tile matched to
      // DIFFRG_LAUNCH_BOUNDS was measured and is INERT (numtracer/gpubench/FINDINGS.md) -- the
      // 1.3-1.5x it appeared to give was GPU clock ramp, not tiling. Not worth the coupling: Kokkos
      // hard-aborts when the tile product exceeds LaunchBounds, so an explicit tile turns a future
      // launch-bounds change into a runtime abort.
      if constexpr (has_cacheable_positions_v<Coordinates>) {
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
        {
          // make subview
          auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

          // get the (precomputed) position for the current index
          device::array<pos_ctype, cdim> pos;
          for (size_t d = 0; d < cdim; ++d)
            pos[d] = static_cast<pos_ctype>(pos_view(idx[0] * cdim + d));

          device::array<ctype, dim> x;
          ctype weight = 1;
          for (int i = 0; i < dim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
            weight *= w[i][idx[1 + i]] * scale[i];
          }

          // Apply x/pos and the trailing arguments as two NESTED packs. Do not be tempted to
          // tuple_cat them into one tuple and apply that once: tuple_cat builds a by-value object,
          // and m_args holds every interpolator by value (device::make_tuple decays), so the
          // concatenated tuple is a full per-thread copy of all of them in local memory. That cost
          // 3576 B of stack frame on QCD_Nf2 (13 interpolators x 272 B) -- 89% of that binary's
          // 4000 B frame -- and 2.3x of runtime, measured. Applying an *lvalue* tuple binds
          // references instead, and KERNEL::kernel takes all of them by const&.
          // See docs/NUMTRACER_PER_THREAD_FRAME.md in the numtracer repo.
          device::apply(
              [&](const auto &...xargs) {
                device::apply([&](const auto &...iargs) { subview() = weight * KERNEL::kernel(xargs..., iargs...); },
                              m_args);
              },
              device::tuple_cat(x, pos));
        };
        Kokkos::parallel_for(make_kokkos_nd_range_divisible<1 + dim, ExecutionSpace>(space, {0}, extents),
                             KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));
      } else {
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

          // Apply x/pos and the trailing arguments as two NESTED packs. Do not be tempted to
          // tuple_cat them into one tuple and apply that once: tuple_cat builds a by-value object,
          // and m_args holds every interpolator by value (device::make_tuple decays), so the
          // concatenated tuple is a full per-thread copy of all of them in local memory. That cost
          // 3576 B of stack frame on QCD_Nf2 (13 interpolators x 272 B) -- 89% of that binary's
          // 4000 B frame -- and 2.3x of runtime, measured. Applying an *lvalue* tuple binds
          // references instead, and KERNEL::kernel takes all of them by const&.
          // See docs/NUMTRACER_PER_THREAD_FRAME.md in the numtracer repo.
          device::apply(
              [&](const auto &...xargs) {
                device::apply([&](const auto &...iargs) { subview() = weight * KERNEL::kernel(xargs..., iargs...); },
                              m_args);
              },
              device::tuple_cat(x, pos));
        };
        Kokkos::parallel_for(make_kokkos_nd_range_divisible<1 + dim, ExecutionSpace>(space, {0}, extents),
                             KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));
      }

      using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
      // reduction with vector lanes for warp-level parallelism
      constexpr int vector_width = 32;
      Kokkos::parallel_for(
          Kokkos::TeamPolicy(space, integral_view.size(), Kokkos::AUTO, vector_width),
          KOKKOS_CLASS_LAMBDA(const TeamType &team) {
            // get the current (continuous) index
            const uint k = team.league_rank();

            if (k >= integral_view.size()) return;

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
              device::array<pos_ctype, cdim> pos;
              if (pos_cached_rt) {
                for (size_t d = 0; d < cdim; ++d)
                  pos[d] = static_cast<pos_ctype>(pos_view(size_t(k) * cdim + d));
              } else {
                pos = coordinates.forward(coordinates.from_linear_index(k));
              }
              // Nested packs, not tuple_cat -- same reason as the phase-1 functor above. This
              // kernel is only 0.3-4% of GPU time (docs/NUMTRACER_GPU_INVESTIGATION.md) but it
              // carried the whole per-thread copy too: 3544 B of stack frame on a kernel that does
              // almost no arithmetic.
              integral_view(k) =
                  res + device::apply(
                            [&](const auto &...pargs) {
                              return device::apply(
                                  [&](const auto &...iargs) { return KERNEL::constant(pargs..., iargs...); }, m_args);
                            },
                            pos);
            });
          });
    }

    /// Points evaluated per external grid point. Half of the scheduler's cost score.
    size_t quadrature_volume() const
    {
      size_t volume = 1;
      for (int i = 0; i < dim; ++i)
        volume *= grid_size[i];
      return volume;
    }

    template <typename Coordinates, typename... Args>
    auto map(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      auto &scheduler = MapScheduler::instance();

      // One staging buffer per integrator, so a second map() from this integrator before a flush
      // would clobber the first result. The *plan* is what decides, not the local pending list: a
      // rank that owned no slice of the earlier map has nothing pending and would skip a flush the
      // other ranks perform, leaving the collectives mismatched.
      if (scheduler.active() && scheduler.plan_contains(integrator_id())) MapCompletion::flush();

      const MapSlice slice = scheduler.schedule(integrator_id(), dest, sizeof(NT), coordinates.size(),
                                                quadrature_volume(), Coordinates::dim == 1);

      if (slice.count == 0) {
        // Not an owner: no kernels, but the plan entry is registered, so this rank still has to
        // take part in the exchange.
        if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
        return ExecutionSpace();
      }
      if (slice.owns_all(coordinates.size())) return map_dist(dest, coordinates, args...);

      return map_dist(dest + slice.offset, SubCoordinates(coordinates, slice.offset, slice.count), args...);
    }

    template <typename Coordinates, typename... Args>
    auto map_dist(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      const size_t n = coordinates.size();

      // Reuse cached device view if large enough, otherwise reallocate (grow-only)
      if (m_dest_device_size < n) {
        m_dest_device =
            Kokkos::View<NT *, ExecutionSpace>(Kokkos::view_alloc(space, "MapIntegrators_device_view"), n);
        m_dest_device_size = n;
      }
      auto dest_device_view = Kokkos::View<NT *, ExecutionSpace>(m_dest_device, Kokkos::make_pair(size_t(0), n));

      if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, CPU_memory>) {
        // Host backend: "device" memory is host memory, so there is nothing to stage or defer.
        auto dest_view = Kokkos::View<NT *, CPU_memory, Kokkos::MemoryUnmanaged>(dest, n);
        map(space, dest_device_view, coordinates, args...);
        Kokkos::deep_copy(space, dest_view, dest_device_view);
        // Nothing to land, but the MPI slices still have to be exchanged before the caller reads.
        if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
        return space;
      } else {
        // One staging buffer per integrator, so a second map() before a flush would clobber the
        // first result. Land the outstanding one first; in the normal call pattern (each flow
        // mapped once per flush interval) this never triggers.
        if (m_dest_pinned_size > 0 && MapCompletion::has_pending(m_dest_pinned.data())) MapCompletion::flush();

        if (m_dest_pinned_size < n) {
          m_dest_pinned = Kokkos::View<NT *, PinnedHost_memory>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "MapIntegrators_pinned_view"), n);
          m_dest_pinned_size = n;
        }
        auto pinned_view =
            Kokkos::View<NT *, PinnedHost_memory>(m_dest_pinned, Kokkos::make_pair(size_t(0), n));

        map(space, dest_device_view, coordinates, args...);

        // Genuinely asynchronous, because the destination is page-locked.
        Kokkos::deep_copy(space, pinned_view, dest_device_view);

        // Caller has promised not to read `dest` until its DeferredMaps scope closes, so leave the
        // result in staging and keep the host running ahead of the device.
        MapCompletion::record(dest, m_dest_pinned.data(), n * sizeof(NT));
        // Original contract outside such a scope: `dest` is valid on return. flush() fences, lands
        // the staged copy and -- under MPI -- exchanges this batch's slices, all of which must
        // happen before the caller looks at `dest`.
        if (!MapCompletion::deferral_enabled()) MapCompletion::flush();

        return space;
      }
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
    // Cached external positions for map(): one forward() per grid point instead of per thread.
    // Keyed on the coordinates' to_string() identity; flat layout [grid_point * cdim + d].
    mutable Kokkos::View<ctype *, typename ExecutionSpace::memory_space> m_positions;
    mutable std::string m_positions_key;
    mutable Kokkos::View<NT *, ExecutionSpace> m_dest_device;
    mutable size_t m_dest_device_size = 0;
    // Page-locked staging for the result copy; see map_dist() and MapCompletion.
    mutable Kokkos::View<NT *, PinnedHost_memory> m_dest_pinned;
    mutable size_t m_dest_pinned_size = 0;
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
      auto &scheduler = MapScheduler::instance();

      // A second map() from this integrator would overwrite `dest` before the first one's slices
      // have been exchanged, so land the open batch first. Plan-driven, not local -- see the GPU
      // overload for why that distinction is what keeps the collectives matched.
      if (scheduler.active() && scheduler.plan_contains(this->integrator_id())) MapCompletion::flush();

      const MapSlice slice = scheduler.schedule(this->integrator_id(), dest, sizeof(NT), coordinates.size(),
                                                Base::quadrature_volume(), Coordinates::dim == 1);

      if (slice.count == 0) {
        // Not an owner, but still part of the batch -- see the GPU overload.
        if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
        return space;
      }
      if (slice.owns_all(coordinates.size()))
        map(space, dest, coordinates, args...);
      else
        map(space, dest + slice.offset, SubCoordinates(coordinates, slice.offset, slice.count), args...);

      // This backend writes straight into `dest`, so there is nothing to land -- but the slices
      // still have to be exchanged before the caller reads them.
      if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
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
