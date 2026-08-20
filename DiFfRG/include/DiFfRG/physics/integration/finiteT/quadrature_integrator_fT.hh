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
// for has_cacheable_positions_v, shared with the vacuum integrator
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

// std
#include <cstdio>
#include <string>

namespace DiFfRG
{
  // True iff the kernel declares `static constexpr bool matsubara_even = true`. Such a
  // kernel was generated as the EVEN part in the Matsubara frequency, so the ±frequency
  // sum kernel(+xt)+kernel(-xt) == 2*kernel(xt) and the kernel is evaluated only once.
  // If the trait is absent or false, the integrator falls back to the explicit two-call
  // form (always correct — an even kernel also satisfies kernel(xt)==kernel(-xt)).
  template <class K> inline constexpr bool kernel_is_matsubara_even = requires { requires K::matsubara_even; };

  // True iff the kernel declares `static constexpr bool matsubara_finite_extent = true`, i.e. every
  // term it sums carries a dR/dt insertion whose argument confines the loop FREQUENCY, so the
  // summand vanishes identically outside |p0| <= the frequency cutoff the integrator is given.
  // Such a sum is finite and exact: enumerating the modes beats approximating the infinite sum
  // with a Gaussian rule. If the trait is absent or false, the integrator keeps the Monien/vacuum
  // rule, which is always correct -- a summand of finite extent is also an integrable one.
  template <class K> inline constexpr bool kernel_has_finite_matsubara_extent = requires { requires K::matsubara_finite_extent; };

  // True iff the kernel declares `static constexpr bool matsubara_split = true`, i.e. it was
  // generated as a MIXED flow and offers two entry points, `kernel_finite_extent` (the terms whose
  // dR/dt insertion confines p0) and `kernel_tail` (the rest), which sum to `kernel`.
  //
  // The integrator then runs ONE launch over the concatenated axis [tail nodes | finite-extent
  // nodes], evaluating only the matching half at each node. That is where the mixed case pays: a
  // term's cost is dominated by the single trace it calls, and the generator's CSE is per-function,
  // so each half's body computes only the traces it uses. On QCD_Nf2's ZA4 the one unbounded term
  // carries a 313-line trace while the five confined ones carry 2918 + 609 + ... -- so the cheap
  // half is what runs the full Gaussian rule and the expensive half runs a handful of exact modes.
  template <class K> inline constexpr bool kernel_has_matsubara_split = requires { requires K::matsubara_split; };

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
     * @brief Execution space to be used for the integration, e.g. GPU_exec, TBB_exec.
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
        : space(quadrature_provider.template next_execution_space<ExecutionSpace>()),
          quadrature_provider(quadrature_provider), T(T), typical_E(typical_E),
          // A SPLIT kernel does not carry `matsubara_finite_extent` -- only its finite-extent HALF
          // has that property -- so it has to opt in here too, or its expensive half would keep
          // running the Gaussian rule and the split would be pure overhead.
          m_allow_exact(kernel_has_finite_matsubara_extent<KERNEL> || kernel_has_matsubara_split<KERNEL>)
    {
      for (int d = 0; d < sdim; ++d)
        grid_size[d] = _grid_size[d];
      for (int i = 0; i < sdim; ++i) {
        nodes[i] = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[i],
                                                                                                    quadrature_type[i]);
        weights[i] = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
            grid_size[i], quadrature_type[i]);
      }
      set_grid_extents(grid_min, grid_max);
      refresh_matsubara();
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
      refresh_matsubara();
    }

    void set_typical_E(const ctype typical_E)
    {
      if (is_close(this->typical_E, typical_E, 1e-4 * T + std::numeric_limits<ctype>::epsilon() * 10)) return;

      this->typical_E = typical_E;
      refresh_matsubara();
    }

    /**
     * @brief The frequency beyond which the summand is known to vanish, enabling the exact sum.
     *
     * Only meaningful for a kernel whose every term carries a `dR/dt` insertion that confines the
     * loop frequency (see kernel_has_finite_matsubara_extent). For a 4D regulator of extent `x_extent`
     * (in `q^2/k^2`) the summand's support is the ball `q0^2 + |q|^2 <= x_extent * k^2`, so the
     * cutoff is `sqrt(x_extent) * k` -- the SAME number the spatial grid is already cut at, which
     * is why the wrappers can supply it without any new configuration.
     *
     * Passing zero (the default) disables the exact sum and keeps the Monien/vacuum rule.
     */
    void set_frequency_cutoff(const ctype freq_cutoff)
    {
      if (is_close(m_freq_cutoff, freq_cutoff, 1e-10 * std::fabs(freq_cutoff))) return;
      m_freq_cutoff = freq_cutoff;
      refresh_matsubara();
    }

    /**
     * @brief Force the exact sum on (or off) regardless of the kernel's trait.
     *
     * The trait is generated from the diagram algebra and is the right default, but a model that
     * knows better -- or a study that wants to price the exact sum against the Gaussian rule on the
     * same kernel -- needs to be able to say so. Forcing it ON for a kernel whose summand does NOT
     * vanish above the cutoff silently truncates the sum.
     */
    void set_allow_exact_matsubara_sum(const bool allow)
    {
      if (m_allow_exact == allow) return;
      m_allow_exact = allow;
      refresh_matsubara();
    }

    /**
     * @brief Multiply the frequency cutoff, i.e. how far past the regulator's support the exact sum
     * keeps summing modes.
     *
     * The cutoff it is handed comes from `x_extent`, which `optimize_x_extent` sizes for SPATIAL
     * quadrature convergence -- a different question from where a discrete sum may be truncated.
     * In practice it overshoots (the 1.15 stepping lands well past the requested tolerance), so the
     * exact sum inherits the error budget the spatial grid already accepts rather than adding to it.
     * This knob is how one CHECKS that rather than assuming it: raise it to 2 and the answer must
     * not move. Modes cost only linearly, so the check is cheap.
     */
    /**
     * @brief Order of the finite-interval frequency rule; 0 (the default) means the SPATIAL order.
     *
     * A summand of finite extent has no tail, and for a 4D regulator `p0` and `|q|` sit in
     * `p0^2 + |q|^2 <= x_extent k^2` symmetrically -- so the order that resolves the spatial radius
     * is the natural one for the frequency too, and `grid_size[0]` (the flow's `x_order`) is the
     * default rather than the `vacuum_quad_size` that sized the tangent map.
     *
     * Know the trade before changing it, because it is NOT free and it is not visible per step.
     * Measured on QCD_Nf2 (x_order = 32, vacuum_quad_size = 64), against a converged reference:
     *
     *   per step, k ~ 1e3   order 32: 6.0e-6     order 64: 6.2e-6   -- indistinguishable
     *   over t = 0..6       order 32: 1.75e-2    order 64: 1.87e-3  -- 9x apart
     *
     * i.e. the frequency error ACCUMULATES over a flow in a way one step does not reveal, and the
     * tangent map this replaced sits at 1.18e-2 in the same comparison. So order 32 is ~1.5x less
     * accurate than the old behaviour end to end, and order 64 is ~6x more accurate than it while
     * still being faster. Raise this to `vacuum_quad_size` if the frequency direction turns out to
     * matter for the observable in hand.
     */
    void set_frequency_order(const size_t order)
    {
      if (m_freq_order == order) return;
      m_freq_order = order;
      refresh_matsubara();
    }

    /// The order actually in use for the finite-interval rule.
    size_t frequency_order() const { return m_freq_order != 0 ? m_freq_order : grid_size[0]; }

    void set_matsubara_extent_margin(const ctype margin)
    {
      if (!(margin > ctype(0)) || is_close(m_extent_margin, margin)) return;
      m_extent_margin = margin;
      refresh_matsubara();
    }

    /// Nodes on the frequency axis, INCLUDING the zero mode.
    size_t get_matsubara_size() const { return matsubara_nodes.size(); }

    /// True if the frequency axis is currently the exact sum rather than a Gaussian quadrature rule.
    bool uses_exact_matsubara_sum() const { return m_using_exact; }

    /**
     * @brief One Matsubara node's contribution at one spatial point, weight excluded.
     *
     * Factored out of get()/map() so the +-frequency and zero-mode logic exists once, and --
     * importantly -- so the `if constexpr` on matsubara_even lives in an ordinary __device__
     * function rather than inside an extended lambda, where nvcc's transformation is fragile.
     *
     * The zero mode is NOT special-cased here: it arrives as an ordinary node `(0, T/2)` at the
     * end of the node list (see MatsubaraQuadrature::sum_nodes), and `w * (f(+0) + f(-0))`
     * reproduces `T f(0)` exactly. The branch this replaces cost the whole warp a full extra
     * kernel evaluation for one active lane.
     *
     * The three packs are applied as NESTED lvalue tuples. Do not tuple_cat them: tuple_cat builds
     * a by-value object, and `args` holds every interpolator by value, so the concatenated tuple
     * becomes a full per-thread copy of all of them in local memory. On QCD_Nf2 that was 3576 B of
     * stack frame and 2.3x of runtime in the vacuum integrator (see the comment in
     * QuadratureIntegrator::map). Applying an lvalue tuple binds references instead.
     */
    template <typename XArr, typename PosArr, typename ArgTuple>
    KOKKOS_FORCEINLINE_FUNCTION static NT node_value(const XArr &x, const PosArr &pos, const ArgTuple &args,
                                                     const ctype xt, const ctype wt, const bool is_tail)
    {
      NT out{};
      device::apply(
          [&](const auto &...xargs) {
            device::apply(
                [&](const auto &...pargs) {
                  device::apply(
                      [&](const auto &...iargs) {
                        // Six near-identical lines rather than the obvious factoring of the
                        // +-frequency rule into a lambda taking the entry point. That factoring was
                        // written and reverted: expanding an OUTER generic lambda's parameter pack
                        // inside a nested lambda makes nvcc drop the pack, and it fails as "too few
                        // arguments" only for the instantiation where the outer packs are EMPTY
                        // (dim == 1, no spatial variables) -- i.e. it would have compiled here and
                        // broken a 1-D flow somewhere else.
                        NT msum;
                        if constexpr (kernel_has_matsubara_split<KERNEL>) {
                          // Runtime branch on the node index, not on data: within a frequency row
                          // there is exactly ONE boundary, so at most one warp per row straddles it
                          // and runs both bodies -- which together cost what the unsplit kernel
                          // costs today. The split can therefore not be slower than not splitting.
                          if (is_tail) {
                            if constexpr (kernel_is_matsubara_even<KERNEL>)
                              msum = ctype(2) * KERNEL::kernel_tail(xargs..., xt, pargs..., iargs...);
                            else
                              msum = KERNEL::kernel_tail(xargs..., xt, pargs..., iargs...) +
                                     KERNEL::kernel_tail(xargs..., -xt, pargs..., iargs...);
                          } else {
                            if constexpr (kernel_is_matsubara_even<KERNEL>)
                              msum = ctype(2) * KERNEL::kernel_finite_extent(xargs..., xt, pargs..., iargs...);
                            else
                              msum = KERNEL::kernel_finite_extent(xargs..., xt, pargs..., iargs...) +
                                     KERNEL::kernel_finite_extent(xargs..., -xt, pargs..., iargs...);
                          }
                        } else {
                          if constexpr (kernel_is_matsubara_even<KERNEL>)
                            // even kernel: kernel(+xt)+kernel(-xt) == 2*kernel(xt) (one evaluation)
                            msum = ctype(2) * KERNEL::kernel(xargs..., xt, pargs..., iargs...);
                          else
                            // positive and negative Matsubara frequencies
                            msum = KERNEL::kernel(xargs..., xt, pargs..., iargs...) +
                                   KERNEL::kernel(xargs..., -xt, pargs..., iargs...);
                        }
                        out = wt * msum;
                      },
                      args);
                },
                pos);
          },
          x);
      return out;
    }

    template <typename... T> void get(NT &dest, const T &...t) const
    {
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
      requires(!std::is_same_v<OT, NT>)
    void get(OT &dest, const T &...t) const
    {
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
      // Nodes before the boundary of the concatenated axis; 0 for an unsplit kernel, where the
      // is_tail flag is dead code that the `if constexpr` in node_value() removes anyway.
      const size_t n_tail = m_n_tail;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

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
        const size_t jt = idx[dim - 1];
        // Empty position pack: this overload's caller passes the external position inside `args`.
        update += weight * node_value(x, device::tuple<>{}, args, m_n[jt], m_w[jt], jt < n_tail);
        device::apply([&](const auto &...iargs) { update += is_first ? KERNEL::constant(iargs...) : NT(0); }, args);
      };

      Kokkos::parallel_reduce("QuadratureIntegral_fT_" + std::to_string(dim) + "D", // name of the kernel
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
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      // Nodes before the boundary of the concatenated axis; 0 for an unsplit kernel, where the
      // is_tail flag is dead code that the `if constexpr` in node_value() removes anyway.
      const size_t n_tail = m_n_tail;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      // The external position is a function of idx[0] alone, but this functor runs
      // integral_view.size() * prod(grid_size) threads -- and grid_size carries the Matsubara axis,
      // so the per-thread forward() (a fp64 expm1/sinh+exp on the logarithmic coordinate classes)
      // is paid tens of times more often here than in the vacuum integrator. Precompute the
      // positions once per coordinate system into a device view, exactly as
      // QuadratureIntegrator::map does; see the comments there for why the key is built this way
      // and why coordinates without a to_string() identity keep the per-thread computation.
      constexpr size_t cdim = Coordinates::dim;
      if constexpr (has_cacheable_positions_v<Coordinates>) {
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
                Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "QuadratureIntegrator_fT_positions"), need);
          const auto pos_fill = m_positions;
          const auto coords = coordinates;
          Kokkos::parallel_for(
              "QuadratureIntegrator_fT_fill_positions",
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
      // Runtime copy for the team lambda below: its constant() evaluation runs once per team, so a
      // plain branch there costs nothing and avoids nvcc's fragile handling of if-constexpr inside
      // extended class lambdas.
      const bool pos_cached_rt = has_cacheable_positions_v<Coordinates>;

      // Two complete functors, selected by a HOST-level if constexpr -- nvcc miscompiles an
      // `if constexpr` inside the extended lambda body. See QuadratureIntegrator::map.
      if constexpr (has_cacheable_positions_v<Coordinates>) {
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
        {
          // make subview
          auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

          // get the (precomputed) position for the current index
          device::array<pos_ctype, cdim> pos;
          for (size_t d = 0; d < cdim; ++d)
            pos[d] = static_cast<pos_ctype>(pos_view(idx[0] * cdim + d));

          device::array<ctype, sdim> x;
          ctype weight = 1;
          for (int i = 0; i < sdim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
            weight *= w[i][idx[1 + i]] * scale[i];
          }
          const size_t jt = idx[1 + dim - 1];
          subview() = weight * node_value(x, pos, m_args, m_n[jt], m_w[jt], jt < n_tail);
        };
        Kokkos::parallel_for(make_kokkos_nd_range_divisible<1 + dim, ExecutionSpace>(space, {0}, extents),
                             KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));
      } else {
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
        {
          // make subview
          auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

          // get the position for the current index
          const auto pos = coordinates.forward(coordinates.from_linear_index(idx[0]));

          device::array<ctype, sdim> x;
          ctype weight = 1;
          for (int i = 0; i < sdim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
            weight *= w[i][idx[1 + i]] * scale[i];
          }
          const size_t jt = idx[1 + dim - 1];
          subview() = weight * node_value(x, pos, m_args, m_n[jt], m_w[jt], jt < n_tail);
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

            // Pre-compute stride array for index decomposition (avoids modulo in the inner loop;
            // matches the vacuum QuadratureIntegrator::map reduction)
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
              // Nested packs, not tuple_cat -- same reason as node_value(). This kernel does almost
              // no arithmetic but carried a full per-thread copy of every interpolator.
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

      // See QuadratureIntegrator::map() for why this is decided from the plan, not from local state.
      if (scheduler.active() && scheduler.plan_contains(integrator_id())) MapCompletion::flush();

      const MapSlice slice = scheduler.schedule(integrator_id(), dest, sizeof(NT), coordinates.size(),
                                                quadrature_volume(), /* splittable */ true,
                                                map_target<ExecutionSpace>());

      if (slice.count == 0) {
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

      if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, CPU_memory>) {
        // Host backend: "device" memory is host memory, so there is nothing to stage. The work is
        // synchronous though, so inside a deferral scope it is queued rather than run here -- see
        // run_or_queue_host().
        run_or_queue_host(dest, coordinates, args...);
        // Nothing to land, but the MPI slices still have to be exchanged before the caller reads.
        if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
        return space;
      } else {
        // One staging buffer per integrator, so a second map() before a flush would clobber the
        // first result. Land the outstanding one first; in the normal call pattern (each flow
        // mapped once per flush interval) this never triggers.
        if (m_dest_pinned_size > 0 && MapCompletion::has_pending(m_dest_pinned.data())) MapCompletion::flush();

        auto dest_device_view = device_scratch(n);

        if (m_dest_pinned_size < n) {
          m_dest_pinned = Kokkos::View<NT *, PinnedHost_memory>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "MapIntegrators_fT_pinned_view"), n);
          m_dest_pinned_size = n;
        }
        auto pinned_view = Kokkos::View<NT *, PinnedHost_memory>(m_dest_pinned, Kokkos::make_pair(size_t(0), n));

        map(space, dest_device_view, coordinates, args...);

        // Genuinely asynchronous, because the destination is page-locked. Copying straight into
        // `dest` -- ordinary pageable caller memory, e.g. a dealii::Vector element range -- is not:
        // the driver has to stage it, so the call blocks until the kernels feeding it have finished
        // and the host gets no run-ahead at all. See MapCompletion for the measurement.
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

  private:
    /**
     * @brief Re-select and re-fetch the frequency rule after T, typical_E or the cutoff changed.
     *
     * Two rules are on offer and the choice is purely one of cost: for a summand of finite extent
     * the exact sum is never *less* accurate than the Gaussian rule -- it is the sum
     * itself -- so whichever has fewer nodes wins. That crossover is crossed during a flow (the
     * exact sum shrinks with k while the Monien rule grows), which is why this is decided per RG step.
     */
    void refresh_matsubara()
    {
      // The kernel must be COMPLETE here, because this function branches on its traits. Every
      // `requires { requires K::trait; }` detector reads false on an incomplete K -- a substitution
      // failure, not an error -- so a translation unit that has not seen the kernel definition
      // would build a different integrator than one that has, pick a different Matsubara rule, and
      // produce a wrong right-hand side with no diagnostic anywhere. That is not hypothetical: the
      // generated flow scaffolds used to forward-declare their kernel in <Flow>.hh, and flows.cc
      // (which instantiates set_T -> here) disagreed with the CT_*.cc translation units (which
      // instantiate map()). Fail loudly instead.
      static_assert(sizeof(KERNEL) > 0,
                    "QuadratureIntegrator_fT: the KERNEL type must be complete here. Include the "
                    "flow's kernel.hh before its <Flow>.hh -- a forward declaration silently turns "
                    "every kernel trait off in this translation unit and yields a wrong RHS.");

      using mem_space = typename ExecutionSpace::memory_space;

      const auto &standard = quadrature_provider.template matsubara_rule<ctype>(T, typical_E);
      m_using_exact = false;

      // Which rule the finite-extent side should run on. Never *less* accurate than the Gaussian
      // rule -- it is the sum itself -- so the choice is purely one of cost, and the crossover is
      // crossed during a flow (the exact sum shrinks with k while the Monien rule grows), which is
      // why it is decided per RG step.
      const MatsubaraQuadrature<ctype> *fe_rule = &standard;
      if (m_allow_exact && T > ctype(0) && m_freq_cutoff > ctype(0)) {
        // One mode of margin. A FERMIONIC insertion confines p0 to an interval centred at -pi T,
        // not at zero, so its support can reach one bosonic mode further in the negative direction
        // than a symmetric list built from the cutoff alone would cover. One node is a cheap price
        // for not having to know which species a given kernel's insertion belongs to.
        const ctype cutoff = m_extent_margin * m_freq_cutoff + ctype(2 * M_PI) * T;
        // Price it BEFORE building it. modes_below() is three flops; the rule itself is an O(N)
        // table and six Kokkos views, and in the UV the answer is thousands of modes that would be
        // discarded on the very next line. Measured on QCD_Nf2 at T = 0.1: every step from k = 1000
        // down to k = 150 built and threw away a rule, 1963 modes at the top and still 294 at the
        // bottom -- and the provider's map never evicts, so each one leaked for the process.
        const size_t n_exact = size_t(MatsubaraQuadrature<ctype>::modes_below(T, cutoff)) + 1;
        if (n_exact < standard.sum_size()) {
          fe_rule = &quadrature_provider.template matsubara_exact_sum<ctype>(T, cutoff);
          m_using_exact = true;
        } else if (is_close(standard.get_T(), ctype(0))) {
          // Above the crossover the exact sum is the expensive rule -- but `standard` is then the
          // T=0 TANGENT MAP, which spends 43% of its nodes past this summand's support evaluating
          // an exact zero, and weights those most heavily. Over a finite interval there is nothing
          // to reach for, so plain Gauss-Legendre at the SPATIAL order is both cheaper and more
          // accurate. (If `standard` is the Monien rule instead, the sum is genuinely thermal and
          // must not be replaced by an integral, so leave it alone.)
          fe_rule = &quadrature_provider.template matsubara_finite_interval<ctype>(
              m_extent_margin * m_freq_cutoff, frequency_order());
        }
      }

      // sum_nodes(), not nodes(): the zero mode is an ordinary node on this axis, so the hot
      // functor has no branch and an exact sum with no positive modes at all still evaluates it.
      if constexpr (kernel_has_matsubara_split<KERNEL>) {
        // Concatenate: [ tail half on the Gaussian rule | finite-extent half on fe_rule ]. Each
        // half carries its own zero mode, because each is a complete rule for its own summand.
        //
        // The axis is built this way even ABOVE the crossover, where fe_rule IS the Gaussian rule
        // and the concatenation is just that rule twice. That looks wasteful and is very nearly
        // free -- each node evaluates half a kernel, so the arithmetic is the same as one pass of
        // the full body -- and it buys something worth more: the integrator never has to call
        // KERNEL::kernel, so only TWO bodies are ever inlined into the launch instead of three.
        // On kernels already sitting at REG=255 with spills, a third inlined body is not free.
        const auto tail_n = standard.template sum_nodes<mem_space>();
        const auto tail_w = standard.template sum_weights<mem_space>();
        const auto fe_n = fe_rule->template sum_nodes<mem_space>();
        const auto fe_w = fe_rule->template sum_weights<mem_space>();

        m_n_tail = tail_n.size();
        const size_t n = m_n_tail + fe_n.size();

        if (m_split_nodes.extent(0) < n) {
          m_split_nodes = Kokkos::View<ctype *, mem_space>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "QuadratureIntegrator_fT_split_nodes"), n);
          m_split_weights = Kokkos::View<ctype *, mem_space>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "QuadratureIntegrator_fT_split_weights"), n);
        }
        const auto head = Kokkos::make_pair(size_t(0), m_n_tail);
        const auto tail = Kokkos::make_pair(m_n_tail, n);
        Kokkos::deep_copy(Kokkos::subview(m_split_nodes, head), tail_n);
        Kokkos::deep_copy(Kokkos::subview(m_split_weights, head), tail_w);
        Kokkos::deep_copy(Kokkos::subview(m_split_nodes, tail), fe_n);
        Kokkos::deep_copy(Kokkos::subview(m_split_weights, tail), fe_w);

        matsubara_nodes = Kokkos::View<const ctype *, mem_space>(m_split_nodes, Kokkos::make_pair(size_t(0), n));
        matsubara_weights = Kokkos::View<const ctype *, mem_space>(m_split_weights, Kokkos::make_pair(size_t(0), n));
      } else {
        m_n_tail = 0;
        matsubara_nodes = fe_rule->template sum_nodes<mem_space>();
        matsubara_weights = fe_rule->template sum_weights<mem_space>();
      }
      grid_size[dim - 1] = matsubara_nodes.size();
    }

    /// Grow-only scratch in the integrator's own execution space, reused across calls.
    Kokkos::View<NT *, ExecutionSpace> device_scratch(const size_t n)
    {
      if (m_dest_device_size < n) {
        m_dest_device =
            Kokkos::View<NT *, ExecutionSpace>(Kokkos::view_alloc(space, "MapIntegrators_device_view"), n);
        m_dest_device_size = n;
      }
      return Kokkos::View<NT *, ExecutionSpace>(m_dest_device, Kokkos::make_pair(size_t(0), n));
    }

    /// Run a host-backend map now, or queue it for flush time if a deferral scope is open. See
    /// MapCompletion::record_work. Compiled out in a CUDA-less build.
    template <typename Coordinates, typename... Args>
    void run_or_queue_host(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      if constexpr (internal::has_device_backend) {
        if (MapCompletion::deferral_enabled()) {
          MapCompletion::record_work(
              [this, dest, coordinates, args...]() { this->run_host(dest, coordinates, args...); });
          return;
        }
      }
      run_host(dest, coordinates, args...);
    }

    /// The host-backend body of map_dist(). Queued jobs run one after another, so the shared
    /// `m_dest_device` scratch is written and drained before the next job touches it.
    template <typename Coordinates, typename... Args>
    void run_host(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      const size_t n = coordinates.size();
      auto dest_device_view = device_scratch(n);
      // create unmanaged host view for dest
      auto dest_view = Kokkos::View<NT *, CPU_memory, Kokkos::MemoryUnmanaged>(dest, n);

      // run the map function
      map(space, dest_device_view, coordinates, args...);

      // copy the result from device to the unmanaged host view
      Kokkos::deep_copy(space, dest_view, dest_device_view);
    }

  protected:
    /// Mutable because the const get() overloads issue work on it: which stream instance a launch
    /// goes to is not part of the integrator's logical state.
    mutable ExecutionSpace space;
    QuadratureProvider &quadrature_provider;
    device::array<device::array<ctype, sdim>, 2> grid_extents;
    device::array<ctype, sdim> grid_start;
    device::array<ctype, sdim> grid_scale;

    device::array<size_t, dim> grid_size;

    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, sdim> nodes;
    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, sdim> weights;

    ctype T, typical_E;
    /// Support boundary in frequency; zero means "unknown", which disables the exact sum.
    ctype m_freq_cutoff = 0;
    /// Nodes before the boundary of the concatenated axis of a split kernel; 0 when not split.
    size_t m_n_tail = 0;
    /// Owned concatenation [tail | finite-extent]; the two halves come from different rules, so
    /// unlike every other node list this one cannot be a view into the provider's cache.
    Kokkos::View<ctype *, typename ExecutionSpace::memory_space> m_split_nodes, m_split_weights;
    bool m_allow_exact = false;
    bool m_using_exact = false;
    /// 0 = follow the spatial order; see set_frequency_order().
    size_t m_freq_order = 0;
    ctype m_extent_margin = 1;

    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> matsubara_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> matsubara_weights;

    // Persistent view caches to avoid per-call GPU memory allocation
    mutable KokkosNDView<1 + dim, NT, ExecutionSpace> m_cache;
    mutable device::array<size_t, 1 + dim> m_cache_extents{};
    // Cached external positions for map(): one forward() per grid point instead of per thread.
    // Keyed on the coordinates' to_string() identity; flat layout [grid_point * cdim + d].
    mutable Kokkos::View<ctype *, typename ExecutionSpace::memory_space> m_positions;
    mutable std::string m_positions_key;
    mutable Kokkos::View<NT *, ExecutionSpace> m_dest_device;
    mutable size_t m_dest_device_size = 0;
    /// Page-locked staging for the device path, so the result copy is genuinely asynchronous.
    mutable Kokkos::View<NT *, PinnedHost_memory> m_dest_pinned;
    mutable size_t m_dest_pinned_size = 0;
    mutable Kokkos::View<NT, typename ExecutionSpace::memory_space> m_result_view;
    mutable typename Kokkos::View<NT, typename ExecutionSpace::memory_space>::host_mirror_type m_result_host;
    mutable bool m_result_views_initialized = false;
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
      // Nodes before the boundary of the concatenated axis; 0 for an unsplit kernel, where the
      // is_tail flag is dead code that the `if constexpr` in node_value() removes anyway.
      const size_t n_tail = m_n_tail;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      auto functor = [&](const device::array<size_t, dim> &idx) {
        device::array<ctype, sdim> x;
        ctype weight = 1;
        for (int i = 0; i < sdim; ++i) {
          x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
          weight *= w[i][idx[i]] * scale[i];
        }
        const size_t jt = idx[dim - 1];
        // Empty position pack: this overload's caller passes the external position inside `args`.
        return weight * Base::node_value(x, device::tuple<>{}, args, m_n[jt], m_w[jt], jt < n_tail);
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

      if (scheduler.active() && scheduler.plan_contains(this->integrator_id())) MapCompletion::flush();

      const MapSlice slice = scheduler.schedule(this->integrator_id(), dest, sizeof(NT), coordinates.size(),
                                                Base::quadrature_volume(), /* splittable */ true,
                                                map_target<execution_space>());

      if (slice.count == 0) {
        if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
        return space;
      }
      if (slice.owns_all(coordinates.size()))
        run_or_queue(dest, coordinates, args...);
      else
        run_or_queue(dest + slice.offset, SubCoordinates(coordinates, slice.offset, slice.count), args...);

      if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
      return space;
    }

  private:
    /// Run now, or queue for flush time inside a deferral scope -- see MapCompletion::record_work.
    /// Compiled out in a CUDA-less build.
    template <typename Coordinates, typename... Args>
    void run_or_queue(NT *dest, const Coordinates &coordinates, const Args &...args)
    {
      if constexpr (internal::has_device_backend) {
        if (MapCompletion::deferral_enabled()) {
          MapCompletion::record_work([this, dest, coordinates, args...]() {
            auto sp = execution_space();
            this->map(sp, dest, coordinates, args...);
          });
          return;
        }
      }
      auto sp = execution_space();
      map(sp, dest, coordinates, args...);
    }

  protected:
    using Base::grid_extents;
    using Base::grid_scale;
    using Base::grid_size;
    using Base::grid_start;
    using Base::quadrature_provider;

    using Base::m_n_tail;
    using Base::matsubara_nodes;
    using Base::matsubara_weights;
    using Base::nodes;
    using Base::weights;

    using Base::T;
    using Base::typical_E;
  };

} // namespace DiFfRG
