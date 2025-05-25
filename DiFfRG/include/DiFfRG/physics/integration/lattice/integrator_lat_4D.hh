#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  template <typename NT, typename KERNEL, typename ExecutionSpace> class IntegratorLat4D
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    IntegratorLat4D(const std::array<uint, 2> grid_size, const std::array<ctype, 2> a, bool q0_symmetric = false)
        : grid_size(grid_size), a(a), q0_symmetric(q0_symmetric)
    {
      if (grid_size[0] % 2 != 0) throw std::runtime_error("IntegratorLat4D: Grid size must be even");
      if (grid_size[1] % 2 != 0) throw std::runtime_error("IntegratorLat4D: Grid size must be even");
    }

    void set_a(const std::array<ctype, 2> a) { this->a = a; }
    void set_q0_symmetric(bool symmetric) { this->q0_symmetric = symmetric; }

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

    template <typename OT, typename... T>
      requires(!std::is_same_v<OT, NT>)
    void get(ExecutionSpace &space, OT &dest, const T &...t) const
    {
      const auto args = std::make_tuple(t...);

      const ctype fac = powr<-1>(a[0] * (ctype)grid_size[0]) * powr<-3>(a[1] * (ctype)grid_size[1]);

      const uint q0_mult = (q0_symmetric ? 2 : 1);

      const ctype x0fac = 2 * M_PI / (ctype)grid_size[0] / a[0];
      const ctype x1fac = 2 * M_PI / (ctype)grid_size[1] / a[1];
      const auto x0size = grid_size[0] / q0_mult;
      const auto &x1size = grid_size[1];

      Kokkos::parallel_reduce(
          "integral_lat_4D", // name of the kernel
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<4>>(space, {0, 0, 0, 0}, {x0size, x1size / 2, x1size / 2, x1size / 2}),
          KOKKOS_LAMBDA(const uint idx_x0, const uint idx_x1, const uint idx_x2, const uint idx_x3, NT &update) {
            const ctype q0 = x0fac * idx_x0;
            const ctype q1 = x1fac * idx_x1;
            const ctype q2 = x1fac * idx_x2;
            const ctype q3 = x1fac * idx_x3;
            const NT result =
                std::apply([&](const auto &...args) { return KERNEL::kernel(q0, q1, q2, q3, args...); }, args);
            update += q0_mult * 8 * fac * result;
          },
          SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
    }

    template <typename view_type, typename Coordinates, typename... Args>
    void map(ExecutionSpace &space, const view_type integral_view, const Coordinates &coordinates, const Args &...args)
    {
      const auto m_args = std::make_tuple(args...);

      using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
      using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;

      const ctype fac = powr<-1>(a[0] * (ctype)grid_size[0]) * powr<-3>(a[1] * (ctype)grid_size[1]);

      const uint q0_mult = (q0_symmetric ? 2 : 1);

      const ctype x0fac = 2 * M_PI / (ctype)grid_size[0] / a[0];
      const ctype x1fac = 2 * M_PI / (ctype)grid_size[1] / a[1];
      const auto x0size = grid_size[0] / q0_mult;
      const auto &x1size = grid_size[1];

      Kokkos::parallel_for(
          Kokkos::TeamPolicy(space, integral_view.size(), Kokkos::AUTO), KOKKOS_LAMBDA(const TeamType &team) {
            // get the current (continuous) index
            const uint k = team.league_rank();
            // make subview
            auto subview = Kokkos::subview(integral_view, k);
            // get the position for the current index
            const auto idx = coordinates.from_continuous_index(k);
            const auto pos = coordinates.forward(idx);

            const auto full_args = std::tuple_cat(pos, m_args);

            NT res = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadMDRange(team, x0size, x1size / 2, x1size / 2, x1size / 2), // range of the kernel
                [&](const uint idx_x0, const uint idx_x1, const uint idx_x2, const uint idx_x3, NT &update) {
                  const ctype q0 = x0fac * idx_x0;
                  const ctype q1 = x1fac * idx_x1;
                  const ctype q2 = x1fac * idx_x2;
                  const ctype q3 = x1fac * idx_x3;
                  const NT result =
                      std::apply([&](const auto &...iargs) { return KERNEL::kernel(q0, q1, q2, q3, iargs...); }, full_args);
                  update += q0_mult * 8 * fac * result;
                },
                res);

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

      return std::move(space);
    }

    const std::array<uint, 2> grid_size;

  private:
    std::array<ctype, 2> a;
    bool q0_symmetric;
  };
} // namespace DiFfRG
