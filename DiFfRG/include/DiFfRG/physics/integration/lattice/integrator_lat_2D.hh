#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  template <typename NT, typename KERNEL, typename ExecutionSpace> class IntegratorLat2D
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    IntegratorLat2D(const std::array<uint, 1> grid_size, const std::array<ctype, 1> a) : grid_size(grid_size), a(a) {}

    void set_a(const std::array<ctype, 1> a) { this->a = a; }

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

      const ctype fac = powr<-1>(a[0] * (ctype)grid_size[0] * a[1] * (ctype)grid_size[1]);

      const ctype &xsize = grid_size[0];
      const ctype xfac = 2 * M_PI / (ctype)grid_size[0] / a[0];

      const ctype &ysize = grid_size[1];
      const ctype yfac = 2 * M_PI / (ctype)grid_size[1] / a[1];

      Kokkos::parallel_reduce(
          "integral_lat_2D",                                                                     // name of the kernel
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(space, {0, 0}, {xsize, ysize}), // range of the kernel
          KOKKOS_LAMBDA(const uint idx_x, const uint idx_y, NT &update) {
            const ctype q1 = xfac * idx_x;
            const ctype q2 = yfac * idx_y;
            const NT result = std::apply([&](const auto &...args) { return KERNEL::kernel(q1, a2, args...); }, args);
            update += fac * result;
          },
          SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
    }

    template <typename view_type, typename Coordinates, typename... Args>
    void map(ExecutionSpace &space, const view_type integral_view, const Coordinates &coordinates, const Args &...args)
    {
      const auto m_args = std::make_tuple(args...);

      using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
      using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;

      const ctype fac = powr<-1>(a[0] * (ctype)grid_size[0] * a[1] * (ctype)grid_size[1]);

      const ctype &xsize = grid_size[0];
      const ctype xfac = 2 * M_PI / (ctype)grid_size[0] / a[0];

      const ctype &ysize = grid_size[1];
      const ctype yfac = 2 * M_PI / (ctype)grid_size[1] / a[1];

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

            // compute the constant value
            // TODO: do this only once...
            const auto constant =
                std::apply([&](const auto &...iargs) { return KERNEL::constant(iargs...); }, full_args);

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, 0, xsize), // range of the kernel
                [&](const uint idx_x, NT &update) {
                  const ctype q1 = xfac * idx_x;
                  const NT result =
                      std::apply([&](const auto &...iargs) { return KERNEL::kernel(q1, iargs...); }, full_args);
                  update += fac * result;
                },
                SumPlus<NT, NT, ExecutionSpace>(subview, constant));
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

    const std::array<uint, 1> grid_size;

  private:
    std::array<ctype, 1> a;
  };
} // namespace DiFfRG
