#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  template <typename NT, typename KERNEL, typename ExecutionSpace> class Integrator3D
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    Integrator3D(QuadratureProvider &quadrature_provider, const std::array<uint, 3> grid_size,
                 std::array<double, 3> extents_x, std::array<double, 3> extents_y,
                 const std::array<QuadratureType, 3> quadrature_type = {QuadratureType::legendre,
                                                                        QuadratureType::legendre,
                                                                        QuadratureType::legendre})
        : quadrature_provider(quadrature_provider), grid_size(grid_size), grid_extents{extents_x, extents_y}
    {
      x_nodes = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[0],
                                                                                                 quadrature_type[0]);
      x_weights = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
          grid_size[0], quadrature_type[0]);

      y_nodes = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[1],
                                                                                                 quadrature_type[1]);
      y_weights = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
          grid_size[1], quadrature_type[1]);

      z_nodes = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[2],
                                                                                                 quadrature_type[2]);
      z_weights = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
          grid_size[2], quadrature_type[2]);
    }

    void set_grid_extents(const std::array<ctype, 3> grid_min, const std::array<ctype, 3> grid_max)
    {
      grid_extents = {grid_min, grid_max};
    }

    /**
     * @brief Request a future for the integral of the kernel.
     *
     * @tparam T Types of the parameters for the kernel.
     * @param k RG-scale.
     * @param t Parameters forwarded to the kernel.
     *
     * @return std::future<NT> future holding the integral of the kernel plus the constant part.
     *
     */
    template <typename... T> void get(NT &dest, const T &...t) const
    {
      const auto args = std::make_tuple(t...);

      Kokkos::View<NT, typename ExecutionSpace::memory_space> result("result");

      const auto &x_n = x_nodes;
      const auto &x_w = x_weights;
      const auto &y_n = y_nodes;
      const auto &y_w = y_weights;
      const auto &z_n = z_nodes;
      const auto &z_w = z_weights;
      const auto &x_start = grid_extents[0][0];
      const auto &y_start = grid_extents[0][1];
      const auto &z_start = grid_extents[0][2];
      const auto x_scale = (grid_extents[1][0] - grid_extents[0][0]);
      const auto y_scale = (grid_extents[1][1] - grid_extents[0][1]);
      const auto z_scale = (grid_extents[1][2] - grid_extents[0][2]);

      Kokkos::parallel_reduce(
          "integral_3D", // name of the kernel
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
              {0, 0, 0}, {grid_size[0], grid_size[1], grid_size[2]}), // range of the kernel
          KOKKOS_LAMBDA(const uint idx_x, const uint idx_y, const uint idx_z, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_start);
            const ctype y = Kokkos::fma(y_scale, y_n[idx_y], y_start);
            const ctype z = Kokkos::fma(z_scale, z_n[idx_z], z_start);
            const ctype weight = x_w[idx_x] * x_scale * y_w[idx_y] * y_scale * z_w[idx_z] * z_scale;
            const NT result = std::apply([&](const auto &...args) { return KERNEL::kernel(x, y, z, args...); }, args);
            update += weight * result;
          },
          SumPlus<NT, NT, ExecutionSpace>(result, KERNEL::constant(t...)));

      Kokkos::fence();
      auto result_host = Kokkos::create_mirror_view(result);
      Kokkos::deep_copy(result_host, result);
      dest = result_host();
    }

    /**
     * @brief Request a future for the integral of the kernel.
     *
     * @tparam T Types of the parameters for the kernel.
     * @param k RG-scale.
     * @param t Parameters forwarded to the kernel.
     *
     * @return std::future<NT> future holding the integral of the kernel plus the constant part.
     *
     */
    template <typename OT, typename... T>
      requires(!std::is_same_v<OT, NT>)
    void get(OT &dest, const T &...t) const
    {
      const auto args = std::make_tuple(t...);

      const auto &x_n = x_nodes;
      const auto &x_w = x_weights;
      const auto &y_n = y_nodes;
      const auto &y_w = y_weights;
      const auto &z_n = z_nodes;
      const auto &z_w = z_weights;

      const auto &x_start = grid_extents[0][0];
      const auto &y_start = grid_extents[0][1];
      const auto &z_start = grid_extents[0][2];
      const auto x_scale = (grid_extents[1][0] - grid_extents[0][0]);
      const auto y_scale = (grid_extents[1][1] - grid_extents[0][1]);
      const auto z_scale = (grid_extents[1][2] - grid_extents[0][2]);

      Kokkos::parallel_reduce(
          "integral_3D", // name of the kernel
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
              {0, 0, 0}, {grid_size[0], grid_size[1], grid_size[2]}), // range of the kernel
          KOKKOS_LAMBDA(const uint idx_x, const uint idx_y, const uint idx_z, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_start);
            const ctype y = Kokkos::fma(y_scale, y_n[idx_y], y_start);
            const ctype z = Kokkos::fma(z_scale, z_n[idx_z], z_start);
            const ctype weight = x_w[idx_x] * x_scale * y_w[idx_y] * y_scale * z_w[idx_z] * z_scale;
            const NT result = std::apply([&](const auto &...args) { return KERNEL::kernel(x, y, z, args...); }, args);
            update += weight * result;
          },
          SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
    }

    template <typename output_type, typename team_type, typename... T>
    void KOKKOS_INLINE_FUNCTION get_nested(output_type &dest, const team_type &team, const T &...t) const
    {
      const auto &x_n = x_nodes;
      const auto &x_w = x_weights;
      const auto &y_n = y_nodes;
      const auto &y_w = y_weights;
      const auto &z_n = z_nodes;
      const auto &z_w = z_weights;

      const auto &x_start = grid_extents[0][0];
      const auto &y_start = grid_extents[0][1];
      const auto &z_start = grid_extents[0][2];
      const auto x_scale = (grid_extents[1][0] - grid_extents[0][0]);
      const auto y_scale = (grid_extents[1][1] - grid_extents[0][1]);
      const auto z_scale = (grid_extents[1][2] - grid_extents[0][2]);

      Kokkos::parallel_reduce(
          Kokkos::TeamThreadMDRange(team, {0, 0, 0}, {grid_size[0], grid_size[1], grid_size[2]}), // range of the kernel
          [=](const uint idx_x, const uint idx_y, const uint idx_z, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_start);
            const ctype y = Kokkos::fma(y_scale, y_n[idx_y], y_start);
            const ctype z = Kokkos::fma(z_scale, z_n[idx_z], z_start);
            const ctype weight = x_w[idx_x] * x_scale * y_w[idx_y] * y_scale * z_w[idx_z] * z_scale;
            const NT result = KERNEL::kernel(x, y, z, t...);
            update += weight * result;
          },
          SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
    }

    template <typename view_type, typename Coordinates, typename... Args>
    void map(ExecutionSpace &space, const view_type integral_view, const Coordinates &coordinates, const Args &...args)
    {
      const auto m_args = std::make_tuple(args...);

      using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
      using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
      using Scratch = typename ExecutionSpace::scratch_memory_space;

      const auto &x_n = x_nodes;
      const auto &x_w = x_weights;
      const auto &x_start = grid_extents[0][0];
      const auto x_scale = (grid_extents[1][0] - grid_extents[0][0]);

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
                Kokkos::TeamThreadRange(team, 0, x_n.size()), // range of the kernel
                [&](const uint idx_x, NT &update) {
                  const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_start);
                  const ctype weight = x_w[idx_x] * x_scale;
                  const NT result =
                      std::apply([&](const auto &...iargs) { return KERNEL::kernel(x, iargs...); }, full_args);
                  update += weight * result;
                },
                SumPlus<NT, NT, ExecutionSpace>(subview, constant));
          });
    }

    template <typename NT, typename Coordinates, typename... Args>
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

    const std::array<uint, 3> grid_size;

  private:
    QuadratureProvider &quadrature_provider;
    std::array<std::array<double, 3>, 2> grid_extents;

    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> x_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> x_weights;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> y_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> y_weights;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> z_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> z_weights;
  };
} // namespace DiFfRG