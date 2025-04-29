#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  template <typename NT, typename KERNEL, typename ExecutionSpace> class Integrator4D
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    Integrator4D(QuadratureProvider &quadrature_provider, const std::array<uint, 4> grid_size,
                 std::array<double, 4> grid_min, std::array<double, 4> grid_max,
                 const std::array<QuadratureType, 4> quadrature_type = {QuadratureType::legendre,
                                                                        QuadratureType::legendre,
                                                                        QuadratureType::legendre,
                                                                        QuadratureType::legendre})
        : quadrature_provider(quadrature_provider), grid_size(grid_size), grid_extents{grid_min, grid_max}
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

      w_nodes = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[3],
                                                                                                 quadrature_type[3]);
      w_weights = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
          grid_size[3], quadrature_type[3]);
    }

    void set_grid_extents(const std::array<ctype, 4> grid_min, const std::array<ctype, 4> grid_max)
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
      const auto &w_n = w_nodes;
      const auto &w_w = w_weights;

      const auto &x_start = grid_extents[0][0];
      const auto &y_start = grid_extents[0][1];
      const auto &z_start = grid_extents[0][2];
      const auto &w_start = grid_extents[0][3];

      const auto x_scale = (grid_extents[1][0] - grid_extents[0][0]);
      const auto y_scale = (grid_extents[1][1] - grid_extents[0][1]);
      const auto z_scale = (grid_extents[1][2] - grid_extents[0][2]);
      const auto w_scale = (grid_extents[1][3] - grid_extents[0][3]);

      Kokkos::parallel_reduce(
          "integral_4D", // name of the kernel
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<4>>(
              {0, 0, 0, 0}, {grid_size[0], grid_size[1], grid_size[2], grid_size[3]}), // range of the kernel
          KOKKOS_LAMBDA(const uint idx_x, const uint idx_y, const uint idx_z, const uint idx_w, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_start);
            const ctype y = Kokkos::fma(y_scale, y_n[idx_y], y_start);
            const ctype z = Kokkos::fma(z_scale, z_n[idx_z], z_start);
            const ctype w = Kokkos::fma(w_scale, w_n[idx_w], w_start);
            const ctype weight =
                x_w[idx_x] * x_scale * y_w[idx_y] * y_scale * z_w[idx_z] * z_scale * w_w[idx_w] * w_scale;
            const NT result = std::apply([&](const auto &...t) { return KERNEL::kernel(x, y, z, w, t...); }, args);
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
      const auto &w_n = w_nodes;
      const auto &w_w = w_weights;

      const auto &x_start = grid_extents[0][0];
      const auto &y_start = grid_extents[0][1];
      const auto &z_start = grid_extents[0][2];
      const auto &w_start = grid_extents[0][3];

      const auto x_scale = (grid_extents[1][0] - grid_extents[0][0]);
      const auto y_scale = (grid_extents[1][1] - grid_extents[0][1]);
      const auto z_scale = (grid_extents[1][2] - grid_extents[0][2]);
      const auto w_scale = (grid_extents[1][3] - grid_extents[0][3]);

      Kokkos::parallel_reduce(
          "integral_4D", // name of the kernel
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<4>>(
              {0, 0, 0, 0}, {grid_size[0], grid_size[1], grid_size[2], grid_size[3]}), // range of the kernel
          KOKKOS_LAMBDA(const uint idx_x, const uint idx_y, const uint idx_z, const uint idx_w, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_start);
            const ctype y = Kokkos::fma(y_scale, y_n[idx_y], y_start);
            const ctype z = Kokkos::fma(z_scale, z_n[idx_z], z_start);
            const ctype w = Kokkos::fma(w_scale, w_n[idx_w], w_start);
            const ctype weight =
                x_w[idx_x] * x_scale * y_w[idx_y] * y_scale * z_w[idx_z] * z_scale * w_w[idx_w] * w_scale;
            const NT result = std::apply([&](const auto &...t) { return KERNEL::kernel(x, y, z, w, t...); }, args);
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
      const auto &w_n = w_nodes;
      const auto &w_w = w_weights;

      const auto &x_start = grid_extents[0][0];
      const auto &y_start = grid_extents[0][1];
      const auto &z_start = grid_extents[0][2];
      const auto &w_start = grid_extents[0][3];

      const auto x_scale = (grid_extents[1][0] - grid_extents[0][0]);
      const auto y_scale = (grid_extents[1][1] - grid_extents[0][1]);
      const auto z_scale = (grid_extents[1][2] - grid_extents[0][2]);
      const auto w_scale = (grid_extents[1][3] - grid_extents[0][3]);

      Kokkos::parallel_reduce(
          Kokkos::TeamThreadMDRange(team, {0, 0, 0, 0},
                                    {grid_size[0], grid_size[1], grid_size[2], grid_size[3]}), // range of the kernel
          [=](const uint idx_x, const uint idx_y, const uint idx_z, const uint idx_w, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_start);
            const ctype y = Kokkos::fma(y_scale, y_n[idx_y], y_start);
            const ctype z = Kokkos::fma(z_scale, z_n[idx_z], z_start);
            const ctype w = Kokkos::fma(w_scale, w_n[idx_w], w_start);
            const ctype weight =
                x_w[idx_x] * x_scale * y_w[idx_y] * y_scale * z_w[idx_z] * z_scale * w_w[idx_w] * w_scale;
            const NT result = KERNEL::kernel(x, y, z, w, t...);
            update += weight * result;
          },
          SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
    }

    const std::array<uint, 4> grid_size;
    const std::array<std::array<double, 4>, 2> grid_extents;

  private:
    QuadratureProvider &quadrature_provider;

    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> x_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> x_weights;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> y_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> y_weights;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> z_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> z_weights;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> w_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> w_weights;
  };
} // namespace DiFfRG