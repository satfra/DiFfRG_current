#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  template <typename NT, typename KERNEL, typename ExecutionSpace> class Integrator1D
  {
  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    using execution_space = ExecutionSpace;

    Integrator1D(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                 const std::array<std::array<double, 2>, 1> extents)
        : quadrature_provider(quadrature_provider), grid_size(grid_size), grid_extents(extents)
    {
      x_nodes = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[0]);
      x_weights = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(grid_size[0]);
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

      const auto &x_n = x_nodes;
      const auto &x_w = x_weights;
      const auto &x_ext = grid_extents[0][0];
      const auto x_scale = (grid_extents[0][1] - grid_extents[0][0]);

      Kokkos::View<NT, typename ExecutionSpace::memory_space> result("result");

      Kokkos::parallel_reduce(
          "integral_1D",                                                                 // name of the kernel
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<uint>>(0, grid_size[0]), // range of the kernel
          KOKKOS_LAMBDA(const uint idx_x, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_ext);
            const ctype weight = x_w[idx_x] * x_scale;
            const NT result = std::apply([&](const auto &...args) { return KERNEL::kernel(x, args...); }, args);
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
      const auto &x_ext = grid_extents[0][0];
      const auto x_scale = (grid_extents[0][1] - grid_extents[0][0]);

      Kokkos::parallel_reduce(
          "integral_1D",                                                                 // name of the kernel
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<uint>>(0, grid_size[0]), // range of the kernel
          KOKKOS_LAMBDA(const uint idx_x, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_ext);
            const ctype weight = x_w[idx_x] * x_scale;
            const NT result = std::apply([&](const auto &...args) { return KERNEL::kernel(x, args...); }, args);
            update += weight * result;
          },
          SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
    }

    template <typename output_type, typename team_type, typename... T>
    void KOKKOS_INLINE_FUNCTION get_nested(output_type &dest, const team_type &team, const T &...t) const
    {
      const auto &x_n = x_nodes;
      const auto &x_w = x_weights;
      const auto &x_ext = grid_extents[0][0];
      const auto x_scale = (grid_extents[0][1] - grid_extents[0][0]);

      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, 0, grid_size[0]), // range of the kernel
          [=](const uint idx_x, NT &update) {
            const ctype x = Kokkos::fma(x_scale, x_n[idx_x], x_ext);
            const ctype weight = x_w[idx_x] * x_scale;
            const NT result = KERNEL::kernel(x, t...);
            update += weight * result;
          },
          SumPlus<NT, NT, ExecutionSpace>(dest, KERNEL::constant(t...)));
    }

    const std::array<uint, 1> grid_size;
    const std::array<std::array<double, 2>, 1> grid_extents;

  private:
    QuadratureProvider &quadrature_provider;

    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> x_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> x_weights;
  };
} // namespace DiFfRG