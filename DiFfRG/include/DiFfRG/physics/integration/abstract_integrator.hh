#pragma once

// DiFfRG
#include "distribution.hh"
#include <DiFfRG/physics/integration/distribution.hh>

namespace DiFfRG
{
  namespace internal
  {
    template <int dim>
    std::array<size_t, dim> make_int_grid(const JSONValue &json, const std::array<std::string, dim> &names)
    {
      std::array<size_t, dim> int_grid;
      for (int i = 0; i < dim; ++i)
        int_grid[i] = json.get_uint("/integration/" + names[i]);
      return int_grid;
    }
  } // namespace internal

  template <typename KERNEL>
  concept provides_regulator = requires { typename KERNEL::Regulator; };

  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  NT multidim_kernel_call(const ARGS &...args)
  {
    if constexpr (dim == 0)
      return KERNEL::kernel(args...);
    else {
      const ctype darg{};
      return multidim_kernel_call<NT, KERNEL, ctype, dim - 1, ARGS...>(darg, args...);
    }
  }

  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  concept provides_kernel =
      requires(const ARGS &...args) { multidim_kernel_call<NT, KERNEL, ctype, dim, ARGS...>(args...); };

  template <typename NT, typename KERNEL, typename... ARGS>
  concept provides_constant = requires(const ARGS &...args) {
    { KERNEL::constant(args...) } -> std::convertible_to<NT>;
  };

  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  concept is_valid_kernel =
      (provides_kernel<NT, KERNEL, ctype, dim, ARGS...> && provides_constant<NT, KERNEL, ARGS...>);

  class AbstractIntegrator
  {
  public:
    void set_node_distribution(const NodeDistribution &distribution);

    const NodeDistribution &get_node_distribution() const;

    void set_load_balancer(IntegrationLoadBalancer &load_balancer);

  protected:
    NodeDistribution node_distribution;
  };
} // namespace DiFfRG