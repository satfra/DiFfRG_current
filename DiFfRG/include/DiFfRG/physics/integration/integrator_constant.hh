#pragma once

// standard library
#include <future>

// DiFfRG
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  template <int d, typename NT, typename KERNEL> class IntegratorConstant
  {
  public:
    using ctype = typename get_type::ctype<NT>;
    IntegratorConstant(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                       const ctype x_extent, const uint max_block_size = 0)
        : IntegratorConstant(quadrature_provider, grid_size[0], x_extent, max_block_size)
    {
    }

    IntegratorConstant(QuadratureProvider &quadrature_provider, const std::array<uint, 1> grid_size,
                       const ctype x_extent, const JSONValue &)
        : IntegratorConstant(quadrature_provider, grid_size[0], x_extent)
    {
    }

    IntegratorConstant(QuadratureProvider &quadrature_provider, const uint grid_size, const ctype x_extent,
                       const uint max_block_size = 0)
        : IntegratorConstant()
    {
      (void)quadrature_provider;
      (void)grid_size;
      (void)x_extent;
      (void)max_block_size;
    }

    IntegratorConstant() {}

    template <typename... T> NT get(const ctype k, const T &...t) const
    {
      const ctype S_d = 2. * std::pow(M_PI, d / 2.) / std::tgammal(d / 2.);

      return KERNEL::constant(k, t...) + KERNEL::kernel(k, t...) * S_d / powr<d>(2. * M_PI);
    }

    template <typename... T> std::future<NT> request(const ctype k, const T &...t) const
    {
      return std::async(std::launch::deferred, [=, this]() { return get(k, t...); });
    }
  };
} // namespace DiFfRG