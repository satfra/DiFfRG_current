#pragma once

template <typename REG> class etaPhi0_kernel;
#include "../def.hh"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    class etaPhi0_integrator
    {
    public:
      etaPhi0_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes, const double x_extent, const double q0_extent, const uint q0_summands,
                         const JSONValue &json);
      etaPhi0_integrator(const etaPhi0_integrator &other);
      ~etaPhi0_integrator();

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>, "Unknown type requested of etaPhi0_integrator::request");
        if constexpr (std::is_same_v<NT, double>)
          return request_CT(std::forward<T>(t)...);
        else if constexpr (std::is_same_v<NT, autodiff::real>)
          return request_AD(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>, "Unknown type requested of etaPhi0_integrator::request");
        if constexpr (std::is_same_v<NT, double>)
          return get_CT(std::forward<T>(t)...);
        else if constexpr (std::is_same_v<NT, autodiff::real>)
          return get_AD(std::forward<T>(t)...);
      }

      void set_T(const double value);
      void set_q0_extent(const double value);

    private:
      std::future<double> request_CT(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1, const double etaQ,
                                     const double etaPhi, const double etaQ0, const double etaPhi0, const double detaPhi0, const double hPhi, const double d1hPhi, const double d2V,
                                     const double d3V, const double rhoPhi, const double m2Pion, const double m2Sigma);
      double get_CT(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1, const double etaQ,
                    const double etaPhi, const double etaQ0, const double etaPhi0, const double detaPhi0, const double hPhi, const double d1hPhi, const double d2V,
                    const double d3V, const double rhoPhi, const double m2Pion, const double m2Sigma);
      std::future<autodiff::real> request_AD(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1,
                                             const autodiff::real etaQ, const autodiff::real etaPhi, const autodiff::real etaQ0, const autodiff::real etaPhi0,
                                             const autodiff::real detaPhi0, const autodiff::real hPhi, const autodiff::real d1hPhi, const autodiff::real d2V,
                                             const autodiff::real d3V, const double rhoPhi, const autodiff::real m2Pion, const autodiff::real m2Sigma);
      autodiff::real get_AD(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1, const autodiff::real etaQ,
                            const autodiff::real etaPhi, const autodiff::real etaQ0, const autodiff::real etaPhi0, const autodiff::real detaPhi0, const autodiff::real hPhi,
                            const autodiff::real d1hPhi, const autodiff::real d2V, const autodiff::real d3V, const double rhoPhi, const autodiff::real m2Pion,
                            const autodiff::real m2Sigma);

      QuadratureProvider &quadrature_provider;
      const std::array<uint, 2> grid_sizes;
      std::array<uint, 2> jac_grid_sizes;
      const double x_extent;
      const double q0_extent;
      const uint q0_summands;
      const double jacobian_quadrature_factor;
      const JSONValue json;

      std::unique_ptr<DiFfRG::IntegratorFiniteTq0TBB<4, double, etaPhi0_kernel<__REGULATOR__>>> integrator;
      std::unique_ptr<DiFfRG::IntegratorFiniteTq0TBB<4, autodiff::real, etaPhi0_kernel<__REGULATOR__>>> integrator_AD;
    };
  } // namespace Flows
} // namespace DiFfRG