#pragma once

template <typename REG> class VPolyakov_kernel;
#include "../def.hh"

#include <future>
#include <memory>

namespace DiFfRG
{
  namespace Flows
  {
    class VPolyakov_integrator
    {
    public:
      VPolyakov_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 1> grid_sizes, const double x_extent, const JSONValue &json);
      VPolyakov_integrator(const VPolyakov_integrator &other);
      ~VPolyakov_integrator();

      template <typename NT, typename... T> std::future<NT> request(T &&...t)
      {
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>, "Unknown type requested of VPolyakov_integrator::request");
        if constexpr (std::is_same_v<NT, double>)
          return request_CT(std::forward<T>(t)...);
        else if constexpr (std::is_same_v<NT, autodiff::real>)
          return request_AD(std::forward<T>(t)...);
      }

      template <typename NT, typename... T> NT get(T &&...t)
      {
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, autodiff::real>, "Unknown type requested of VPolyakov_integrator::request");
        if constexpr (std::is_same_v<NT, double>)
          return get_CT(std::forward<T>(t)...);
        else if constexpr (std::is_same_v<NT, autodiff::real>)
          return get_AD(std::forward<T>(t)...);
      }

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
      const std::array<uint, 1> grid_sizes;
      std::array<uint, 1> jac_grid_sizes;
      const double x_extent;
      const double jacobian_quadrature_factor;
      const JSONValue json;

      std::unique_ptr<DiFfRG::IntegratorTBB<3, double, VPolyakov_kernel<__REGULATOR__>>> integrator;
      std::unique_ptr<DiFfRG::IntegratorTBB<3, autodiff::real, VPolyakov_kernel<__REGULATOR__>>> integrator_AD;
    };
  } // namespace Flows
} // namespace DiFfRG