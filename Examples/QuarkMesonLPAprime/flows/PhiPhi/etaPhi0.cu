#define FLOW_CODE

#include "etaPhi0.hh"
#include "etaPhi0.kernel"

namespace DiFfRG
{
  namespace Flows
  {
    etaPhi0_integrator::etaPhi0_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 2> grid_sizes, const double x_extent, const double q0_extent,
                                           const uint q0_summands, const JSONValue &json)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent), q0_extent(q0_extent), q0_summands(q0_summands),
          jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
    {
      integrator =
          std::make_unique<DiFfRG::IntegratorFiniteTq0TBB<4, double, etaPhi0_kernel<__REGULATOR__>>>(quadrature_provider, grid_sizes, x_extent, q0_extent, q0_summands, json);
      for (uint i = 0; i < 2; ++i)
        jac_grid_sizes[i] = uint(jacobian_quadrature_factor * grid_sizes[i]);
      integrator_AD = std::make_unique<DiFfRG::IntegratorFiniteTq0TBB<4, autodiff::real, etaPhi0_kernel<__REGULATOR__>>>(quadrature_provider, jac_grid_sizes, x_extent, q0_extent,
                                                                                                                         q0_summands, json);
    }

    etaPhi0_integrator::etaPhi0_integrator(const etaPhi0_integrator &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes), jac_grid_sizes(other.jac_grid_sizes), x_extent(other.x_extent), q0_extent(other.q0_extent),
          q0_summands(other.q0_summands), jacobian_quadrature_factor(other.jacobian_quadrature_factor), json(other.json),
          integrator(std::make_unique<DiFfRG::IntegratorFiniteTq0TBB<4, double, etaPhi0_kernel<__REGULATOR__>>>(other.quadrature_provider, other.grid_sizes, other.x_extent,
                                                                                                                other.q0_extent, other.q0_summands, other.json)),
          integrator_AD(std::make_unique<DiFfRG::IntegratorFiniteTq0TBB<4, autodiff::real, etaPhi0_kernel<__REGULATOR__>>>(
              other.quadrature_provider, other.jac_grid_sizes, other.x_extent, other.q0_extent, other.q0_summands, other.json))
    {
    }

    etaPhi0_integrator::~etaPhi0_integrator() = default;

    void etaPhi0_integrator::set_T(const double value)
    {
      integrator->set_T(value);
      integrator_AD->set_T(value);
    }
    void etaPhi0_integrator::set_q0_extent(const double value)
    {
      integrator->set_q0_extent(value);
      integrator_AD->set_q0_extent(value);
    }

    std::future<double> etaPhi0_integrator::request_CT(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1,
                                                       const double etaQ, const double etaPhi, const double etaQ0, const double etaPhi0, const double detaPhi0, const double hPhi,
                                                       const double d1hPhi, const double d2V, const double d3V, const double rhoPhi, const double m2Pion, const double m2Sigma)
    {
      return integrator->request(k, p0f, p, T, muq, A0, gAqbq1, etaQ, etaPhi, etaQ0, etaPhi0, detaPhi0, hPhi, d1hPhi, d2V, d3V, rhoPhi, m2Pion, m2Sigma);
    }

    double etaPhi0_integrator::get_CT(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1, const double etaQ,
                                      const double etaPhi, const double etaQ0, const double etaPhi0, const double detaPhi0, const double hPhi, const double d1hPhi,
                                      const double d2V, const double d3V, const double rhoPhi, const double m2Pion, const double m2Sigma)
    {
      return integrator->get(k, p0f, p, T, muq, A0, gAqbq1, etaQ, etaPhi, etaQ0, etaPhi0, detaPhi0, hPhi, d1hPhi, d2V, d3V, rhoPhi, m2Pion, m2Sigma);
    }

    std::future<autodiff::real> etaPhi0_integrator::request_AD(const double k, const double p0f, const double p, const double T, const double muq, const double A0,
                                                               const double gAqbq1, const autodiff::real etaQ, const autodiff::real etaPhi, const autodiff::real etaQ0,
                                                               const autodiff::real etaPhi0, const autodiff::real detaPhi0, const autodiff::real hPhi, const autodiff::real d1hPhi,
                                                               const autodiff::real d2V, const autodiff::real d3V, const double rhoPhi, const autodiff::real m2Pion,
                                                               const autodiff::real m2Sigma)
    {
      return integrator_AD->request(k, p0f, p, T, muq, A0, gAqbq1, etaQ, etaPhi, etaQ0, etaPhi0, detaPhi0, hPhi, d1hPhi, d2V, d3V, rhoPhi, m2Pion, m2Sigma);
    }

    autodiff::real etaPhi0_integrator::get_AD(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1,
                                              const autodiff::real etaQ, const autodiff::real etaPhi, const autodiff::real etaQ0, const autodiff::real etaPhi0,
                                              const autodiff::real detaPhi0, const autodiff::real hPhi, const autodiff::real d1hPhi, const autodiff::real d2V,
                                              const autodiff::real d3V, const double rhoPhi, const autodiff::real m2Pion, const autodiff::real m2Sigma)
    {
      return integrator_AD->get(k, p0f, p, T, muq, A0, gAqbq1, etaQ, etaPhi, etaQ0, etaPhi0, detaPhi0, hPhi, d1hPhi, d2V, d3V, rhoPhi, m2Pion, m2Sigma);
    }
  } // namespace Flows
} // namespace DiFfRG