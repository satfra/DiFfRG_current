#define FLOW_CODE

#include "VlowT.hh"
#include "VlowT.kernel"

namespace DiFfRG
{
  namespace Flows
  {
    VlowT_integrator::VlowT_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 1> grid_sizes, const double x_extent, const JSONValue &json)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent),
          jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
    {
      integrator = std::make_unique<DiFfRG::IntegratorTBB<3, double, VlowT_kernel<__REGULATOR__>>>(quadrature_provider, grid_sizes, x_extent, json);
      for (uint i = 0; i < 1; ++i)
        jac_grid_sizes[i] = uint(jacobian_quadrature_factor * grid_sizes[i]);
      integrator_AD = std::make_unique<DiFfRG::IntegratorTBB<3, autodiff::real, VlowT_kernel<__REGULATOR__>>>(quadrature_provider, jac_grid_sizes, x_extent, json);
    }

    VlowT_integrator::VlowT_integrator(const VlowT_integrator &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes), jac_grid_sizes(other.jac_grid_sizes), x_extent(other.x_extent),
          jacobian_quadrature_factor(other.jacobian_quadrature_factor), json(other.json),
          integrator(std::make_unique<DiFfRG::IntegratorTBB<3, double, VlowT_kernel<__REGULATOR__>>>(other.quadrature_provider, other.grid_sizes, other.x_extent, other.json)),
          integrator_AD(
              std::make_unique<DiFfRG::IntegratorTBB<3, autodiff::real, VlowT_kernel<__REGULATOR__>>>(other.quadrature_provider, other.jac_grid_sizes, other.x_extent, other.json))
    {
    }

    VlowT_integrator::~VlowT_integrator() = default;

    std::future<double> VlowT_integrator::request_CT(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1,
                                                     const double etaQ, const double etaPhi, const double etaQ0, const double etaPhi0, const double detaPhi0, const double hPhi,
                                                     const double d1hPhi, const double d2V, const double d3V, const double rhoPhi, const double m2Pion, const double m2Sigma)
    {
      return integrator->request(k, p0f, p, T, muq, A0, gAqbq1, etaQ, etaPhi, etaQ0, etaPhi0, detaPhi0, hPhi, d1hPhi, d2V, d3V, rhoPhi, m2Pion, m2Sigma);
    }

    double VlowT_integrator::get_CT(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1, const double etaQ,
                                    const double etaPhi, const double etaQ0, const double etaPhi0, const double detaPhi0, const double hPhi, const double d1hPhi, const double d2V,
                                    const double d3V, const double rhoPhi, const double m2Pion, const double m2Sigma)
    {
      return integrator->get(k, p0f, p, T, muq, A0, gAqbq1, etaQ, etaPhi, etaQ0, etaPhi0, detaPhi0, hPhi, d1hPhi, d2V, d3V, rhoPhi, m2Pion, m2Sigma);
    }

    std::future<autodiff::real> VlowT_integrator::request_AD(const double k, const double p0f, const double p, const double T, const double muq, const double A0,
                                                             const double gAqbq1, const autodiff::real etaQ, const autodiff::real etaPhi, const autodiff::real etaQ0,
                                                             const autodiff::real etaPhi0, const autodiff::real detaPhi0, const autodiff::real hPhi, const autodiff::real d1hPhi,
                                                             const autodiff::real d2V, const autodiff::real d3V, const double rhoPhi, const autodiff::real m2Pion,
                                                             const autodiff::real m2Sigma)
    {
      return integrator_AD->request(k, p0f, p, T, muq, A0, gAqbq1, etaQ, etaPhi, etaQ0, etaPhi0, detaPhi0, hPhi, d1hPhi, d2V, d3V, rhoPhi, m2Pion, m2Sigma);
    }

    autodiff::real VlowT_integrator::get_AD(const double k, const double p0f, const double p, const double T, const double muq, const double A0, const double gAqbq1,
                                            const autodiff::real etaQ, const autodiff::real etaPhi, const autodiff::real etaQ0, const autodiff::real etaPhi0,
                                            const autodiff::real detaPhi0, const autodiff::real hPhi, const autodiff::real d1hPhi, const autodiff::real d2V,
                                            const autodiff::real d3V, const double rhoPhi, const autodiff::real m2Pion, const autodiff::real m2Sigma)
    {
      return integrator_AD->get(k, p0f, p, T, muq, A0, gAqbq1, etaQ, etaPhi, etaQ0, etaPhi0, detaPhi0, hPhi, d1hPhi, d2V, d3V, rhoPhi, m2Pion, m2Sigma);
    }
  } // namespace Flows
} // namespace DiFfRG