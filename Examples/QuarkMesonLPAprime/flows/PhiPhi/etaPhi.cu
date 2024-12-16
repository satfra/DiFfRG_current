#define FLOW_CODE

#include "etaPhi.hh"
#include "etaPhi.kernel"

namespace DiFfRG
{
  namespace Flows
  {
    etaPhi_integrator::etaPhi_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 3> grid_sizes, const double x_extent, const double q0_extent,
                                         const uint q0_summands, const JSONValue &json)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent), q0_extent(q0_extent), q0_summands(q0_summands),
          jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
    {
      integrator =
          std::make_unique<DiFfRG::IntegratorAngleFiniteTq0TBB<4, double, etaPhi_kernel<__REGULATOR__>>>(quadrature_provider, grid_sizes, x_extent, q0_extent, q0_summands, json);
      for (uint i = 0; i < 3; ++i)
        jac_grid_sizes[i] = uint(jacobian_quadrature_factor * grid_sizes[i]);
      integrator_AD = std::make_unique<DiFfRG::IntegratorAngleFiniteTq0TBB<4, autodiff::real, etaPhi_kernel<__REGULATOR__>>>(quadrature_provider, jac_grid_sizes, x_extent,
                                                                                                                             q0_extent, q0_summands, json);
    }

    etaPhi_integrator::etaPhi_integrator(const etaPhi_integrator &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes), jac_grid_sizes(other.jac_grid_sizes), x_extent(other.x_extent), q0_extent(other.q0_extent),
          q0_summands(other.q0_summands), jacobian_quadrature_factor(other.jacobian_quadrature_factor), json(other.json),
          integrator(std::make_unique<DiFfRG::IntegratorAngleFiniteTq0TBB<4, double, etaPhi_kernel<__REGULATOR__>>>(other.quadrature_provider, other.grid_sizes, other.x_extent,
                                                                                                                    other.q0_extent, other.q0_summands, other.json)),
          integrator_AD(std::make_unique<DiFfRG::IntegratorAngleFiniteTq0TBB<4, autodiff::real, etaPhi_kernel<__REGULATOR__>>>(
              other.quadrature_provider, other.jac_grid_sizes, other.x_extent, other.q0_extent, other.q0_summands, other.json))
    {
    }

    etaPhi_integrator::~etaPhi_integrator() = default;

    void etaPhi_integrator::set_T(const double value)
    {
      integrator->set_T(value);
      integrator_AD->set_T(value);
    }
    void etaPhi_integrator::set_q0_extent(const double value)
    {
      integrator->set_q0_extent(value);
      integrator_AD->set_q0_extent(value);
    }

    std::future<double> etaPhi_integrator::request_CT(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T, const double muq,
                                                      const double etaQ, const double etaPhi, const double hPhi, const double d1V, const double d2V, const double d3V,
                                                      const double rhoPhi)
    {
      return integrator->request(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }

    double etaPhi_integrator::get_CT(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T, const double muq, const double etaQ,
                                     const double etaPhi, const double hPhi, const double d1V, const double d2V, const double d3V, const double rhoPhi)
    {
      return integrator->get(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }

    std::future<autodiff::real> etaPhi_integrator::request_AD(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T, const double muq,
                                                              const autodiff::real etaQ, const autodiff::real etaPhi, const autodiff::real hPhi, const autodiff::real d1V,
                                                              const autodiff::real d2V, const autodiff::real d3V, const double rhoPhi)
    {
      return integrator_AD->request(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }

    autodiff::real etaPhi_integrator::get_AD(const double k, const double p0f, const double p, const double Nc, const double Nf, const double T, const double muq,
                                             const autodiff::real etaQ, const autodiff::real etaPhi, const autodiff::real hPhi, const autodiff::real d1V, const autodiff::real d2V,
                                             const autodiff::real d3V, const double rhoPhi)
    {
      return integrator_AD->get(k, p0f, p, Nc, Nf, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
    }
  } // namespace Flows
} // namespace DiFfRG