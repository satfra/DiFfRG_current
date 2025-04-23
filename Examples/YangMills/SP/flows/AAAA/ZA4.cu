#define FLOW_CODE

#include "ZA4.hh"
#include "ZA4.kernel"

namespace DiFfRG
{
  namespace Flows
  {
    ZA4_integrator::ZA4_integrator(QuadratureProvider &quadrature_provider, std::array<uint, 4> grid_sizes,
                                   const double x_extent, const JSONValue &json)
        : quadrature_provider(quadrature_provider), grid_sizes(grid_sizes), x_extent(x_extent),
          jacobian_quadrature_factor(json.get_double("/integration/jacobian_quadrature_factor")), json(json)
    {
      integrator = std::make_unique<DiFfRG::Integrator4DGPU<double, ZA4_kernel<__REGULATOR__>>>(
          quadrature_provider, grid_sizes, x_extent, json);
    }

    ZA4_integrator::ZA4_integrator(const ZA4_integrator &other)
        : quadrature_provider(other.quadrature_provider), grid_sizes(other.grid_sizes),
          jac_grid_sizes(other.jac_grid_sizes), x_extent(other.x_extent),
          jacobian_quadrature_factor(other.jacobian_quadrature_factor), json(other.json),
          integrator(std::make_unique<DiFfRG::Integrator4DGPU<double, ZA4_kernel<__REGULATOR__>>>(
              other.quadrature_provider, other.grid_sizes, other.x_extent, other.json))
    {
    }

    ZA4_integrator::~ZA4_integrator() = default;

    std::future<double> ZA4_integrator::request_CT(
        const double k, const double p, const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA3,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZAcbc,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA,
        const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA, const double m2A)
    {
      return integrator->request(k, p, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA, m2A);
    }

    double ZA4_integrator::get_CT(const double k, const double p,
                                  const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA3,
                                  const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZAcbc,
                                  const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA4,
                                  const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZc,
                                  const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &Zc,
                                  const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &dtZA,
                                  const TexLinearInterpolator1D<double, LogarithmicCoordinates1D<float>> &ZA,
                                  const double m2A)
    {
      return integrator->get(k, p, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA, m2A);
    }

  } // namespace Flows
} // namespace DiFfRG