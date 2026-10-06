#pragma once

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows_batched/flows.hh"

#include <vector>

/**
 * The O(N) model of model.hh with a batched flux: the flux at all quadrature points of the CG assembler is
 * one map_points() call of the V integrator, on the backend chosen at compile time. Physics and parameters
 * are those of model.hh.
 */
namespace ON_batched
{
  struct Parameters {
    Parameters(const ConfigTree &config)
        : Lambda(config.get_double("/physical/Lambda")), N(config.get_double("/physical/N")), T(config.get_double("/physical/T")), lambda2(config.get_double("/physical/lambda2")),
          lambda4(config.get_double("/physical/lambda4")), lambda6(config.get_double("/physical/lambda6"))
    {
    }
    double Lambda, N, T, lambda2, lambda4, lambda6;
  };

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"m2">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  constexpr auto idxf = FEFunctionDesc{};

  /// Where the V integral runs: TBB on the CPU, or the GPU in double or single precision.
  enum class Backend { TBB, GPU, GPU_float };

  /**
   * @tparam backend integrator of the batched flux; the per-point flux always uses the TBB one, as a
   * GPU get() must not be called from several threads at once.
   * @tparam batched whether the model evaluates its flux with map_points. Without it, flux_source_batch is
   * AbstractModel's default: the per-point flux in a flat parallel loop.
   */
  template <Backend backend = Backend::TBB, bool batched = true>
  class Model : public def::AbstractModel<Model<backend, batched>, Components>,
                public def::fRG,
                public def::LLFFlux<Model<backend, batched>>,
                public def::FlowBoundaries<Model<backend, batched>>,
                public def::AD<Model<backend, batched>>
  {
  public:
    static constexpr uint dim = 1;
    static constexpr bool batch_reads_hessians = false;

  protected:
    const Parameters prm;
    mutable ONFiniteTBatchedFlows flow_equations;

  public:
    Model(const ConfigTree &config) : def::fRG(config.get_double("/physical/Lambda")), prm(config), flow_equations(config)
    {
      flow_equations.set_k(Lambda);
      flow_equations.set_T(prm.T);
    }

    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      const auto &rho = pos[0];
      values[idxf("m2")] = prm.lambda2 + prm.lambda4 * rho + prm.lambda6 * powr<2>(rho);
    }

    void set_time(double t_)
    {
      t = t_;
      k = std::exp(-t) * prm.Lambda;
      flow_equations.set_k(k);
    }

    template <typename NT, typename Solution> void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux, const Point<dim> &x, const Solution &sol) const
    {
      const auto rho = x[0];
      const auto &fe_functions = get<"fe_functions">(sol);
      const auto &fe_derivatives = get<"fe_derivatives">(sol);

      const auto m2Pi = fe_functions[idxf("m2")];
      const auto m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

      flow_equations.V.get(flux[idxf("m2")][0], k, prm.N, prm.T, m2Pi, m2Sigma);
    }

    template <typename Out, typename Batch> void flux_source_batch(Out &out, const Batch &batch) const
    {
      if constexpr (!batched) {
        def::AbstractModel<Model, Components>::flux_source_batch(out, batch);
        return;
      }
      using NT = typename Batch::number_type;
      const size_t n = batch.size();
      const auto m2Pi = batch.values(idxf("m2"));
      const auto dm2 = batch.derivatives(idxf("m2"), 0);
      const auto rho = batch.coordinates(0);

      std::vector<NT> m2Sigma(n);
      for (size_t i = 0; i < n; ++i)
        m2Sigma[i] = m2Pi.data[i] + 2. * rho.data[i] * dm2.data[i];

      integrator().map_points(out.flux(idxf("m2"), 0), n, k, prm.N, prm.T, m2Pi, PointArray<NT>{m2Sigma.data(), n});
    }

    template <int dim, typename NumberType, typename Solution> void cell_indicator(NumberType &indicator, const Point<dim> & /*p*/, const Solution &sol) const
    {
      indicator = get<"fe_hessians">(sol)[idxf("m2")][0][0];
    }

    template <int dim, typename DataOut, typename Solutions> void readouts(DataOut &output, const Point<dim> &x, const Solutions &sol) const
    {
      const auto &fe_functions = get<"fe_functions">(sol);
      const auto &fe_derivatives = get<"fe_derivatives">(sol);

      const double rho = x[0];
      const double m2Pi = fe_functions[idxf("m2")];
      const double m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

      auto out_file = output.table("data.csv");
      out_file.set_Lambda(Lambda);
      out_file.value("sigma [GeV]", std::sqrt(2. * rho));
      out_file.value("m^2_{pi} [GeV^2]", m2Pi);
      out_file.value("m^2_{sigma} [GeV^2]", m2Sigma);
      out_file.value("m_{pi} [GeV]", m2Pi > 0. ? std::sqrt(m2Pi) : 0.);
      out_file.value("m_{sigma} [GeV]", m2Sigma > 0. ? std::sqrt(m2Sigma) : 0.);
    }

    /// The V integrator of the chosen backend, for direct access (e.g. its map_points policy).
    auto &integrator() const
    {
      if constexpr (backend == Backend::TBB)
        return flow_equations.V;
      else if constexpr (backend == Backend::GPU)
        return flow_equations.V_GPU;
      else
        return flow_equations.V_GPU_f;
    }
  };
} // namespace ON_batched
