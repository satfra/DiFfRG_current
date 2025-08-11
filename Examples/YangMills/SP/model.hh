#pragma once

#include "flows/flows.hh"
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/interpolation/spline_interpolator_1D.hh>
#include <DiFfRG/physics/utils.hh>
#include <Kokkos_Core_fwd.hpp>
#include <cmath>

using namespace dealii;
using namespace DiFfRG;

struct Parameters {
  Parameters(const JSONValue &json)
  {
    try {
      Lambda = json.get_double("/physical/Lambda");

      alphaA3 = json.get_double("/physical/alphaA3");
      alphaA4 = json.get_double("/physical/alphaA4");
      alphaAcbc = json.get_double("/physical/alphaAcbc");

      tilt_A3 = json.get_double("/physical/tilt_A3");
      tilt_A4 = json.get_double("/physical/tilt_A4");
      tilt_Acbc = json.get_double("/physical/tilt_Acbc");

      m2A = json.get_double("/physical/m2A");

      p_grid_min = json.get_double("/discretization/p_grid_min");
      p_grid_max = json.get_double("/discretization/p_grid_max");
      p_grid_bias = json.get_double("/discretization/p_grid_bias");
    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }

  double Lambda;
  double alphaA3, alphaA4, alphaAcbc;
  double tilt_A3, tilt_A4, tilt_Acbc;
  double m2A;

  double p_grid_min, p_grid_max, p_grid_bias;
};

// Size of the momentum grid
static constexpr uint p_grid_size = 96;
// As for components, we have one FE function (u) and several extractors.
using VariableDesc =
    VariableDescriptor<FunctionND<"ZA3", p_grid_size>, FunctionND<"ZAcbc", p_grid_size>, FunctionND<"ZA4", p_grid_size>,

                       FunctionND<"ZA", p_grid_size>, FunctionND<"Zc", p_grid_size>>;
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDescriptor<>>;

constexpr auto idxv = VariableDesc{};

/**
 * @brief This class implements the numerical model for the quark-meson model at finite temperature and chemical
 * potential.
 */
class YangMills : public def::AbstractModel<YangMills, Components>,
                  public def::fRG,        // this handles the fRG time
                  public def::NoJacobians // define all jacobians per AD
{
  const Parameters prm;

  using Coordinates1D = LogarithmicCoordinates1D<double>;
  const Coordinates1D coordinates1D;

  const std::vector<double> grid1D;

  mutable YangMillsFlows flow_equations;

  mutable SplineInterpolator1D<double, Coordinates1D, GPU_memory> dtZc, dtZA, ZA, Zc;
  mutable SplineInterpolator1D<double, Coordinates1D, GPU_memory> ZA4, ZAcbc, ZA3;

public:
  YangMills(const JSONValue &json)
      : def::fRG(json.get_double("/physical/Lambda")), prm(json),
        coordinates1D(p_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_bias), grid1D(make_grid(coordinates1D)),
        flow_equations(json), dtZc(coordinates1D), dtZA(coordinates1D), ZA(coordinates1D),
        Zc(coordinates1D),                                           // propagators
        ZA4(coordinates1D), ZAcbc(coordinates1D), ZA3(coordinates1D) // couplings
  {
    flow_equations.set_k(prm.Lambda);
    k = std::exp(-t) * Lambda;
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const
  {
    for (uint i = 0; i < p_grid_size; ++i) {
      values[idxv("ZA4") + i] = 4. * M_PI * prm.alphaA4 + prm.tilt_A4 * std::log(grid1D[i] / prm.p_grid_max);
      values[idxv("ZA3") + i] = std::sqrt(4. * M_PI * prm.alphaA3) + prm.tilt_A3 * std::log(grid1D[i] / prm.p_grid_max);
      values[idxv("ZAcbc") + i] =
          std::sqrt(4. * M_PI * prm.alphaAcbc) + prm.tilt_Acbc * std::log(grid1D[i] / prm.p_grid_max);

      values[idxv("ZA") + i] = (powr<2>(grid1D[i]) + prm.m2A) / powr<2>(grid1D[i]);
      values[idxv("Zc") + i] = 1.;
    }
  }

  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * Lambda;
    flow_equations.set_k(k);
  }

  template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &data) const
  {
    const auto &variables = get<"variables">(data);

    ZA3.update(&variables.data()[idxv("ZA3")]);
    ZAcbc.update(&variables.data()[idxv("ZAcbc")]);
    ZA4.update(&variables.data()[idxv("ZA4")]);

    ZA.update(&variables.data()[idxv("ZA")]);
    Zc.update(&variables.data()[idxv("Zc")]);

    // set up arguments for the integrators
    const auto arguments = device::tie(k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);

    // copy the propagators for comparison
    std::vector<double> old_dtZA(p_grid_size);
    std::vector<double> old_dtZc(p_grid_size);

    // start by solving the equations for propagators
    bool eta_converged = false;
    uint n_iter = 0;
    while (!eta_converged) {
      // copy
      for (uint i = 0; i < p_grid_size; ++i) {
        old_dtZA[i] = dtZA[i];
        old_dtZc[i] = dtZc[i];
      }

      double res;
      flow_equations.ZA.get(res, 0.01, arguments);
      std::cout << "dtZA(0.01) = " << res << std::endl;

      const auto ZA_exec = flow_equations.ZA.map(&residual[idxv("ZA")], coordinates1D, arguments);
      const auto Zc_exec = flow_equations.Zc.map(&residual[idxv("Zc")], coordinates1D, arguments);
      Kokkos::fence();

      dtZA.update(&residual[idxv("ZA")]);
      dtZc.update(&residual[idxv("Zc")]);

      // check distance
      double dist = 0.;
      for (uint i = 0; i < p_grid_size; ++i) {
        dist = std::max(dist, std::abs(dtZA[i] - old_dtZA[i]) / std::abs(dtZA[i]));
        dist = std::max(dist, std::abs(dtZc[i] - old_dtZc[i]) / std::abs(dtZc[i]));
      }
      if (dist < 1e-4 || n_iter > 10) eta_converged = true;
      n_iter++;
    }
    std::cout << "Converged after " << n_iter << " iterations." << std::endl;

    const auto exec_ZA4 = flow_equations.ZA4.map(&residual[idxv("ZA4")], coordinates1D, arguments);
    const auto exec_ZAcbc = flow_equations.ZAcbc.map(&residual[idxv("ZAcbc")], coordinates1D, arguments);
    const auto exec_ZA3 = flow_equations.ZA3.map(&residual[idxv("ZA3")], coordinates1D, arguments);
    Kokkos::fence();
  }

  template <int dim, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim> &, const Solutions &sol) const
  {
    const auto &variables = get<"variables">(sol);

    this->Zc.update(&variables.data()[idxv("Zc")]);

    auto &out_file = output.csv_file("data.csv");
    out_file.set_Lambda(Lambda);

    const auto *ZA = &variables.data()[idxv("ZA")];
    const auto *Zc = &variables.data()[idxv("Zc")];
    std::vector<std::vector<double>> Zs_data(grid1D.size(), std::vector<double>(6, 0.));
    for (uint i = 0; i < p_grid_size; ++i) {
      Zs_data[i][0] = k;
      Zs_data[i][1] = grid1D[i];
      Zs_data[i][2] = dtZA[i];
      Zs_data[i][3] = dtZc[i];
      Zs_data[i][4] = ZA[i];
      Zs_data[i][5] = Zc[i];
    }

    const auto *ZA3 = &variables.data()[idxv("ZA3")];
    const auto *ZAcbc = &variables.data()[idxv("ZAcbc")];
    const auto *ZA4 = &variables.data()[idxv("ZA4")];
    std::vector<std::vector<double>> strong_couplings_data(grid1D.size(), std::vector<double>(8, 0.));
    for (uint i = 0; i < p_grid_size; ++i) {
      const double ZA_p = (ZA[i]); // * powr<2>(grid1D[i]) + m2A) / powr<2>(grid1D[i]);
      const double Zc_p = Zc[i];
      strong_couplings_data[i][0] = k;
      strong_couplings_data[i][1] = grid1D[i];
      strong_couplings_data[i][2] = powr<2>(ZAcbc[i]) / (4. * M_PI * ZA_p * powr<2>(Zc_p));
      strong_couplings_data[i][3] = powr<2>(ZA3[i]) / (4. * M_PI * powr<3>(ZA_p));
      strong_couplings_data[i][4] = ZA4[i] / (4. * M_PI * powr<2>(ZA_p));
      strong_couplings_data[i][5] = ZAcbc[i];
      strong_couplings_data[i][6] = ZA3[i];
      strong_couplings_data[i][7] = ZA4[i];
    }

    if (is_close(t, 0.)) {
      const std::vector<std::string> strong_couplings_header =
          std::vector<std::string>{"k [GeV]", "p [GeV]", "alphaAcbc", "alphaA3", "alphaA4", "ZAcbc", "ZA3", "ZA4"};
      const std::vector<std::string> Zs_header =
          std::vector<std::string>{"k [GeV]", "p [GeV]", "dtZA", "dtZc", "ZA", "Zc"};

      output.dump_to_csv("strong_couplings.csv", strong_couplings_data, false, strong_couplings_header);
      output.dump_to_csv("Zs.csv", Zs_data, false, Zs_header);
    } else {
      output.dump_to_csv("strong_couplings.csv", strong_couplings_data, true);
      output.dump_to_csv("Zs.csv", Zs_data, true);
    }
  }
};
