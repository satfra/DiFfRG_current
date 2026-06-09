#pragma once

#include <DiFfRG/DiFfRG.hh>
using namespace DiFfRG;

#include "flows/flows.hh"

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

      eta_iter_max = json.get_int("/physical/eta_iter_max");
      eta_tol = json.get_double("/physical/eta_tol");

    } catch (std::exception &e) {
      std::cout << "Error in reading parameters: " << e.what() << std::endl;
    }
  }

  double Lambda;
  double alphaA3, alphaA4, alphaAcbc;
  double tilt_A3, tilt_A4, tilt_Acbc;
  double m2A;

  int eta_iter_max;
  double eta_tol;

  double p_grid_min, p_grid_max, p_grid_bias;
};

// Size of the 1D (propagator / symmetric-point) momentum grid.
static constexpr uint p_grid_size = 96;
// Size of the angle-resolved vertex grid. The 3-point dressings are tabulated over the symmetric
// triangle variables (S0, S1, SPhi): S0 the overall scale (logarithmic), S1 the shape (linear,
// [0,1)), SPhi an angle (linear). This is the legacy parametrisation; gridding this way keeps the
// transverse projector's collinear singularity confined to a single grid corner (S1->1, SPhi->0),
// instead of a whole face as with (|p1|,|p2|,cos), which is what makes the flow tunable.
static constexpr uint S0_grid_size = 96; // match the legacy momentum resolution on the vertex scale axis
static constexpr uint S1_grid_size = 8;
static constexpr uint SPhi_grid_size = 7;
static constexpr uint vertex_grid_size = S0_grid_size * S1_grid_size * SPhi_grid_size;

// The 3-gluon (ZA3), ghost-gluon (ZAcbc) and 4-gluon tadpole (ZA4tadpole) are angle-resolved 3D
// grids over (S0,S1,SPhi); the 4-gluon symmetric point (ZA4SP) is 1D; ZA and Zc are the
// gluon/ghost dressings (1D). The tadpole feeds the gluon self-energy with loop-dependent
// (external,loop) kinematics, as in the legacy code.
using VariableDesc =
    VariableDescriptor<FunctionND<"ZA3", vertex_grid_size>, FunctionND<"ZAcbc", vertex_grid_size>,
                       FunctionND<"ZA4SP", p_grid_size>, FunctionND<"ZA4tadpole", vertex_grid_size>,

                       FunctionND<"ZA", p_grid_size>, FunctionND<"Zc", p_grid_size>>;
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDescriptor<>>;

constexpr auto idxv = VariableDesc{};

/**
 * @brief Fully momentum- and angle-dependent Yang-Mills truncation. The gluon mass enters only
 * through the initial condition of the gluon dressing ZA, and is read off from ZA at the lowest
 * grid point in the readouts (no separate m2A variable).
 */
class YangMills : public def::AbstractModel<YangMills, Components>,
                  public def::fRG,        // this handles the fRG time
                  public def::NoJacobians // define all jacobians per AD
{
  const Parameters prm;

  using Coordinates1D = LogarithmicCoordinates1D<double>;
  using Coordinates3D = LogLinLinCoordinates; // CoordinatePackND<Log, Lin, Lin> over (S0, S1, SPhi)

  const Coordinates1D coordinates1D;
  const Coordinates1D S0_coordinates;
  const LinearCoordinates1D<double> S1_coordinates;
  const LinearCoordinates1D<double> SPhi_coordinates;
  const Coordinates3D coordinates3D;

  mutable YangMillsFlows flow_equations;

  mutable SplineInterpolator1D<double, Coordinates1D, GPU_memory> dtZc, dtZA, ZA, Zc, ZA4SP;
  mutable LinearInterpolatorND<double, Coordinates3D, GPU_memory> ZA3, ZAcbc, ZA4tadpole;

public:
  YangMills(const JSONValue &json)
      : def::fRG(json.get_double("/physical/Lambda")), prm(json),
        coordinates1D(p_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_bias),
        S0_coordinates(S0_grid_size, prm.p_grid_min, prm.p_grid_max, prm.p_grid_bias),
        S1_coordinates(S1_grid_size, 0.0, 0.9999),       // shape variable, [0,1)
        SPhi_coordinates(SPhi_grid_size, -M_PI, M_PI),   // angle, matches the atan2 feed range
        coordinates3D(S0_coordinates, S1_coordinates, SPhi_coordinates), flow_equations(json),
        dtZc(coordinates1D), dtZA(coordinates1D), ZA(coordinates1D), Zc(coordinates1D), ZA4SP(coordinates1D),
        ZA3(coordinates3D), ZAcbc(coordinates3D), ZA4tadpole(coordinates3D)
  {
    flow_equations.set_k(prm.Lambda);
    k = std::exp(-t) * Lambda;
  }

  template <typename Vector> void initial_condition_variables(Vector &values) const
  {
    // angle-resolved 3-point dressings (3D grids over S0,S1,SPhi)
    for (uint i = 0; i < S0_grid_size; ++i)
      for (uint j = 0; j < S1_grid_size; ++j)
        for (uint l = 0; l < SPhi_grid_size; ++l) {
          const auto point = coordinates3D.forward(i, j, l);
          const double S0 = point[0]; // overall scale, used only for the (small) initial tilt
          const size_t idx = (i * S1_grid_size + j) * SPhi_grid_size + l;
          values[idxv("ZA3") + idx] =
              std::sqrt(4. * M_PI * prm.alphaA3) + prm.tilt_A3 * std::log(S0 / prm.p_grid_max);
          values[idxv("ZAcbc") + idx] =
              std::sqrt(4. * M_PI * prm.alphaAcbc) + prm.tilt_Acbc * std::log(S0 / prm.p_grid_max);
          values[idxv("ZA4tadpole") + idx] =
              4. * M_PI * prm.alphaA4 + prm.tilt_A4 * std::log(S0 / prm.p_grid_max);
        }

    // 1D dressings
    for (uint i = 0; i < p_grid_size; ++i) {
      const double p = coordinates1D.forward(i);
      values[idxv("ZA4SP") + i] = 4. * M_PI * prm.alphaA4 + prm.tilt_A4 * std::log(p / prm.p_grid_max);
      values[idxv("ZA") + i] = (powr<2>(p) + prm.m2A) / powr<2>(p);
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
    ZA4SP.update(&variables.data()[idxv("ZA4SP")]);
    ZA4tadpole.update(&variables.data()[idxv("ZA4tadpole")]);

    ZA.update(&variables.data()[idxv("ZA")]);
    Zc.update(&variables.data()[idxv("Zc")]);

    // set up arguments for the integrators (order matches the generated map() signature)
    const auto arguments = device::tie(k, ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA);

    // copy the propagators for comparison
    std::vector<double> old_dtZA(p_grid_size);
    std::vector<double> old_dtZc(p_grid_size);

    // start by solving the equations for the propagators self-consistently in their anomalous dimensions
    bool eta_converged = false;
    int n_iter = 0;
    while (!eta_converged) {
      for (uint i = 0; i < p_grid_size; ++i) {
        old_dtZA[i] = dtZA[i];
        old_dtZc[i] = dtZc[i];
      }

      flow_equations.ZA.map(&residual[idxv("ZA")], coordinates1D, arguments);
      flow_equations.Zc.map(&residual[idxv("Zc")], coordinates1D, arguments);

      dtZA.update(&residual[idxv("ZA")]);
      dtZc.update(&residual[idxv("Zc")]);

      double dist = 0.;
      for (uint i = 0; i < p_grid_size; ++i) {
        dist = std::max(dist, std::abs(dtZA[i] - old_dtZA[i]) / std::abs(dtZA[i]));
        dist = std::max(dist, std::abs(dtZc[i] - old_dtZc[i]) / std::abs(dtZc[i]));
      }
      if (dist < prm.eta_tol || n_iter > prm.eta_iter_max) eta_converged = true;
      n_iter++;
    }
    std::cout << "Converged after " << n_iter << " iterations." << std::endl;

    // vertices
    flow_equations.ZA3.map(&residual[idxv("ZA3")], coordinates3D, arguments);
    flow_equations.ZAcbc.map(&residual[idxv("ZAcbc")], coordinates3D, arguments);
    flow_equations.ZA4SP.map(&residual[idxv("ZA4SP")], coordinates1D, arguments);
    flow_equations.ZA4tadpole.map(&residual[idxv("ZA4tadpole")], coordinates3D, arguments);
  }

  template <int dim, typename DataOut, typename Solutions>
  void readouts(DataOut &output, const Point<dim> &, const Solutions &sol) const
  {
    const auto &variables = get<"variables">(sol);

    Zc.update(&variables.data()[idxv("Zc")]);

    // sanity check: make sure 0 < Zc[0] < 1
    if (Zc[0] < 0 || Zc[0] > 1) throw std::runtime_error("Diverging result: Zc(0) = " + std::to_string(Zc[0]));

    auto &hdf = output.hdf5();
    hdf.map("ZA", coordinates1D, &(variables.data()[idxv("ZA")]));
    hdf.map("Zc", coordinates1D, &(variables.data()[idxv("Zc")]));
    hdf.map("ZA4SP", coordinates1D, &(variables.data()[idxv("ZA4SP")]));

    hdf.map("dtZA", dtZA);
    hdf.map("dtZc", dtZc);

    hdf.map("ZA3", coordinates3D, &(variables.data()[idxv("ZA3")]));
    hdf.map("ZAcbc", coordinates3D, &(variables.data()[idxv("ZAcbc")]));

    // Extract the gluon mass gap m2A from the deep-IR shape of ZA. In the IR the gluon
    // acquires a mass gap, ZA(p) = 1 + m2A/p² + (corrections), so the per-point estimate
    // m2A_i := (ZA(p_i) − 1)·p_i² obeys m2A_i = m2A + b·p_i² + … and extrapolates to m2A at
    // p = 0. A single deepest point is fragile to grid noise; instead least-squares fit a
    // straight line m2A_i = m2A + b·p² over the n_fit deepest-IR grid points and report the
    // intercept (the genuine p → 0 limit, not a window average). Ported from
    // qcd-codes/vacuum/QCD_vacuum_4F.
    const uint n_fit = std::min<uint>(6u, p_grid_size);
    const double *ZA_data = &(variables.data()[idxv("ZA")]);
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (uint i = 0; i < n_fit; ++i) {
      const double p2 = powr<2>(coordinates1D.forward(i));
      const double m2A_i = (ZA_data[i] - 1.0) * p2;
      sx += p2;
      sy += m2A_i;
      sxx += p2 * p2;
      sxy += p2 * m2A_i;
    }
    // Intercept of the least-squares line; fall back to the mean of the deep-IR points if
    // they are degenerate in p² (denom == 0).
    const double denom = n_fit * sxx - sx * sx;
    const double m2A = std::abs(denom) > 0.0 ? (sy * sxx - sx * sxy) / denom : sy / n_fit;

    hdf.scalar("k", k);
    hdf.scalar("m2A", m2A);
  }
};
