#pragma once

#include <DiFfRG/common/minimization.hh>
#include <DiFfRG/common/root_finding.hh>
#include <DiFfRG/common/utils.hh>

#include <filesystem>

template <typename FUN>
double tune_m2A(JSONValue &json, const std::string top_folder, const std::string output_name, const FUN &run,
                double hint_m2A = 0.0, double hint_delta = -1.0, double tol_override = -1.0)
{
  const double tol = tol_override > 0 ? tol_override : json.get_double("/tuning/m2A_tol");
  const uint precision = (uint)std::max(-std::log10(tol), 11.) + 1;

  double lower_m2A, upper_m2A;

  auto logger = build_logger("m2Alog", top_folder + output_name + "_m2A_tuning/" + "tuning.log");

  if (hint_delta > 0) {
    // Warm-start: use hint as center with narrow bracket
    lower_m2A = hint_m2A - hint_delta;
    upper_m2A = hint_m2A + hint_delta;
    spdlog::get("m2Alog")->info("Warm-start m2A bracket: [{:.12e}, {:.12e}]", lower_m2A, upper_m2A);

    // Validate bracket: test both endpoints
    json.set_double("/physical/m2A", lower_m2A);
    {
      std::stringstream ss;
      ss << top_folder << output_name << "_m2A_tuning/";
      ss << std::scientific << std::setprecision(precision);
      ss << "m2A_" << lower_m2A;
      json.set_string("/output/folder", ss.str());
    }
    const bool lower_result = run(json, "m2Alog");

    json.set_double("/physical/m2A", upper_m2A);
    {
      std::stringstream ss;
      ss << top_folder << output_name << "_m2A_tuning/";
      ss << std::scientific << std::setprecision(precision);
      ss << "m2A_" << upper_m2A;
      json.set_string("/output/folder", ss.str());
    }
    const bool upper_result = run(json, "m2Alog");

    if (lower_result == upper_result) {
      // Bracket doesn't straddle root — expand up to 5x, then fall back
      spdlog::get("m2Alog")->info("Warm-start bracket invalid, expanding...");
      bool found = false;
      for (int attempt = 0; attempt < 5; ++attempt) {
        hint_delta *= 2.0;
        lower_m2A = hint_m2A - hint_delta;
        upper_m2A = hint_m2A + hint_delta;
        if (lower_result) {
          // Both succeeded — need lower to fail, so expand downward
          json.set_double("/physical/m2A", lower_m2A);
          std::stringstream ss;
          ss << top_folder << output_name << "_m2A_tuning/";
          ss << std::scientific << std::setprecision(precision);
          ss << "m2A_" << lower_m2A;
          json.set_string("/output/folder", ss.str());
          if (!run(json, "m2Alog")) {
            found = true;
            break;
          }
        } else {
          // Both failed — need upper to succeed, so expand upward
          json.set_double("/physical/m2A", upper_m2A);
          std::stringstream ss;
          ss << top_folder << output_name << "_m2A_tuning/";
          ss << std::scientific << std::setprecision(precision);
          ss << "m2A_" << upper_m2A;
          json.set_string("/output/folder", ss.str());
          if (run(json, "m2Alog")) {
            found = true;
            break;
          }
        }
      }
      if (!found) {
        spdlog::get("m2Alog")->warn("Warm-start expansion failed, falling back to cold-start");
        hint_delta = -1.0;
      }
    }
  }

  if (hint_delta <= 0) {
    // Cold-start: read bounds from JSON and validate
    lower_m2A = json.get_double("/tuning/lower_m2A");
    upper_m2A = json.get_double("/tuning/upper_m2A");
    const double init_m2A = 0.5 * (lower_m2A + upper_m2A);

    // First, check upper and lower bounds: If the lower one does not fail, or
    // m2A > 0, halve lower_m2A and try again. If upper_m2A does not give a
    // positive m2A, double it and try again.
    const int max_bound_iter = 50;
    bool bounds_ok = false;
    int bound_iter = 0;
    while (!bounds_ok && bound_iter++ < max_bound_iter) {
      std::stringstream ss;
      ss << top_folder << output_name << "_m2A_tuning/";
      ss << std::scientific << std::setprecision(precision);
      ss << "m2A_" << lower_m2A;
      json.set_string("/output/folder", ss.str());
      json.set_double("/physical/m2A", lower_m2A);

      spdlog::get("m2Alog")->info("Testing lower bound for m2A: {:.12e}", lower_m2A);

      if (run(json, "m2Alog"))
        lower_m2A -= 0.1 * std::abs(init_m2A);
      else
        // The lower bound is too low, but that's fine, the tuning will take
        // care of it.
        bounds_ok = true;
    }
    if (!bounds_ok)
      throw std::runtime_error("tune_m2A: could not find valid lower bound after " + std::to_string(max_bound_iter) +
                               " iterations");

    // Now, do the same for the upper bound
    bounds_ok = false;
    bound_iter = 0;
    while (!bounds_ok && bound_iter++ < max_bound_iter) {
      std::stringstream ss;
      ss << top_folder << output_name << "_m2A_tuning/";
      ss << std::scientific << std::setprecision(precision);
      ss << "m2A_" << upper_m2A;
      json.set_string("/output/folder", ss.str());
      json.set_double("/physical/m2A", upper_m2A);

      spdlog::get("m2Alog")->info("Testing upper bound for m2A: {:.12e}", upper_m2A);

      if (!run(json, "m2Alog")) {
        // The upper bound is too low, so we double it and try again.
        lower_m2A = upper_m2A;
        upper_m2A += 0.1 * std::abs(init_m2A);
        continue;
      } else
        bounds_ok = true;
    }
    if (!bounds_ok)
      throw std::runtime_error("tune_m2A: could not find valid upper bound after " + std::to_string(max_bound_iter) +
                               " iterations");
  }

  spdlog::get("m2Alog")->info("Lower bound for m2A: {:.12e}", lower_m2A);
  spdlog::get("m2Alog")->info("Upper bound for m2A: {:.12e}", upper_m2A);

  BisectionRootFinder search(
      [&](const double x) -> bool {
        std::stringstream ss;
        ss << top_folder << output_name << "_m2A_tuning/";
        ss << std::scientific << std::setprecision(precision);
        ss << "m2A_" << x;
        json.set_string("/output/folder", ss.str());
        json.set_double("/physical/m2A", x);

        spdlog::get("m2Alog")->info("Tuning step {} with m2A = {:.12e}", search.get_iter(), x);

        return run(json, "m2Alog");
      },
      tol, 100);
  search.set_bounds(lower_m2A, upper_m2A);
  const double m2A = search.search();
  const int tuning_steps = search.get_iter();

  const std::string last_sim_folder = json.get_string("/output/folder");
  std::filesystem::copy(last_sim_folder, top_folder,
                        std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);

  spdlog::get("m2Alog")->info("m2A Tuning finished after {} steps, m2A = {:.12e}({:.12e})", tuning_steps, m2A, tol);
  spdlog::drop("m2Alog");
  return m2A;
}

template <typename FUN>
void tune_STI(JSONValue &json, const std::string top_folder, const std::string output_name, const FUN &run)
{
  auto logger = build_logger("STIlog", top_folder + output_name + "_STI_tuning/tuning.log");

  const double original_final_time = json.get_double("/timestepping/final_time");

  // Save original quadrature orders and apply reduction factor for STI search
  const uint original_x_order = json.get_uint("/integration/x_order");
  const uint original_cos1_order = json.get_uint("/integration/cos1_order");
  const uint original_cos2_order = json.get_uint("/integration/cos2_order");
  const uint original_phi_order = json.get_uint("/integration/phi_order");

  const double STI_quadrature_factor = json.get_double("/tuning/STI_quadrature_factor", 1.0);
  json.set_uint("/integration/x_order", std::max(1u, (uint)(original_x_order * STI_quadrature_factor)));
  json.set_uint("/integration/cos1_order", std::max(1u, (uint)(original_cos1_order * STI_quadrature_factor)));
  json.set_uint("/integration/cos2_order", std::max(1u, (uint)(original_cos2_order * STI_quadrature_factor)));
  json.set_uint("/integration/phi_order", std::max(1u, (uint)(original_phi_order * STI_quadrature_factor)));

  spdlog::get("STIlog")->info("Quadrature reduction factor: {:.2f} (x: {} -> {}, cos1: {} -> {}, cos2: "
                              "{} -> {}, phi: {} -> {})",
                              STI_quadrature_factor, original_x_order, json.get_uint("/integration/x_order"),
                              original_cos1_order, json.get_uint("/integration/cos1_order"), original_cos2_order,
                              json.get_uint("/integration/cos2_order"), original_phi_order,
                              json.get_uint("/integration/phi_order"));

  uint counter = 0;
  double last_m2A = 0.0;
  double warm_delta = -1.0;
  std::vector<std::array<double, 2>> x_values;
  GSLSimplexMinimizer<2> minimizer(
      [&](const std::array<double, 2> &x) -> double {
        std::stringstream ss;
        ss << top_folder << output_name << "_STI_tuning/step_" << counter << "/";
        json.set_string("/output/folder", ss.str());

        json.set_double("/physical/alphaA3", x[0]);
        json.set_double("/physical/alphaA4", x[1]);

        x_values.push_back(x);

        spdlog::get("STIlog")->info("STI tuning step {}:\n    alphaAcbc = {:.8e},\n    alphaA3 = "
                                    "{:.8e},\n    alphaA4 = {:.8e}",
                                    counter, json.get_double("/physical/alphaAcbc"), x[0], x[1]);

        // Use shorter flow during STI search
        json.set_double("/timestepping/final_time", json.get_double("/tuning/STI_flow_final_time"));

        last_m2A =
            tune_m2A(json, ss.str(), output_name, run, last_m2A, warm_delta, json.get_double("/tuning/m2A_tol_coarse"));
        // Enable warm-starting for subsequent calls
        warm_delta = json.get_double("/tuning/m2A_warm_start_delta");
        counter++;

        HDF5Input hdf5_input(ss.str() + "/" + output_name + ".h5");
        const auto ZA3 = hdf5_input.load_map("ZA3");
        const auto ZA4 = hdf5_input.load_map("ZA4");
        const auto ZAcbc = hdf5_input.load_map("ZAcbc");
        const auto ZA = hdf5_input.load_map("ZA");
        const auto Zc = hdf5_input.load_map("Zc");
        const auto pGeV = hdf5_input.load_map_coord<1>("ZA");

        // starting from the last line, find the index closest to p = 10
        const double p_low = json.get_double("/tuning/STI_scale");
        double dist_low = std::numeric_limits<double>::max();
        uint idx_low = 0;
        for (int i = pGeV.size() - 1; i >= 0; --i) {
          if (std::abs(pGeV[i] - p_low) < dist_low) {
            dist_low = std::abs(pGeV[i] - p_low);
            idx_low = i;
          } else
            break;
        }
        const double p_high = 2 * p_low;
        double dist_high = std::numeric_limits<double>::max();
        uint idx_high = 0;
        for (int i = pGeV.size() - 1; i >= 0; --i) {
          if (std::abs(pGeV[i] - p_high) < dist_high && pGeV[i] > p_low) {
            dist_high = std::abs(pGeV[i] - p_high);
            idx_high = i;
          } else
            break;
        }

        // Invariant couplings: alpha = g^2/(4pi), derived from Z-factors
        const double alphaA3_low = powr<2>(ZA3[idx_low]) / (powr<3>(ZA[idx_low]) * 4 * M_PI);
        const double alphaA4_low = ZA4[idx_low] / (powr<2>(ZA[idx_low]) * 4 * M_PI);
        const double alphaAcbc_low = powr<2>(ZAcbc[idx_low]) / (ZA[idx_low] * powr<2>(Zc[idx_low]) * 4 * M_PI);

        const double alphaA3_high = powr<2>(ZA3[idx_high]) / (powr<3>(ZA[idx_high]) * 4 * M_PI);
        const double alphaA4_high = ZA4[idx_high] / (powr<2>(ZA[idx_high]) * 4 * M_PI);
        const double alphaAcbc_high = powr<2>(ZAcbc[idx_high]) / (ZA[idx_high] * powr<2>(Zc[idx_high]) * 4 * M_PI);

        // return the sum of squared differences
        const double diff_low = std::max({std::abs(alphaA3_low - alphaAcbc_low), std::abs(alphaA3_low - alphaA4_low),
                                          std::abs(alphaA4_low - alphaAcbc_low)});
        const double diff_high =
            std::max({std::abs(alphaA3_high - alphaAcbc_high), std::abs(alphaA3_high - alphaA4_high),
                      std::abs(alphaA4_high - alphaAcbc_high)});
        const double diff = 0.7 * powr<2>(diff_low) + 0.3 * powr<2>(diff_high);

        spdlog::get("STIlog")->info("distance = {:.8e}", diff);
        return diff;
      },
      json.get_double("/tuning/STI_tol"), 1000);

  minimizer.set_step_size(json.get_double("/tuning/STI_step_size"));
  const std::array<double, 2> x0{{json.get_double("/physical/alphaA3"), json.get_double("/physical/alphaA4")}};
  minimizer.set_x0(x0);
  const std::array<double, 2> minimum = minimizer.minimize();

  // Final refinement: run one full-quality flow with optimal couplings
  spdlog::get("STIlog")->info("Running final refinement with optimal couplings:\n"
                              "    alphaA3 = {:.8e},\n    alphaA4 = {:.8e}",
                              minimum[0], minimum[1]);

  json.set_double("/physical/alphaA3", minimum[0]);
  json.set_double("/physical/alphaA4", minimum[1]);
  json.set_double("/timestepping/final_time", original_final_time);

  // Restore original quadrature orders for final refinement
  json.set_uint("/integration/x_order", original_x_order);
  json.set_uint("/integration/cos1_order", original_cos1_order);
  json.set_uint("/integration/cos2_order", original_cos2_order);
  json.set_uint("/integration/phi_order", original_phi_order);

  std::stringstream ss_final;
  ss_final << top_folder << output_name << "_STI_tuning/final/";
  tune_m2A(json, ss_final.str(), output_name, run, last_m2A, warm_delta);

  std::filesystem::copy(ss_final.str(), top_folder,
                        std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);

  spdlog::get("STIlog")->info("Optimal STI couplings:\n    alphaA3 = {:.8e},\n    alphaA4 = {:.8e}", minimum[0],
                              minimum[1]);
  spdlog::get("STIlog")->info("STI Tuning finished after {} steps", counter);
  spdlog::drop("STIlog");
}