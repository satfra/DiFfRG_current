#pragma once

#include <DiFfRG/common/minimization.hh>
#include <DiFfRG/common/root_finding.hh>
#include <DiFfRG/common/utils.hh>

template <typename FUN> void tune_m2A(JSONValue &json, const std::string top_folder, const std::string output_name, const FUN &run)
{
  const uint precision = (uint)std::max(-std::log10(json.get_double("/tuning/m2A_tol")), 11.) + 1;

  double lower_m2A = json.get_double("/tuning/lower_m2A");
  double upper_m2A = json.get_double("/tuning/upper_m2A");
  const double init_m2A = 0.5 * (lower_m2A + upper_m2A);
  double m2A_out = 0.;

  auto logger = build_logger("m2Alog", top_folder + output_name + "_m2A_tuning/" + "tuning.log");

  // First, check upper and lower bounds: If the lower one does not fail, or m2A > 0, halve lower_m2A and try again. If
  // upper_m2A does not give a positive m2A, double it and try again.
  bool bounds_ok = false;
  while (!bounds_ok) {
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
      // The lower bound is too low, but that's fine, the tuning will take care of it.
      bounds_ok = true;
  }

  // Now, do the same for the upper bound
  bounds_ok = false;
  while (!bounds_ok) {
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
    }
    CSVReader reader(ss.str() + "/" + output_name + "_data.csv", ',', true);
    m2A_out = reader.value("m2A", reader.n_rows() - 1);
    if (m2A_out > 0)
      bounds_ok = true;
    else {
      lower_m2A = upper_m2A;
      upper_m2A += 0.1 * std::abs(init_m2A);
    }
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
      json.get_double("/tuning/m2A_tol"), 100);
  search.set_bounds(lower_m2A, upper_m2A);
  const double m2A = search.search();
  const int tuning_steps = search.get_iter();

  const std::string last_sim_folder = json.get_string("/output/folder");
  std::filesystem::copy(last_sim_folder, top_folder, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);

  spdlog::get("m2Alog")->info("m2A Tuning finished after {} steps, m2A = {:.12e}({:.12e})", tuning_steps, m2A, json.get_double("/tuning/m2A_tol"));
  spdlog::drop("m2Alog");
}

template <typename FUN> void tune_STI(JSONValue &json, const std::string top_folder, const std::string output_name, const FUN &run)
{
  auto logger = build_logger("STIlog", top_folder + output_name + "_STI_tuning/tuning.log");

  uint counter = 0;
  std::vector<std::array<double, 2>> x_values;
  GSLSimplexMinimizer<2> minimizer(
      [&](const std::array<double, 2> &x) -> double {
        std::stringstream ss;
        ss << top_folder << output_name << "_STI_tuning/step_" << counter << "/";
        json.set_string("/output/folder", ss.str());

        json.set_double("/physical/alphaA3", x[0]);
        json.set_double("/physical/alphaA4", x[1]);

        x_values.push_back(x);

        spdlog::get("STIlog")->info("STI tuning step {}:\n    alphaAcbc = {:.8e},\n    alphaA3 = {:.8e},\n    alphaA4 = {:.8e}", counter, json.get_double("/physical/alphaAcbc"),
                                    x[0], x[1]);
        tune_m2A(json, ss.str(), output_name, run);
        counter++;

        CSVReader reader(ss.str() + "/STI_couplings.csv", ',', true);
        // starting from the last line, find the index closest to p = 10
        const double p_low = json.get_double("/tuning/STI_scale");
        double dist_low = std::numeric_limits<double>::max();
        uint idx_low = 0;
        for (int i = reader.n_rows() - 1; i >= 0; --i) {
          if (std::abs(reader.value("pGeV", i) - p_low) < dist_low) {
            dist_low = std::abs(reader.value("pGeV", i) - p_low);
            idx_low = i;
          } else
            break;
        }
        const double p_high = 10.;
        double dist_high = std::numeric_limits<double>::max();
        uint idx_high = 0;
        for (int i = reader.n_rows() - 1; i >= 0; --i) {
          if (std::abs(reader.value("pGeV", i) - p_high) < dist_high) {
            dist_high = std::abs(reader.value("pGeV", i) - p_high);
            idx_high = i;
          } else
            break;
        }
        const double alphaA3_low = reader.value("alphaA3", idx_low);
        const double alphaAcbc_low = reader.value("alphaAcbc", idx_low);
        const double alphaA4_low = reader.value("alphaA4", idx_low);
        const double alphaA3_high = reader.value("alphaA3", idx_high);
        const double alphaAcbc_high = reader.value("alphaAcbc", idx_high);
        const double alphaA4_high = reader.value("alphaA4", idx_high);

        // return the sum of squared differences
        const double diff_low = std::max({std::abs(alphaA3_low - alphaAcbc_low), std::abs(alphaA3_low - alphaA4_low), std::abs(alphaA4_low - alphaAcbc_low)});
        const double diff_high = std::max({std::abs(alphaA3_high - alphaAcbc_high), std::abs(alphaA3_high - alphaA4_high), std::abs(alphaA4_high - alphaAcbc_high)});
        const double diff = 0.7 * powr<2>(diff_low) + 0.3 * powr<2>(diff_high);

        spdlog::get("STIlog")->info("distance = {:.8e}", diff);
        return diff;
      },
      json.get_double("/tuning/STI_tol"), 1000);

  minimizer.set_step_size(json.get_double("/tuning/STI_step_size"));
  const std::array<double, 2> x0{{json.get_double("/physical/alphaA3"), json.get_double("/physical/alphaA4")}};
  minimizer.set_x0(x0);
  const std::array<double, 2> minimum = minimizer.minimize();

  // Find the idx of the minimum
  double min_diff = std::numeric_limits<double>::max();
  uint min_idx = 0;
  for (uint i = 0; i < x_values.size(); ++i) {
    const double diff = std::abs(x_values[i][0] - minimum[0]) + std::abs(x_values[i][1] - minimum[1]);
    if (diff < min_diff) {
      min_diff = diff;
      min_idx = i;
    }
  }
  // Copy the folder of the minimum to the top folder
  std::string last_sim_folder = top_folder + output_name + "_STI_tuning/step_" + std::to_string(min_idx) + "/";
  std::filesystem::copy(last_sim_folder, top_folder, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);

  spdlog::get("STIlog")->info("Optimal STI couplings:\n    alphaA3 = {:.8e},\n    alphaA4 = {:.8e}",
                              minimum[0], minimum[1]);
  spdlog::get("STIlog")->info("STI Tuning finished after {} steps", counter);
  spdlog::drop("STIlog");
}