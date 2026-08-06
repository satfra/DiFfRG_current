#pragma once

#include <DiFfRG/common/json.hh>

#include <string>

namespace DiFfRG::Config
{
  /** Typed configuration for reconstructed-potential EoM searches. */
  struct EoMConfig {
    inline static constexpr double default_abs_tol = 1e-12;
    inline static constexpr unsigned int default_max_iter = 100;
    inline static constexpr double default_smoothing_length = -1.;
    inline static constexpr double default_bound_tolerance = 1e-12;
    inline static constexpr double default_armijo_coefficient = 1e-4;
    inline static constexpr unsigned int default_max_backtracks = 20;

    EoMConfig() = default;
    EoMConfig(double abs_tol, unsigned int max_iter, double smoothing_length,
              double bound_tolerance = default_bound_tolerance, double armijo_coefficient = default_armijo_coefficient,
              unsigned int max_backtracks = default_max_backtracks);
    explicit EoMConfig(const JSONValue &json);

    static std::string get_defaults();
    void validate() const;

    double abs_tol = default_abs_tol;
    unsigned int max_iter = default_max_iter;
    double smoothing_length = default_smoothing_length;
    double bound_tolerance = default_bound_tolerance;
    double armijo_coefficient = default_armijo_coefficient;
    unsigned int max_backtracks = default_max_backtracks;
  };
} // namespace DiFfRG::Config
