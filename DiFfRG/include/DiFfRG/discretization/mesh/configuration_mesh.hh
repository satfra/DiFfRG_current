#pragma once
#include "DiFfRG/common/math.hh"
#include <DiFfRG/common/utils.hh>
#include <stdexcept>
#include <string>
#include <vector>

namespace DiFfRG
{

  namespace Config
  {

    namespace internal
    {
      inline std::vector<std::string> string_to_substrings_array(const std::string str)
      {
        std::vector<std::string> array;
        std::istringstream ss(str);
        std::string buf;
        while (std::getline(ss, buf, ','))
          array.push_back(buf);
        return array;
      }
      inline std::array<double, 3> string_to_range(const std::string str)
      {
        std::vector<double> vec;
        std::istringstream ss(str);
        std::string buf;
        while (std::getline(ss, buf, ':'))
          vec.push_back(std::stod(buf));
        if (vec.size() != 3) throw std::runtime_error(str + " is not a range");

        std::array<double, 3> arr;
        std::copy_n(vec.begin(), 3, arr.begin());
        return arr;
      }
    } // namespace internal

    class GridAxis
    {
    public:
      GridAxis() = delete;
      GridAxis(std::string json_string)
      {
        auto subrange = internal::string_to_range(json_string);
        min = subrange[0];
        step = subrange[1];
        max = subrange[2];

        validate();
      };

      double min;
      double max;
      double step;

      /**
       * Return a vector of uniform step widths covering [min, max].
       * If the configured step fits the interval (within is_close) it is used;
       * otherwise the count is increased by one and the width is adjusted.
       */
      std::vector<double> get_stepwiths() const
      {
        double local_step = step;
        auto steps = static_cast<uint>((max - min) / local_step);
        if (!is_close((min + local_step * steps), max)) {
          local_step = (max - min) / (steps + 1);
          ++steps;
        }
        return std::vector<double>(steps, local_step);
      }

    private:
      void validate()
      {
        bool condition = (min < max) && (step <= (max - min));
        if (!condition) {
          throw std::runtime_error(std::format(
              "invalid range: min={}, step={}, max={} (require min < max and 0 < step <= max-min)", min, step, max));
        }
      }
    };
    namespace internal
    {
      inline void check_ranges_consistency(const std::vector<GridAxis> &ranges)
      {
        for (uint i = 1; i < ranges.size(); ++i)
          if (!is_close(ranges[i].min, ranges[i - 1].max))
            throw std::runtime_error("Your range definition is inconsistent!");
      }

      template <typename T, typename J> void append_range(T &to_append, const J &range)
      {
        for (const auto &element : range) {
          to_append.push_back(element);
        }
      }
    } // namespace internal

    template <int dim> class ConfigurationMesh
    {
    public:
      ConfigurationMesh() = delete;
      ConfigurationMesh(const DiFfRG::JSONValue &json)
      {
        static_assert(dim >= 1 && dim <= 3, "ConfigurationMesh only supports dim = 1, 2, 3");

        const std::array<std::string, 3> grid_names = {"/discretization/grid/x_grid", "/discretization/grid/y_grid",
                                                       "/discretization/grid/z_grid"};

        for (int i = 0; i < dim; ++i) {
          auto subranges = internal::string_to_substrings_array(json.get_string(grid_names[i]));
          for (const auto &subrange_str : subranges)
            grids[i].emplace_back(subrange_str);
          internal::check_ranges_consistency(grids[i]);
        }

        refine = json.get_uint("/discretization/grid/refine", 0);
      }

      inline std::vector<std::vector<double>> get_step_withs_for_triangulation() const
      {
        std::vector<std::vector<double>> step_sizes(dim);
        for (int i = 0; i < dim; ++i) {
          for (const auto &sub_grid_axis : grids[i]) {
            internal::append_range(step_sizes[i], sub_grid_axis.get_stepwiths());
          }
        }
        return step_sizes;
      }

      inline dealii::Point<dim, double> get_lower_left() const
      {
        dealii::Point<dim, double> lower_left;
        for (int i = 0; i < dim; ++i) {
          lower_left[i] = grids[i][0].min;
        }
        return lower_left;
      }

      inline dealii::Point<dim, double> get_upper_right() const
      {
        dealii::Point<dim, double> upper_right;
        for (int i = 0; i < dim; ++i) {
          upper_right[i] = grids[i].back().max;
        }
        return upper_right;
      }

      std::array<std::vector<GridAxis>, dim> grids;
      uint refine;
    };

  } // namespace Config
} // namespace DiFfRG