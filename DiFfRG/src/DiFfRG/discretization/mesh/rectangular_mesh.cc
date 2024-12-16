// standard library
#include <fstream>

// external libraries
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

namespace DiFfRG
{
  using namespace dealii;

  namespace internal
  {
    std::array<double, 3> string_to_range(const std::string str);
    std::vector<std::string> string_to_substrings_array(const std::string str);
    void check_ranges_consistency(const std::vector<std::array<double, 3>> &ranges);
    void append_range(std::vector<std::vector<double>> &dest, const std::array<double, 3> &range, const uint d);
  } // namespace internal

  template <uint dim> RectangularMesh<dim>::RectangularMesh(const JSONValue &json) : json(json)
  {
    static_assert(dim > 0 && dim < 4);
    make_grid();
  }

  template <uint dim> void RectangularMesh<dim>::make_grid()
  {
    std::vector<std::vector<double>> step_sizes;
    std::vector<std::array<double, 3>> x_grid_ranges;
    std::vector<std::array<double, 3>> y_grid_ranges;
    std::vector<std::array<double, 3>> z_grid_ranges;
    try {
      auto x_grid_ranges_str = internal::string_to_substrings_array(json.get_string("/discretization/grid/x_grid"));
      for (const auto &str : x_grid_ranges_str)
        x_grid_ranges.emplace_back(internal::string_to_range(str));
      internal::check_ranges_consistency(x_grid_ranges);
      for (const auto &range : x_grid_ranges)
        internal::append_range(step_sizes, range, 0);
      if constexpr (dim > 1) {
        auto y_grid_ranges_str = internal::string_to_substrings_array(json.get_string("/discretization/grid/y_grid"));
        for (const auto &str : y_grid_ranges_str)
          y_grid_ranges.emplace_back(internal::string_to_range(str));
        internal::check_ranges_consistency(y_grid_ranges);
        for (const auto &range : y_grid_ranges)
          internal::append_range(step_sizes, range, 1);
      }
      if constexpr (dim > 2) {
        auto z_grid_ranges_str = internal::string_to_substrings_array(json.get_string("/discretization/grid/z_grid"));
        for (const auto &str : z_grid_ranges_str)
          z_grid_ranges.emplace_back(internal::string_to_range(str));
        internal::check_ranges_consistency(z_grid_ranges);
        for (const auto &range : z_grid_ranges)
          internal::append_range(step_sizes, range, 2);
      }
    } catch (std::exception &e) {
      std::cerr << "Error in grid construction: " + std::string(e.what());
      spdlog::get("log")->error("Error in grid construction: {}", e.what());
    }

    if constexpr (dim == 1) {
      dealii::Point<1> origin{x_grid_ranges[0][0]};
      dealii::Point<1> extent{x_grid_ranges[x_grid_ranges.size() - 1][2]};
      dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, origin, extent);
    } else if constexpr (dim == 2) {
      dealii::Point<2> origin{x_grid_ranges[0][0], y_grid_ranges[0][0]};
      dealii::Point<2> extent{x_grid_ranges[x_grid_ranges.size() - 1][2], y_grid_ranges[y_grid_ranges.size() - 1][2]};
      dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, origin, extent);
    } else if constexpr (dim == 3) {
      dealii::Point<3> origin{x_grid_ranges[0][0], y_grid_ranges[0][0], z_grid_ranges[0][0]};
      dealii::Point<3> extent{x_grid_ranges[x_grid_ranges.size() - 1][2], y_grid_ranges[y_grid_ranges.size() - 1][2],
                              z_grid_ranges[z_grid_ranges.size() - 1][2]};
      dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, origin, extent);
    }

    triangulation.refine_global(json.get_uint("/discretization/grid/refine"));

    if constexpr (dim == 2) {
      std::ofstream out("grid.svg");
      dealii::GridOut grid_out;
      grid_out.write_svg(triangulation, out);
    }
  }

  namespace internal
  {
    std::array<double, 3> string_to_range(const std::string str)
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

    std::vector<std::string> string_to_substrings_array(const std::string str)
    {
      std::vector<std::string> array;
      std::istringstream ss(str);
      std::string buf;
      while (std::getline(ss, buf, ','))
        array.push_back(buf);
      return array;
    }

    void check_ranges_consistency(const std::vector<std::array<double, 3>> &ranges)
    {
      for (uint i = 1; i < ranges.size(); ++i)
        if (!is_close(ranges[i][0], ranges[i - 1][2]))
          throw std::runtime_error("Your range definition is inconsistent!");
    }

    void append_range(std::vector<std::vector<double>> &dest, const std::array<double, 3> &range, const uint d)
    {
      if (dest.size() <= d) dest.resize(d + 1);

      uint steps = uint((range[2] - range[0]) / range[1]);
      for (uint i = 0; i < steps; ++i)
        dest[d].push_back(range[1]);

      if (!is_close(range[0] + range[1] * steps, range[2])) {
        if (range[0] + range[1] * steps > range[2]) throw std::runtime_error("grid construction broke down!");
        dest[d].push_back(range[2] - (range[0] + range[1] * steps));
      }
    }
  } // namespace internal

  template class RectangularMesh<1>;
  template class RectangularMesh<2>;
  template class RectangularMesh<3>;
} // namespace DiFfRG