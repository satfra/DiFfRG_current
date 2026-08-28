#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/affine_constraints.h>

// DiFfRG
#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/data/data.hh>
// #include <DiFfRG/discretization/discretization.hh>

#include <iostream>
#include <string>

namespace DiFfRG
{
  namespace FV
  {
    using namespace dealii;

    /**
     * @brief Class to manage the system on which we solve, i.e. fe spaces, grids, etc.
     * This class is a System for FV systems.
     *
     * @tparam Model_ The Model class used for the Simulation
     */
    template <typename Components_, typename NumberType_, typename Mesh_> class Discretization
    {
    public:
      using Components = Components_;
      using NumberType = NumberType_;
      using VectorType = Vector<NumberType>;
      using SparseMatrixType = SparseMatrix<NumberType>;
      using Mesh = Mesh_;
      static constexpr uint dim = Mesh::dim;
      static constexpr bool is_fv_discretization = true;
      Discretization(Mesh &mesh, const ConfigTree &config, ReportPort report_port = {})
          : mesh(mesh), config(config), log(std::move(report_port)),
            fe(std::make_shared<FESystem<dim>>(FE_DGQ<dim>(0), Components::count_fe_functions(0))),
            dof_handler(mesh.get_triangulation())
      {
        warn_on_ignored_fe_order();
        setup_dofs();
      }

      const auto &get_constraints(const uint i = 0) const
      {
        (void)i;
        return constraints;
      }
      auto &get_constraints(const uint i = 0)
      {
        (void)i;
        return constraints;
      }
      const auto &get_dof_handler(const uint i = 0) const
      {
        (void)i;
        return dof_handler;
      }
      auto &get_dof_handler(const uint i = 0)
      {
        (void)i;
        return dof_handler;
      }
      const auto &get_fe(uint i = 0) const
      {
        if (i != 0) throw std::runtime_error("Wrong FE index");
        return *fe;
      }
      const auto &get_mapping() const { return mapping; }
      const auto &get_triangulation() const { return mesh.get_triangulation(); }
      auto &get_triangulation() { return mesh.get_triangulation(); }
      const Point<dim> &get_support_point(const uint &dof) const { return support_points[dof]; }
      const auto &get_support_points() const { return support_points; }
      const auto &get_config() const { return config; }
      ReportPort report_port() const { return log; }

      void reinit() { setup_dofs(); }

      uint get_closest_dof(const Point<dim> &p) const
      {
        uint dof = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (uint i = 0; i < support_points.size(); ++i) {
          const auto dist = p.distance(support_points[i]);
          if (dist < min_dist) {
            min_dist = dist;
            dof = i;
          }
        }
        return dof;
      }

      std::vector<uint> get_block_structure() const
      {
        std::vector<uint> block_structure{dof_handler.n_dofs()};
        if (Components::count_variables() > 0) block_structure.push_back(Components::count_variables());
        return block_structure;
      }

    protected:
      /**
       * @brief The finite volume discretization always works on cell averages, i.e. FE_DGQ(0), and never reads
       * /discretization/fe_order. Warn the user if they set it to something nonzero, as it has no effect.
       */
      void warn_on_ignored_fe_order() const
      {
        const uint fe_order = config.get_uint("/discretization/fe_order", 0);
        if (fe_order == 0) return;

        const std::string message =
            "FV: /discretization/fe_order = " + std::to_string(fe_order) +
            " is ignored. The finite volume discretization always uses piecewise constants (fe_order = 0); "
            "reconstruction order is set by the choice of reconstructor and limiter instead.";
        log.warn("{}", message);
      }

      void setup_dofs()
      {
        dof_handler.distribute_dofs(*fe);

        log.info("FV: Number of active cells: {}", mesh.get_triangulation().n_active_cells());
        log.info("FV: Number of degrees of freedom: {}", dof_handler.n_dofs());

        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        support_points.resize(dof_handler.n_dofs());
        DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
      }

      Mesh &mesh;
      ConfigTree config;
      ReportPort log;

      std::shared_ptr<FESystem<dim>> fe;
      DoFHandler<dim> dof_handler;
      AffineConstraints<NumberType> constraints;
      MappingQ1<dim> mapping;
      std::vector<Point<dim>> support_points;
    };
  } // namespace FV
} // namespace DiFfRG
