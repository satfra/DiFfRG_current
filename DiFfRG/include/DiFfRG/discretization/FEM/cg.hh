#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
// #include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/affine_constraints.h>

// DiFfRG
#include <DiFfRG/common/run_reporter.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/assembler/cg.hh>
#include <DiFfRG/discretization/discretization.hh>

namespace DiFfRG
{
  namespace CG
  {
    using namespace dealii;

    /**
     * @brief Class to manage the system on which we solve, i.e. fe spaces, grids, etc.
     * This class is a System for CG systems.
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
      Discretization(Mesh &mesh, const ConfigTree &config, ReportPort report_port = {})
          : mesh(mesh), config(config), log(std::move(report_port)),
            fe(std::make_shared<FESystem<dim>>(FE_Q<dim>(config.get_uint_or_warn("/discretization/fe_order", 3)),
                                               Components::count_fe_functions(0))),
            dof_handler(mesh.get_triangulation())
      {
        setup_dofs();
      };

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
      const auto &get_fe(const uint i = 0) const
      {
        if (i != 0) throw std::runtime_error("Wrong FE index");
        return *fe;
      }
      const auto &get_mapping() const { return mapping; }
      const auto &get_triangulation() const { return mesh.get_triangulation(); }
      auto &get_triangulation() { return mesh.get_triangulation(); }
      const Point<dim> &get_support_point(const uint dof) const { return support_points[dof]; }
      const auto &get_support_points() const { return support_points; }
      const auto &get_config() const { return config; }
      ReportPort report_port() const { return log; }

      void reinit() { setup_dofs(); }

      std::vector<uint> get_block_structure() const
      {
        std::vector<uint> block_structure{dof_handler.n_dofs()};
        if (Components::count_variables() > 0) block_structure.push_back(Components::count_variables());
        return block_structure;
      }

    protected:
      void setup_dofs()
      {
        dof_handler.distribute_dofs(*fe);
        // DoFRenumbering::component_wise(dof_handler);

        log.info("FEM: Number of active cells: {}", mesh.get_triangulation().n_active_cells());
        log.info("FEM: Number of degrees of freedom: {}", dof_handler.n_dofs());

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
  } // namespace CG
} // namespace DiFfRG
