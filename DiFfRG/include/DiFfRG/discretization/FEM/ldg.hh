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
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/assembler/ldg.hh>

namespace DiFfRG
{
  namespace LDG
  {
    using namespace dealii;

    /**
     * @brief Class to manage the system on which we solve, i.e. fe spaces, grids, etc.
     * This class is a System for LDG systems, i.e. DG with additional projections (e.g. derivatives).
     *
     * @tparam Model_ The Model class used for the Simulation
     */
    template <typename Components_, typename NumberType_, typename Mesh_> class Discretization
    {
    public:
      using Components = Components_;
      using NumberType = NumberType_;
      using VectorType = Vector<NumberType>;
      using SparseMatrixType = BlockSparseMatrix<NumberType>;
      using Mesh = Mesh_;
      static constexpr uint dim = Mesh::dim;

      Discretization(Mesh &mesh, const JSONValue &json) : mesh(mesh), json(json)
      {
        static_assert(Components::count_fe_subsystems() > 1,
                      "LDG must have a defined submodel of the Model with index 1.");
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i) {
          fe.emplace_back(std::make_shared<FESystem<dim>>(FE_DGQ<dim>(json.get_uint("/discretization/fe_order")),
                                                          Components::count_fe_functions(i)));
          dof_handler.emplace_back(std::make_shared<DoFHandler<dim>>(mesh.get_triangulation()));
          constraints.emplace_back();
        }
        setup_dofs();
      };

      const auto &get_constraints(const uint i = 0) const { return constraints[i]; }
      auto &get_constraints(const uint i = 0)
      {
        (void)i;
        return constraints[i];
      }
      const auto &get_dof_handler(const uint i = 0) const { return *(dof_handler[i]); }
      auto &get_dof_handler(const uint i = 0) { return *(dof_handler[i]); }
      const auto get_dof_handler_list() const
      {
        std::vector<const DoFHandler<dim> *> ret;
        for (uint i = 0; i < Components::count_fe_subsystems(); ++i)
          ret.push_back(&(get_dof_handler(i)));
        return ret;
      }
      const auto &get_fe(uint i = 0) const { return *(fe[i]); }
      const auto &get_mapping() const { return mapping; }
      const auto &get_triangulation() const { return mesh.get_triangulation(); }
      auto &get_triangulation() { return mesh.get_triangulation(); }
      const Point<dim> &get_support_point(const uint &dof) const { return support_points[dof]; }
      const auto &get_support_points() const { return support_points; }
      const auto &get_json() const { return json; }

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

    protected:
      void setup_dofs()
      {
        spdlog::get("log")->info("FEM: Number of active cells: {}", mesh.get_triangulation().n_active_cells());

        for (uint i = 0; i < Components::count_fe_subsystems(); ++i) {
          dof_handler[i]->distribute_dofs(*(fe[i]));
          DoFRenumbering::component_wise(*(dof_handler[i]));
          constraints[i].clear();
          DoFTools::make_hanging_node_constraints(*(dof_handler[i]), constraints[i]);
          constraints[i].close();

          if (i == 0) spdlog::get("log")->info("FEM: Number of degrees of freedom: {}", dof_handler[0]->n_dofs());
        }

        support_points.resize(dof_handler[0]->n_dofs());
        DoFTools::map_dofs_to_support_points(mapping, *dof_handler[0], support_points);
      }

      Mesh &mesh;
      JSONValue json;

      std::vector<std::shared_ptr<FESystem<dim>>> fe;
      std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handler;
      std::vector<AffineConstraints<NumberType>> constraints;
      MappingQ1<dim> mapping;
      std::vector<Point<dim>> support_points;
    };
  } // namespace LDG
} // namespace DiFfRG