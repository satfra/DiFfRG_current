#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/affine_constraints.h>
#include <spdlog/spdlog.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  namespace FV
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

      Discretization(Mesh &mesh, const JSONValue &json) : mesh(mesh), json(json) {};

      const auto &get_mapping() const { return mapping; }
      const auto &get_triangulation() const { return mesh.get_triangulation(); }
      auto &get_triangulation() { return mesh.get_triangulation(); }
      const auto &get_json() const { return json; }

      void reinit() {}

    protected:
      Mesh &mesh;
      JSONValue json;

      AffineConstraints<NumberType> constraints;
      MappingQ1<dim> mapping;
    };
  } // namespace FV
} // namespace DiFfRG
