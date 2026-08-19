#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
// #include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/run_logger.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/assembler/ddg.hh>
#include <DiFfRG/discretization/FEM/assembler/dg.hh>
#include <DiFfRG/discretization/common/parallel_dofs.hh>
#include <DiFfRG/discretization/discretization.hh>

// std
#include <type_traits>

namespace DiFfRG
{
  namespace DG
  {
    using namespace dealii;

    /**
     * @brief Class to manage the system on which we solve, i.e. fe spaces, grids, etc.
     * This class is a System for DG systems, i.e. without LDG.
     *
     * @tparam Model_ The Model class used for the Simulation
     */
    template <typename Components_, typename NumberType_, typename Mesh_,
              typename VectorType_ = dealii::Vector<NumberType_>,
              typename SparseMatrixType_ = dealii::SparseMatrix<NumberType_>>
    class Discretization
    {
    public:
      using Components = Components_;
      using NumberType = NumberType_;
      using VectorType = VectorType_;
      using SparseMatrixType = SparseMatrixType_;
      using Mesh = Mesh_;
      static constexpr uint dim = Mesh::dim;

      /**
       * @brief Whether the linear algebra is distributed over MPI ranks.
       */
      static constexpr bool is_distributed = !std::is_same_v<VectorType, dealii::Vector<NumberType>>;

      // A distributed vector partitions its rows by rank ownership, which only exists once the
      // mesh has been partitioned. Pairing one with a serial triangulation compiles cleanly and
      // then hands every rank the same complete index set, so every rank owns every row and the
      // assembled residual is summed n_ranks times -- a wrong answer with no error anywhere.
      static_assert(!is_distributed || Mesh::is_parallel,
                    "A distributed VectorType requires a partitioned mesh, i.e. "
                    "RectangularMesh<dim, dealii::parallel::shared::Triangulation<dim>>.");

      [[deprecated("Pass output.log_port() or an intentional LogPort{}")]] Discretization(
          Mesh &mesh, DiFfRG::internal::LegacyDefaultLogPortArgument<Mesh, ConfigTree> config)
          : Discretization(mesh, config.value(), DiFfRG::internal::legacy_default_log_port<Mesh>())
      {
      }

      Discretization(Mesh &mesh, const ConfigTree &config, LogPort log_port)
          : mesh(mesh), config(config), log_port(std::move(log_port)),
            fe(std::make_shared<FESystem<dim>>(FE_DGQ<dim>(config.get_uint_or_warn("/discretization/fe_order", 3)),
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
      const auto &get_fe(uint i = 0) const
      {
        if (i != 0) throw std::runtime_error("Wrong FE index");
        return *fe;
      }
      const auto &get_mapping() const { return mapping; }
      const auto &get_triangulation() const { return mesh.get_triangulation(); }
      auto &get_triangulation() { return mesh.get_triangulation(); }

      /**
       * @brief Get the support point for a given dof.
       *
       * @param dof The dof index.
       * @return const dealii::Point<dim>& The support point for the given dof.
       */
      const Point<dim> &get_support_point(const uint dof) const { return support_points[dof]; }
      /**
       * @brief Get the support points for all dofs.
       *
       * @return const std::vector<Point<dim>>& The support points for all dofs.
       */
      const auto &get_support_points() const { return support_points; }

      const auto &get_config() const { return config; }

      /**
       * @brief The dof rows this rank owns. Complete on a serial mesh.
       */
      const dealii::IndexSet &get_locally_owned_dofs() const { return locally_owned_dofs; }
      /**
       * @brief The dof rows this rank can read. See ParallelDoFs::make_ghost_set for why this is
       * everything at the replicated-mesh rung rather than the one-ring relevant set.
       */
      const dealii::IndexSet &get_locally_relevant_dofs() const { return locally_relevant_dofs; }
      /**
       * @brief MPI_COMM_SELF on a serial mesh, so callers never have to branch on the build type.
       */
      MPI_Comm get_communicator() const { return dof_handler.get_mpi_communicator(); }

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
      void setup_dofs()
      {
        dof_handler.distribute_dofs(*fe);

        log_port.info("FEM: Number of active cells: {}", mesh.get_triangulation().n_active_cells());
        log_port.info("FEM: Number of degrees of freedom: {}", dof_handler.n_dofs());

        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        locally_relevant_dofs = ParallelDoFs::make_ghost_set(dof_handler);

        // NOT DoFTools::map_dofs_to_support_points(..., std::vector<Point>&): that overload is
        // documented to *error out* on a DoFHandler built on a parallel::TriangulationBase,
        // because its precondition is an array sized to the global dof count. See
        // ParallelDoFs::build_support_points.
        ParallelDoFs::build_support_points(mapping, dof_handler, support_points);
      }

      Mesh &mesh;
      ConfigTree config;
      LogPort log_port;

      std::shared_ptr<FESystem<dim>> fe;
      DoFHandler<dim> dof_handler;
      AffineConstraints<NumberType> constraints;
      MappingQ1<dim> mapping;
      std::vector<Point<dim>> support_points;
      dealii::IndexSet locally_owned_dofs;
      dealii::IndexSet locally_relevant_dofs;
    };
  } // namespace DG
} // namespace DiFfRG
