#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/kt_models.hh>
#include <boilerplate/models.hh>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/FEM/cg.hh>
#include <DiFfRG/discretization/FEM/dg.hh>
#include <DiFfRG/discretization/FV/assembler/KurganovTadmor.hh>
#include <DiFfRG/discretization/FV/discretization.hh>
#include <DiFfRG/discretization/data/data.hh>
#include <DiFfRG/model/model.hh>

#include <vector>

// Distributed assembly: does an n-rank residual equal the serial one?
//
// This is the gate that catches the failure mode with no error message. On a partitioned mesh
// an interior face is visited only when *both* neighbours are locally owned unless
// assemble_ghost_faces_once is set, so a face between an owned and a ghost cell is assembled by
// neither rank and every partition-boundary flux silently vanishes. The answer stays finite,
// smooth and plausible -- it is just wrong.
//
// The comparison is made inside one process: every rank builds a serial-policy discretization
// alongside the distributed one and assembles both. That keeps the test to a couple of seconds
// (no timestepping, no output, no cross-run file comparison) and removes the serial reference
// as a source of drift.
//
// The file must contribute at least one test case in a serial build: catch_discover_tests
// registers a <target>_NOT_BUILT placeholder that is only replaced once cases are discovered.
// The DG refinement-indicator case below is that, and is a regression test in its own right.

namespace
{
  using namespace DiFfRG;

  /// 16 cells on [0, 1] at fe_order 2 -- small enough to stay fast at 4 ranks, big enough that
  /// every rank owns several cells and there are real partition boundaries to get wrong.
  ///
  /// The integer literals here matter: boost::json stores an `unsigned` in a different arm than
  /// an `int`, and ConfigTree::get_uint used to throw on the unsigned one, so passing fe_order
  /// as an `unsigned` made the key silently unreadable and the discretization fell back to its
  /// default order. That is fixed in config_tree.cc now, but write plain ints regardless.
  ConfigTree make_json()
  {
    return json::value(
        {{"physical", {}},
         {"integration", {{"x_quadrature_order", 8}, {"x_extent_tolerance", 1e-3}}},
         {"discretization",
          {{"fe_order", 2},
           {"mesh_workers", 0},
           {"batch_size", 16},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", "0:0.0625:1"}, {"y_grid", "0:0.5:1"}, {"z_grid", "0:0.5:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 5e-2}}}}},
         {"timestepping", {{"final_time", 1.}, {"output_dt", 1e-1}}},
         {"output", {{"live_plot", false}, {"verbosity", 0}}}});
  }

  /// u(x) = 0.2 + x, i.e. a state that is neither zero nor constant.
  ///
  /// PhysicalParameters holds four std::array<double, 3>, so the obvious
  /// `PhysicalParameters{0., 1.}` does NOT mean "x0 = 0, x1 = 1": brace elision fills
  /// initial_x0 with {0, 1, 0} and leaves initial_x1 at zero, giving u == 0 everywhere. With a
  /// zero state ModelBurgers' flux is zero, the residual is zero, and a residual comparison
  /// compares zero against zero and passes no matter what the assembler does. Set the fields by
  /// name instead.
  Testing::PhysicalParameters nontrivial_parameters()
  {
    Testing::PhysicalParameters prm;
    prm.initial_x0 = {{0.2, 0., 0.}};
    prm.initial_x1 = {{1., 0., 0.}};
    return prm;
  }

  /// A model whose face_indicator is non-zero and whose cell_indicator is left at the default
  /// (zero), so refinement_indicator is non-zero if and only if faces were actually visited.
  template <uint dim>
  class FaceOnlyIndicatorModel : public Testing::ModelBurgers<dim>
  {
  public:
    using Testing::ModelBurgers<dim>::ModelBurgers;

    template <int mdim, typename NumberType, typename Solutions_s, typename Solutions_n>
    void face_indicator(std::array<NumberType, 2> &indicator, const Tensor<1, mdim> & /*normal*/,
                        const Point<mdim> & /*p*/, const Solutions_s & /*sol_s*/,
                        const Solutions_n & /*sol_n*/) const
    {
      indicator[0] = 1.;
      indicator[1] = 1.;
    }
  };
} // namespace

TEST_CASE("DG refinement indicator sees face contributions", "[discretization][dg]")
{
  DiFfRG::Init();
  using namespace dealii;
  constexpr uint dim = 1;
  using Model = FaceOnlyIndicatorModel<dim>;
  using DGDiscretization = DG::Discretization<typename Model::Components, double, RectangularMesh<dim>>;
  using Assembler = DG::Assembler<DGDiscretization, Model>;

  const auto json = make_json();
  Model model(nontrivial_parameters());
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  DGDiscretization discretization(mesh, json, DiFfRG::LogPort{});
  Assembler assembler(discretization, model, json, DiFfRG::LogPort{});

  FE::FlowingVariables<DGDiscretization> initial_condition(discretization);
  initial_condition.interpolate(model);

  Vector<double> indicator(mesh.get_triangulation().n_active_cells());
  assembler.refinement_indicator(indicator, initial_condition.spatial_data());

  // Regression: dg.hh passed a face_worker to mesh_loop but omitted
  // assemble_own_interior_faces_once from the flags, so mesh_loop never called it and
  // model.face_indicator contributed exactly nothing -- DG refined on the cell term alone.
  // With a model whose only indicator contribution is the face one, that bug makes every
  // entry zero. ddg.hh and ldg.hh always set the flag.
  REQUIRE(indicator.linfty_norm() > 0.);
}

#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_PETSC)

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

#include <DiFfRG/discretization/common/solution_view.hh>

namespace
{
  template <uint dim> using ParallelMesh = RectangularMesh<dim, dealii::parallel::shared::Triangulation<dim>>;

  /**
   * @brief Assemble one residual serially and one distributed, and require them to agree.
   *
   * Matching is done cell by cell rather than by global index: a DoFHandler on a
   * parallel::shared::Triangulation renumbers so that each rank owns a contiguous range, so
   * global index i denotes different nodes in the two handlers. The meshes are structurally
   * identical, so cells correspond by (level, index).
   */
  template <typename SerialAssembler, typename SerialDisc, typename ParallelAssembler, typename ParallelDisc,
            typename SerialVars, typename ParallelVars, typename Model>
  void require_distributed_residual_matches_serial(Model &model, const ConfigTree &json)
  {
    using namespace dealii;
    constexpr uint dim = SerialDisc::dim;

    RectangularMesh<dim> serial_mesh{Config::ConfigurationMesh<dim>(json)};
    SerialDisc serial_disc(serial_mesh, json, DiFfRG::LogPort{});
    SerialAssembler serial_assembler(serial_disc, model, json, DiFfRG::LogPort{});
    SerialVars serial_ic(serial_disc);
    serial_ic.interpolate(model);
    typename SerialDisc::VectorType serial_residual(serial_ic.spatial_data());
    serial_residual = 0.;
    serial_assembler.residual(serial_residual, serial_ic.spatial_data(), 1., serial_ic.spatial_data(), 1.);

    ParallelMesh<dim> parallel_mesh{Config::ConfigurationMesh<dim>(json)};
    ParallelDisc parallel_disc(parallel_mesh, json, DiFfRG::LogPort{});
    ParallelAssembler parallel_assembler(parallel_disc, model, json, DiFfRG::LogPort{});
    ParallelVars parallel_ic(parallel_disc);
    parallel_ic.interpolate(model);

    using ParVector = typename ParallelDisc::VectorType;
    // The assemblers read every dof of a cell they own, including ones another rank owns, so the
    // state has to be handed over as a fully replicated view -- exactly what the timestepper does.
    SolutionView<ParVector> state;
    state.reinit(parallel_disc.get_locally_owned_dofs(), parallel_disc.get_locally_relevant_dofs(),
                 parallel_disc.get_communicator());
    state.refresh(parallel_ic.spatial_data());

    ParVector parallel_residual(parallel_ic.spatial_data());
    parallel_residual = 0.;
    parallel_assembler.residual(parallel_residual, state.get(), 1., state.get(), 1.);

    SolutionView<ParVector> gathered;
    gathered.reinit(parallel_disc.get_locally_owned_dofs(), parallel_disc.get_locally_relevant_dofs(),
                    parallel_disc.get_communicator());
    gathered.refresh(parallel_residual);

    const auto &serial_dh = serial_disc.get_dof_handler();
    const auto &parallel_dh = parallel_disc.get_dof_handler();
    REQUIRE(serial_dh.n_dofs() == parallel_dh.n_dofs());

    const double scale = std::max(serial_residual.linfty_norm(), 1e-300);
    std::vector<types::global_dof_index> serial_indices, parallel_indices;
    double worst = 0.;
    for (const auto &parallel_cell : parallel_dh.active_cell_iterators()) {
      const typename DoFHandler<dim>::active_cell_iterator serial_cell(
          &serial_dh.get_triangulation(), parallel_cell->level(), parallel_cell->index(), &serial_dh);
      const auto n = parallel_cell->get_fe().n_dofs_per_cell();
      serial_indices.resize(n);
      parallel_indices.resize(n);
      parallel_cell->get_dof_indices(parallel_indices);
      serial_cell->get_dof_indices(serial_indices);
      for (unsigned int k = 0; k < n; ++k)
        worst = std::max(worst, std::abs(gathered[parallel_indices[k]] - serial_residual[serial_indices[k]]));
    }

    // Guard against the test quietly becoming a no-op. A comparison against a zero reference
    // passes for every possible assembler bug, and that is exactly what happened here once
    // already -- the state was degenerate, so both residuals were identically zero and stripping
    // assemble_ghost_faces_once did not make the test fail. Require a reference with something
    // in it before believing the agreement below.
    REQUIRE(scale > 1e-8);

    // Distributed assembly sums off-process row contributions in MPI arrival order, so this is
    // not bitwise -- but a dropped partition-boundary flux is an O(1) error, not an O(eps) one.
    INFO("worst |distributed - serial| = " << worst << ", residual scale = " << scale);
    REQUIRE(worst < 1e-10 * scale);
  }
} // namespace

TEST_CASE("CG residual is unchanged by distribution", "[discretization][cg][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using Components = typename Model::Components;
  using SerialDisc = CG::Discretization<Components, double, RectangularMesh<dim>>;
  using ParallelDisc = CG::Discretization<Components, double, ParallelMesh<dim>, dealii::PETScWrappers::MPI::Vector,
                                          dealii::PETScWrappers::MPI::SparseMatrix>;
  Model model(nontrivial_parameters());
  require_distributed_residual_matches_serial<CG::Assembler<SerialDisc, Model>, SerialDisc,
                                              CG::Assembler<ParallelDisc, Model>, ParallelDisc,
                                              FE::FlowingVariables<SerialDisc>,
                                              FE::FlowingVariables<ParallelDisc>>(model, make_json());
}

TEST_CASE("DG residual is unchanged by distribution", "[discretization][dg][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using Components = typename Model::Components;
  using SerialDisc = DG::Discretization<Components, double, RectangularMesh<dim>>;
  using ParallelDisc = DG::Discretization<Components, double, ParallelMesh<dim>, dealii::PETScWrappers::MPI::Vector,
                                          dealii::PETScWrappers::MPI::SparseMatrix>;
  Model model(nontrivial_parameters());
  require_distributed_residual_matches_serial<DG::Assembler<SerialDisc, Model>, SerialDisc,
                                              DG::Assembler<ParallelDisc, Model>, ParallelDisc,
                                              FE::FlowingVariables<SerialDisc>,
                                              FE::FlowingVariables<ParallelDisc>>(model, make_json());
}

TEST_CASE("dDG residual is unchanged by distribution", "[discretization][ddg][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using Components = typename Model::Components;
  using SerialDisc = DG::Discretization<Components, double, RectangularMesh<dim>>;
  using ParallelDisc = DG::Discretization<Components, double, ParallelMesh<dim>, dealii::PETScWrappers::MPI::Vector,
                                          dealii::PETScWrappers::MPI::SparseMatrix>;
  Model model(nontrivial_parameters());
  require_distributed_residual_matches_serial<dDG::Assembler<SerialDisc, Model>, SerialDisc,
                                              dDG::Assembler<ParallelDisc, Model>, ParallelDisc,
                                              FE::FlowingVariables<SerialDisc>,
                                              FE::FlowingVariables<ParallelDisc>>(model, make_json());
}

TEST_CASE("KT residual is unchanged by distribution", "[discretization][kt][mpi]")
{
  DiFfRG::Init();
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersKT<dim>;
  using Components = typename Model::Components;
  using SerialDisc = FV::Discretization<Components, double, RectangularMesh<dim>>;
  using ParallelDisc = FV::Discretization<Components, double, ParallelMesh<dim>, dealii::PETScWrappers::MPI::Vector,
                                          dealii::PETScWrappers::MPI::SparseMatrix>;
  Model model(nontrivial_parameters());
  require_distributed_residual_matches_serial<FV::KurganovTadmor::Assembler<SerialDisc, Model>, SerialDisc,
                                              FV::KurganovTadmor::Assembler<ParallelDisc, Model>, ParallelDisc,
                                              FV::FlowingVariables<SerialDisc>,
                                              FV::FlowingVariables<ParallelDisc>>(model, make_json());
}

#endif // DEAL_II_WITH_MPI && DEAL_II_WITH_PETSC
