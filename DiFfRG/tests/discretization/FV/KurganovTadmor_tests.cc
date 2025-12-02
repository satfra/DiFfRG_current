#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/discretization/mesh/configuration_mesh.hh"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <algorithm>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <iterator>
#include <oneapi/tbb/parallel_for_each.h>
#include <petscvec.h>
#include <vector>

const int dim = 1;
using NumberType = double;
using VectorType = dealii::Vector<NumberType>;
using namespace dealii;
using namespace DiFfRG::FV::KurganovTadmor;
using DiFfRG::FV::KurganovTadmor::internal::Cache_Data;
using DiFfRG::FV::KurganovTadmor::internal::copy_from_cell_to_array;
using DiFfRG::FV::KurganovTadmor::internal::GhostLayer;
using DiFfRG::FV::KurganovTadmor::internal::ScratchData;
struct CopyData {
};

TEST_CASE("Initialize cache data and neighbors", "[KT]")
{
  const auto grid_axis = std::vector<DiFfRG::Config::GridAxis>{{0.0, 0.2, 1.0}};
  const auto mesh_config = DiFfRG::Config::ConfigurationMesh<dim>(0, grid_axis);
  const auto rectangular_mesh = DiFfRG::RectangularMesh<dim>(mesh_config);
  auto dof_handler = DoFHandler<1>(rectangular_mesh.get_triangulation());
  const VectorType solution = VectorType{0.0, 1.0, 2.0, 3.0, 4.0};
  const auto fe_system = FESystem<1>(FE_DGQ<1>(0), 1);
  dof_handler.distribute_dofs(fe_system);
  auto mapping = MappingQ1<dim>();
  auto quadrature = QGauss<dim>(1);
  ScratchData<dim, NumberType> scratch_data(mapping, fe_system, quadrature, solution);
  const auto copier = [&](const auto & /* c */) {};
  std::vector<Cache_Data<dim, NumberType>> cache_data(rectangular_mesh.get_triangulation().n_active_cells());
  const auto cell_worker = copy_from_cell_to_array<1, NumberType, CopyData>(cache_data);
  MeshWorker::mesh_loop(dof_handler.active_cell_iterators(), cell_worker, copier, scratch_data, CopyData(),
                        MeshWorker::assemble_own_cells, nullptr, nullptr, 1, 1);

  // the vector does not has to be orderd, this is done here only for easier checking
  std::sort(cache_data.begin(), cache_data.end(), [](auto a, auto b) { return a.position[0] < b.position[0]; });
  CHECK(cache_data[0].position[0] == Catch::Approx(0.1));
  CHECK(cache_data[1].position[0] == Catch::Approx(0.3));
  CHECK(cache_data[2].position[0] == Catch::Approx(0.5));
  CHECK(cache_data[3].position[0] == Catch::Approx(0.7));
  CHECK(cache_data[4].position[0] == Catch::Approx(0.9));

  CHECK(cache_data[0].u == Catch::Approx(0.0));
  CHECK(cache_data[1].u == Catch::Approx(1.0));
  CHECK(cache_data[2].u == Catch::Approx(2.0));
  CHECK(cache_data[3].u == Catch::Approx(3.0));
  CHECK(cache_data[4].u == Catch::Approx(4.0));

  CHECK(!cache_data[0].left_neighbor.has_value());
  CHECK(cache_data[0].right_neighbor->get().position == cache_data[1].position);
  CHECK(cache_data[1].left_neighbor->get().position == cache_data[0].position);
  CHECK(cache_data[1].right_neighbor->get().position == cache_data[2].position);
  CHECK(cache_data[2].left_neighbor->get().position == cache_data[1].position);
  CHECK(cache_data[2].right_neighbor->get().position == cache_data[3].position);
  CHECK(cache_data[3].left_neighbor->get().position == cache_data[2].position);
  CHECK(cache_data[3].right_neighbor->get().position == cache_data[4].position);
  CHECK(cache_data[4].left_neighbor->get().position == cache_data[3].position);
  CHECK(!cache_data[4].right_neighbor.has_value());
}

class CacheDataWithNeighborsFixture
{
protected:
  std::vector<Cache_Data<1, NumberType>> cache_data;

public:
  CacheDataWithNeighborsFixture() : cache_data(4)
  {
    cache_data[0].position = Point<dim>{1.0};
    cache_data[0].u = 1.0;
    cache_data[1].position = Point<dim>{3.0};
    cache_data[1].u = 9.0;
    cache_data[2].position = Point<dim>{5.0};
    cache_data[2].u = 25.0;
    cache_data[3].position = Point<dim>{7.0};
    cache_data[3].u = 49.0;
    cache_data[0].right_neighbor = std::ref(cache_data[1]);
    cache_data[1].left_neighbor = std::ref(cache_data[0]);
    cache_data[1].right_neighbor = std::ref(cache_data[2]);
    cache_data[2].left_neighbor = std::ref(cache_data[1]);
    cache_data[2].right_neighbor = std::ref(cache_data[3]);
    cache_data[3].left_neighbor = std::ref(cache_data[2]);
  };
};

TEST_CASE_METHOD(CacheDataWithNeighborsFixture, "Generate boundary ghost cells", "[KT]")
{
  Cache_Data<1, NumberType> left_left_boundary_cell, left_boundary_cell;
  DiFfRG::FV::KurganovTadmor::LeftAntisymmetricBoundary(cache_data, left_left_boundary_cell, left_boundary_cell);

  Cache_Data<1, NumberType> right_boundary_cell, right_right_boundary_cell;
  DiFfRG::FV::KurganovTadmor::RightExtrapolationBoundary(cache_data, right_boundary_cell, right_right_boundary_cell);

  const size_t N = cache_data.size();

  CHECK(left_boundary_cell.u == -1.0);
  CHECK(left_left_boundary_cell.u == -9.0);
  CHECK(right_boundary_cell.u == 73.0);
  CHECK(right_right_boundary_cell.u == 97.0);

  CHECK(left_boundary_cell.position[0] == -1.0);
  CHECK(left_left_boundary_cell.position[0] == -3.0);
  CHECK(right_boundary_cell.position[0] == 9.0);
  CHECK(right_right_boundary_cell.position[0] == 11.0);

  CHECK(cache_data[0].left_neighbor->get().position[0] == left_boundary_cell.position[0]);
  CHECK(left_boundary_cell.right_neighbor->get().position[0] == cache_data[0].position[0]);
  CHECK(left_boundary_cell.left_neighbor->get().position[0] == left_left_boundary_cell.position[0]);
  CHECK(left_left_boundary_cell.right_neighbor->get().position[0] == left_boundary_cell.position[0]);
  CHECK(!left_left_boundary_cell.left_neighbor.has_value());

  CHECK(cache_data[N - 1].right_neighbor->get().position[0] == right_boundary_cell.position[0]);
  CHECK(right_boundary_cell.left_neighbor->get().position[0] == cache_data[N - 1].position[0]);
  CHECK(right_boundary_cell.right_neighbor->get().position[0] == right_right_boundary_cell.position[0]);
  CHECK(right_right_boundary_cell.left_neighbor->get().position[0] == right_boundary_cell.position[0]);
  CHECK(!right_right_boundary_cell.right_neighbor.has_value());
};

TEST_CASE_METHOD(CacheDataWithNeighborsFixture, "Access data via GhostLayer", "[KT]")
{
  GhostLayer<dim, NumberType> ghost_layer(cache_data,
                                          DiFfRG::FV::KurganovTadmor::LeftAntisymmetricBoundary<dim, NumberType>,
                                          DiFfRG::FV::KurganovTadmor::RightExtrapolationBoundary<dim, NumberType>);

  CHECK(ghost_layer[0].position[0] == -3.0);
  CHECK(ghost_layer[1].position[0] == -1.0);
  CHECK(ghost_layer[2].position[0] == 1.0);
  CHECK(ghost_layer[3].position[0] == 3.0);
  CHECK(ghost_layer[4].position[0] == 5.0);
  CHECK(ghost_layer[5].position[0] == 7.0);
  CHECK(ghost_layer[6].position[0] == 9.0);
  CHECK(ghost_layer[7].position[0] == 11.0);

  CHECK(ghost_layer[0].u == Catch::Approx(-9.0));
  CHECK(ghost_layer[1].u == Catch::Approx(-1.0));
  CHECK(ghost_layer[2].u == Catch::Approx(1.0));
  CHECK(ghost_layer[3].u == Catch::Approx(9.0));
  CHECK(ghost_layer[4].u == Catch::Approx(25.0));
  CHECK(ghost_layer[5].u == Catch::Approx(49.0));
  CHECK(ghost_layer[6].u == Catch::Approx(73.0));
  CHECK(ghost_layer[7].u == Catch::Approx(97.0));
}

TEST_CASE_METHOD(CacheDataWithNeighborsFixture, "Compute intermediate derivatives", "[KT]")
{
  GhostLayer<dim, NumberType> ghost_layer(cache_data,
                                          DiFfRG::FV::KurganovTadmor::LeftAntisymmetricBoundary<dim, NumberType>,
                                          DiFfRG::FV::KurganovTadmor::RightExtrapolationBoundary<dim, NumberType>);

  auto functor_val = DiFfRG::FV::KurganovTadmor::internal::compute_intermediate_derivates<1, NumberType>();

  ghost_layer.execute_parallel_function(functor_val);

  CHECK(cache_data[0].du_dx_half == 4.0);  // (9 - 1) / (3 - 1)
  CHECK(cache_data[1].du_dx_half == 8.0);  // (25 - 9) / (5 - 3)
  CHECK(cache_data[2].du_dx_half == 12.0); // (49 - 25) / (7 - 5)
  CHECK(cache_data[3].du_dx_half == 12.0); // linear extrapolation at the right boundary
}
