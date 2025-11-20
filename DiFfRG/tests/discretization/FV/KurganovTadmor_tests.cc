#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/discretization/mesh/configuration_mesh.hh"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <algorithm>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <petscvec.h>
#include <vector>

using NumberType = double;
using VectorType = dealii::Vector<NumberType>;
using namespace dealii;
using namespace DiFfRG::FV::KurganovTadmor;
using DiFfRG::FV::KurganovTadmor::internal::Cache_Data;
using DiFfRG::FV::KurganovTadmor::internal::copy_from_cell_to_array;
using DiFfRG::FV::KurganovTadmor::internal::ScratchData;
struct CopyData {
};

TEST_CASE("Test Cache Data Initialization", "[KT]")
{
  const int dim = 1;
  const auto grid_axis = std::vector<DiFfRG::Config::GridAxis>{{0.0, 0.2, 1.0}};
  const auto mesh_config = DiFfRG::Config::ConfigurationMesh<1>(0, grid_axis);
  const auto rectangular_mesh = DiFfRG::RectangularMesh<1>(mesh_config);
  auto dof_handler = DoFHandler<1>(rectangular_mesh.get_triangulation());
  const VectorType solution = VectorType{0.0, 1.0, 2.0, 3.0, 4.0};
  const auto fe_system = FESystem<1>(FE_DGQ<1>(0), 1);
  dof_handler.distribute_dofs(fe_system);
  auto mapping = MappingQ1<1>();
  auto quadrature = QGauss<1>(1);
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
