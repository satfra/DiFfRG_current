#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test explicit euler with DG constant model", "[timestepping][constant][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperExplicitEuler<VectorType, SparseMatrixType, dim>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_explicit_euler_constant_dg", 1e-4));
}
TEST_CASE("Test explicit euler with DG exponential model", "[timestepping][exponential][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperExplicitEuler<VectorType, SparseMatrixType, dim>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_explicit_euler_exponential_dg", 1e-4));
}
TEST_CASE("Test explicit euler with DG Burgers model", "[timestepping][Burgers][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperExplicitEuler<VectorType, SparseMatrixType, dim>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_explicit_euler_burgers_dg", 1e-4));
}