#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test implicit euler with DG constant model", "[timestepping][constant][implicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperImplicitEuler<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_constant_dg", 5e-2));
}
TEST_CASE("Test implicit euler with DG exponential model", "[timestepping][exponential][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperImplicitEuler<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_exponential_dg", 5e-2));
}
TEST_CASE("Test implicit euler with DG Burgers model", "[timestepping][Burgers][explicit_euler][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperImplicitEuler<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_implicit_euler_burgers_dg", 5e-2));
}