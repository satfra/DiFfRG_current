#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test SUNDIALS ARKODE with DG constant model", "[timestepping][constant][arkode][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_arkode_constant_dg", 1e-9));
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper, true>("test_arkode_constant_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS ARKODE with DG exponential model", "[timestepping][exponential][arkode][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_arkode_exponential_dg", 1e-9));
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper, true>("test_arkode_exponential_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS ARKODE with DG Burgers model", "[timestepping][Burgers][arkode][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_ARKode<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_arkode_burgers_dg", 1e-9));
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper, true>("test_arkode_burgers_dg", 1e-9));
}