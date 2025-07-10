#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/models.hh>
#include <boilerplate/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test TRBDF2 with DG constant model", "[timestepping][constant][trbdf2][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperTRBDF2<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_trbdf2_constant_dg", 1e-3));
}
TEST_CASE("Test TRBDF2 with DG exponential model", "[timestepping][exponential][trbdf2][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperTRBDF2<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_trbdf2_exponential_dg", 1e-3));
}
TEST_CASE("Test TRBDF2 with DG Burgers model", "[timestepping][Burgers][trbdf2][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperTRBDF2<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_trbdf2_burgers_dg", 1e-3));
}