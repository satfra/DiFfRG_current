#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <boilerplate/kt_models.hh>
#include <boilerplate/timestepping.hh>

using namespace DiFfRG;

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test SUNDIALS IDA with DG constant model", "[timestepping][constant][sundials_ida][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_constant_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with DG exponential model", "[timestepping][exponential][sundials_ida][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_exponential_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with DG Burgers model", "[timestepping][Burgers][sundials_ida][dg]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = DG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = DG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_burgers_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with CG constant model", "[timestepping][constant][sundials_ida][fem]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = CG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_constant_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with CG exponential model", "[timestepping][exponential][sundials_ida][fem]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelExp<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = CG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_exponential_dg", 1e-9));
}
TEST_CASE("Test SUNDIALS IDA with CG Burgers model", "[timestepping][Burgers][sundials_ida][fem]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgers<dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = CG::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_burgers_dg", 1e-9));
}

TEST_CASE("Test SUNDIALS IDA with KT Burgers model", "[timestepping][Burgers][sundials_ida][kt]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_burgers_kt", 1e-9));
}

TEST_CASE("Test SUNDIALS IDA with KT viscous Burgers traveling wave",
          "[timestepping][Burgers][sundials_ida][kt][traveling_wave]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelBurgersTravelingWaveKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_traveling_wave_kt", 5e-7));
}

TEST_CASE("Test SUNDIALS IDA with KT two-component Burgers system",
          "[timestepping][Burgers][sundials_ida][kt][two_component]")
{
  constexpr uint dim = 1;
  using Model = Testing::ModelTwoComponentBurgersKT<dim>;
  using NumberType = double;
  using Discretization = FV::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>;
  REQUIRE(run<Model, Discretization, Assembler, TimeStepper>("test_sundials_ida_two_component_burgers_kt", 5e-7));
}
