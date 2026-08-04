#pragma once

// Test runners: each one constructs the model + assembler + IDA time-stepper
// stack and runs it from t=0 to target_time. Throws (via Catch's REQUIRE-on-
// exception machinery) if the time-stepper can't reach the target.
//
// Four variants:
//   run_flow_to_time<Model>         : default Assembler (MinMod + differentiated max-eigenvalue speed)
//   run_flow_to_time_ws<Model, WS>  : explicit WaveSpeedStrategy override
//   run_flow_to_time_cg<Model>      : CG discretisation (fe_order >= 1)
//   run_flow_to_time_tol<Model, ...>: solver-tolerance override

#include "kt_3D_models.hh"
#include "kt_3D_setup.hh"

#include <DiFfRG/discretization/FEM/assembler/cg.hh>
#include <DiFfRG/discretization/FEM/cg.hh>

#include <memory>

namespace on_kt_3D
{
  // Default-strategy runner: MinMod limiter, MaxEigenvalueWaveSpeed.
  // Used by every smoke and diagnostic test that doesn't sweep KT internals.
  template <typename ModelType>
  void run_flow_to_time(const GridSettings &grid, const double target_time, const int threads = 1)
  {
    const JSONValue json = make_json(threads);
    ModelType model(json, grid);
    Mesh mesh(make_mesh_config(grid));
    Discretization discretization(mesh, json, DiFfRG::LogPort{});
    Assembler<ModelType> assembler(discretization, model, json, DiFfRG::LogPort{});

    auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "on_kt_3D", "output");
    OutputSession<dim, VectorType> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == grid.cells);

    time_stepper.run(&state, 0.0, target_time);
  }

  // Wave-speed-strategy-parameterised runner for deliberately different
  // physical speed definitions. Derivatives of the standard speed are AD-backed.
  template <typename ModelType, typename WaveSpeedStrategy>
  void run_flow_to_time_ws(const GridSettings &grid, const double target_time, const int threads = 1)
  {
    using AssemblerWS = FV::KurganovTadmor::Assembler<Discretization, ModelType, Reconstructor, WaveSpeedStrategy>;

    const JSONValue json = make_json(threads);
    ModelType model(json, grid);
    Mesh mesh(make_mesh_config(grid));
    Discretization discretization(mesh, json, DiFfRG::LogPort{});
    AssemblerWS assembler(discretization, model, json, DiFfRG::LogPort{});

    auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "on_kt_3D", "output");
    OutputSession<dim, VectorType> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == grid.cells);

    time_stepper.run(&state, 0.0, target_time);
  }

  // Solver-tolerance-parameterised runner. Used by the Example-settings
  // comparison tests (abs_tol = 1e-16, rel_tol = 1e-4 vs default abs/rel = 1e-7).
  template <typename ModelType, typename WaveSpeedStrategy = FV::KurganovTadmor::MaxEigenvalueWaveSpeed>
  void run_flow_to_time_tol(const GridSettings &grid, const double target_time, const double abs_tol,
                            const double rel_tol, const int threads = 1)
  {
    using AssemblerTol = FV::KurganovTadmor::Assembler<Discretization, ModelType, Reconstructor, WaveSpeedStrategy>;

    const JSONValue json = make_json(threads, /*fe_order=*/0, /*x_order=*/32, abs_tol, rel_tol);
    ModelType model(json, grid);
    Mesh mesh(make_mesh_config(grid));
    Discretization discretization(mesh, json, DiFfRG::LogPort{});
    AssemblerTol assembler(discretization, model, json, DiFfRG::LogPort{});

    auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "on_kt_3D", "output");
    OutputSession<dim, VectorType> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    const auto &support_points = discretization.get_support_points();
    REQUIRE(support_points.size() == grid.cells);

    time_stepper.run(&state, 0.0, target_time);
  }

  // CG-discretisation runner. Used by LSM_CG; lets us run an integrator-based
  // model on the SAME physics as the KT cases for direct CG-vs-KT comparison.
  template <typename ModelType>
  void run_flow_to_time_cg(const GridSettings &grid, const double target_time, const int threads = 1,
                           const int fe_order = 4, const double abs_tol = 1.0e-7, const double rel_tol = 1.0e-7)
  {
    using CGDiscretization = CG::Discretization<Components, NumberType, Mesh>;
    using CGAssembler = CG::Assembler<CGDiscretization, ModelType>;
    using CGTimeStepper = TimeStepperSUNDIALS_IDA<typename CGDiscretization::VectorType,
                                                  typename CGDiscretization::SparseMatrixType, dim, UMFPack>;

    const JSONValue json = make_json(threads, fe_order, /*x_order=*/32, abs_tol, rel_tol);
    ModelType model(json, grid);
    Mesh mesh(make_mesh_config(grid));
    CGDiscretization discretization(mesh, json, DiFfRG::LogPort{});
    CGAssembler assembler(discretization, model, json, DiFfRG::LogPort{});

    auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "on_kt_3D_cg", "output");
    OutputSession<dim, typename CGDiscretization::VectorType> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<typename CGDiscretization::VectorType>>();
    CGTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FE::FlowingVariables state(discretization);
    state.interpolate(model);

    time_stepper.run(&state, 0.0, target_time);
  }

  // KT runner on a non-uniform (adaptive) initial grid. Tests whether KT's
  // failure on the Example's uniform grid is grid-related (vs. an intrinsic
  // KT-discretisation issue).
  template <typename ModelType, typename WaveSpeedStrategy = FV::KurganovTadmor::MaxEigenvalueWaveSpeed>
  void run_flow_to_time_kt_adaptive(const std::vector<GridSubrange> &subranges, const double target_time,
                                    const int threads = 1, const double abs_tol = 1.0e-7, const double rel_tol = 1.0e-7)
  {
    using AssemblerAd = FV::KurganovTadmor::Assembler<Discretization, ModelType, Reconstructor, WaveSpeedStrategy>;

    const JSONValue json = make_json(threads, /*fe_order=*/0, /*x_order=*/32, abs_tol, rel_tol);
    const std::size_t n_cells_in_grid = total_cells(subranges);
    GridSettings grid{n_cells_in_grid, subranges.front().min, subranges.back().max};
    ModelType model(json, grid);
    Mesh mesh(make_adaptive_mesh_config(subranges));
    Discretization discretization(mesh, json, DiFfRG::LogPort{});
    AssemblerAd assembler(discretization, model, json, DiFfRG::LogPort{});

    auto data_out_path =
        OutputPath::temporary(TemporaryRetention::remove_on_destruction, "on_kt_3D_adaptive", "output");
    OutputSession<dim, VectorType> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    ImplicitTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);

    time_stepper.run(&state, 0.0, target_time);
  }

  // CG counterpart on the adaptive grid.
  template <typename ModelType>
  void run_flow_to_time_cg_adaptive(const std::vector<GridSubrange> &subranges, const double target_time,
                                    const int threads = 1, const int fe_order = 4, const double abs_tol = 1.0e-7,
                                    const double rel_tol = 1.0e-7)
  {
    using CGDiscretization = CG::Discretization<Components, NumberType, Mesh>;
    using CGAssembler = CG::Assembler<CGDiscretization, ModelType>;
    using CGTimeStepper = TimeStepperSUNDIALS_IDA<typename CGDiscretization::VectorType,
                                                  typename CGDiscretization::SparseMatrixType, dim, UMFPack>;

    const JSONValue json = make_json(threads, fe_order, /*x_order=*/32, abs_tol, rel_tol);
    const std::size_t n_cells_in_grid = total_cells(subranges);
    GridSettings grid{n_cells_in_grid, subranges.front().min, subranges.back().max};
    ModelType model(json, grid);
    Mesh mesh(make_adaptive_mesh_config(subranges));
    CGDiscretization discretization(mesh, json, DiFfRG::LogPort{});
    CGAssembler assembler(discretization, model, json, DiFfRG::LogPort{});

    auto data_out_path =
        OutputPath::temporary(TemporaryRetention::remove_on_destruction, "on_kt_3D_cg_adaptive", "output");
    OutputSession<dim, typename CGDiscretization::VectorType> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<typename CGDiscretization::VectorType>>();
    CGTimeStepper time_stepper(json, &assembler, &data_out, adaptor.get());

    FE::FlowingVariables state(discretization);
    state.interpolate(model);

    time_stepper.run(&state, 0.0, target_time);
  }

} // namespace on_kt_3D
