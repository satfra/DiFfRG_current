#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/model/model.hh"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include <DiFfRG/common/json.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <autodiff/forward/real.hpp>
#include <boilerplate/kt_models.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deal.II/base/numbers.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <filesystem>
#include <limits>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

// #include <petscvec.h>

using NumberType = double;
using VectorType = dealii::Vector<NumberType>;
using namespace dealii;
namespace KT = DiFfRG::FV::KurganovTadmor;
using WaveSpeedStrategy = DiFfRG::FV::KurganovTadmor::MaxEigenvalueWaveSpeed;
using KT::internal::compute_numerical_flux;
using KT::internal::reconstruct_u;

template <typename Assembler, typename Vector>
concept CanMakeAssemblyContextViewFromTemporaryCache = requires(const Assembler &assembler, const Vector &solution) {
  assembler.make_assembly_context_view(
      assembler.template build_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution));
};

struct CopyData {
};

using FEFunctionDesc = DiFfRG::FEFunctionDescriptor<DiFfRG::Scalar<"u">>;
using Components = DiFfRG::ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

class TestModel : public DiFfRG::def::AbstractModel<TestModel, Components>
{
public:
  template <int dim, typename NumberType, typename Solutions, size_t n_fe_functions>
  static void
  KurganovTadmor_advection_flux([[maybe_unused]] std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i,
                                [[maybe_unused]] const Point<dim> &x, [[maybe_unused]] const Solutions &sol)
  {
    auto u = get<"fe_functions">(sol);
    F_i[idxf("u")][0] = u[0] * u[0] / 2.0 + x[0];
  }
};

class ResidualContributionModel : public DiFfRG::def::AbstractModel<ResidualContributionModel, Components>,
                                  public DiFfRG::def::Time,
                                  public DiFfRG::def::LLFFlux<ResidualContributionModel>,
                                  public DiFfRG::def::FlowBoundaries<ResidualContributionModel>,
                                  public DiFfRG::def::FVDefaultBoundaries<ResidualContributionModel>,
                                  public DiFfRG::def::AD<ResidualContributionModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.25 * pos[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const { return {1.0 + 0.25 * pos[0]}; }

  template <typename NT, typename Solution>
  void KurganovTadmor_advection_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/,
                                     const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    F_i[0][0] = 0.5 * u[0] * u[0];
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &du = get<"fe_derivatives">(sol);
    F_i[0][0] = 0.1 * du[0][0];
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, 1> &s_i, const Point<1> &pos, const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    s_i[0] = 0.5 + pos[0] + 0.25 * u[0];
  }

  template <typename NT, typename Vector, typename VectorDot>
  void mass(std::array<NT, 1> &m_i, const Point<1> & /*pos*/, const Vector &u, const VectorDot &dt_u) const
  {
    m_i[0] = dt_u[0] + 0.125 * u[0];
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(DiFfRG::def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              DiFfRG::def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    DiFfRG::Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                               [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }
};

class FaceHookProbeModel : public DiFfRG::def::AbstractModel<FaceHookProbeModel, Components>,
                           public DiFfRG::def::Time,
                           public DiFfRG::def::LLFFlux<FaceHookProbeModel>,
                           public DiFfRG::def::FlowBoundaries<FaceHookProbeModel>,
                           public DiFfRG::def::FVDefaultBoundaries<FaceHookProbeModel>,
                           public DiFfRG::def::AD<FaceHookProbeModel>
{
public:
  template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
  {
    values[0] = 1.0 + 0.5 * pos[0];
  }

  std::array<double, 1> solution(const Point<1> &pos) const { return {1.0 + 0.5 * pos[0]}; }

  template <KT::HasAssemblyContextView Context>
  void fv_kt_pre_assembly(const KT::AssemblyStage stage, const Context &context)
  {
    ++hook_calls;
    hook_stage = stage;
    hook_face_count = 0;
    hook_cell_count = 0;
    saw_boundary_face = false;
    saw_interior_face = false;
    hook_ready = false;

    hook_min_u_plus = std::numeric_limits<double>::infinity();
    for (const auto face : context.faces()) {
      ++hook_face_count;
      saw_boundary_face = saw_boundary_face || face.at_boundary();
      saw_interior_face = saw_interior_face || !face.at_boundary();
      const auto &reconstruction = face.reconstruction();
      last_u_minus = reconstruction.u_minus[0];
      last_u_plus = reconstruction.u_plus[0];
      last_grad_minus = reconstruction.face_grad_minus[0][0];
      last_grad_plus = reconstruction.face_grad_plus[0][0];
      hook_min_u_plus = std::min(hook_min_u_plus, static_cast<double>(reconstruction.u_plus[0]));
    }

    const auto &faces = context.faces();
    hook_parallel_min_u_plus = tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, faces.size()), std::numeric_limits<double>::infinity(),
        [&](const tbb::blocked_range<std::size_t> &range, double local_min) {
          for (std::size_t i = range.begin(); i != range.end(); ++i) {
            const auto face = faces[i];
            local_min = std::min(local_min, static_cast<double>(face.reconstruction().u_plus[0]));
          }
          return local_min;
        },
        [](const double left, const double right) { return std::min(left, right); });

    hook_min_cell_u = std::numeric_limits<double>::infinity();
    const auto &cells = context.cells();
    for (std::size_t i = 0; i < cells.size(); ++i) {
      const auto cell = cells[i];
      ++hook_cell_count;
      last_cell_u = cell.stencil().cell.u[0];
      last_cell_point = cell.point()[0];
      hook_min_cell_u = std::min(hook_min_cell_u, static_cast<double>(cell.stencil().cell.u[0]));
    }

    hook_flux_scale = hook_min_u_plus;
    hook_ready = true;
  }

  template <typename NT, typename Solution>
  void KurganovTadmor_advection_flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/,
                                     const Solution & /*sol*/) const
  {
    F_i[0][0] = 0.0;
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, 1, NT>, 1> &F_i, const Point<1> & /*pos*/, const Solution &sol) const
  {
    const auto &du = get<"fe_derivatives">(sol);
    ++flux_calls;
    flux_saw_hook_state = flux_saw_hook_state || hook_ready;
    F_i[0][0] = NT(hook_flux_scale) * du[0][0];
  }

  template <int mdim, typename NT, size_t n_components>
  bool apply_boundary_stencil(DiFfRG::def::BoundaryStencilValues<mdim, NT, n_components> &u_stencil,
                              DiFfRG::def::BoundaryStencilPoints<mdim> &x_stencil, const Point<mdim> &x_face) const
  {
    static_assert(mdim == 1);
    DiFfRG::Testing::fill_face_ghost_solution_boundary_stencil(u_stencil, x_stencil, x_face,
                                                               [this](const Point<1> &pos) { return solution(pos); });
    return true;
  }

  int hook_calls = 0;
  KT::AssemblyStage hook_stage = KT::AssemblyStage::diagnostics;
  unsigned int hook_face_count = 0;
  unsigned int hook_cell_count = 0;
  bool saw_boundary_face = false;
  bool saw_interior_face = false;
  bool hook_ready = false;
  double hook_min_u_plus = std::numeric_limits<double>::infinity();
  double hook_parallel_min_u_plus = std::numeric_limits<double>::infinity();
  double hook_min_cell_u = std::numeric_limits<double>::infinity();
  double hook_flux_scale = -1.0;
  double last_u_minus = std::numeric_limits<double>::quiet_NaN();
  double last_u_plus = std::numeric_limits<double>::quiet_NaN();
  double last_grad_minus = std::numeric_limits<double>::quiet_NaN();
  double last_grad_plus = std::numeric_limits<double>::quiet_NaN();
  double last_cell_u = std::numeric_limits<double>::quiet_NaN();
  double last_cell_point = std::numeric_limits<double>::quiet_NaN();
  mutable int flux_calls = 0;
  mutable bool flux_saw_hook_state = false;
};

static DiFfRG::JSONValue make_fv_reconstruction_diagnostics_json()
{
  return DiFfRG::json::value(
      {{"physical", {{"Lambda", 1.}}},
       {"discretization",
        {{"fe_order", 0},
         {"threads", 1},
         {"batch_size", 8},
         {"overintegration", 0},
         {"output_subdivisions", 1},
         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},
         {"grid", {{"x_grid", "0:0.25:1"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
       {"output",
        {{"verbosity", 0},
         {"vtk", false},
         {"hdf5", true},
         {"fv_reconstruction_diagnostics", true},
         {"fv_residual_contribution_diagnostics", false},
         {"output_buffer_size", 1}}}});
}

static DiFfRG::JSONValue make_fv_residual_contribution_diagnostics_json()
{
  auto json = make_fv_reconstruction_diagnostics_json();
  json.set_bool("/output/fv_reconstruction_diagnostics", false);
  json.set_bool("/output/fv_residual_contribution_diagnostics", true);
  return json;
}

static std::filesystem::path make_unique_test_directory(const std::string &prefix)
{
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  auto path = std::filesystem::temp_directory_path() / (prefix + "_" + std::to_string(static_cast<long long>(nonce)));
  std::filesystem::create_directories(path);
  return path;
}

static void ensure_logger()
{
  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
  }
}

template <typename State> static void check_face_reconstruction_state_1d(const State &actual, const State &expected)
{
  REQUIRE(actual.u_minus.size() == expected.u_minus.size());
  for (std::size_t c = 0; c < actual.u_minus.size(); ++c) {
    CHECK(actual.u_minus[c] == Catch::Approx(expected.u_minus[c]).margin(1e-14));
    CHECK(actual.u_plus[c] == Catch::Approx(expected.u_plus[c]).margin(1e-14));
    CHECK(actual.center_grad_minus[c][0] == Catch::Approx(expected.center_grad_minus[c][0]).margin(1e-14));
    CHECK(actual.center_grad_plus[c][0] == Catch::Approx(expected.center_grad_plus[c][0]).margin(1e-14));
    CHECK(actual.face_grad_minus[c][0] == Catch::Approx(expected.face_grad_minus[c][0]).margin(1e-14));
    CHECK(actual.face_grad_plus[c][0] == Catch::Approx(expected.face_grad_plus[c][0]).margin(1e-14));
  }
}

template <typename State> static void check_face_reconstruction_state_reversed_1d(const State &left, const State &right)
{
  REQUIRE(left.u_minus.size() == right.u_minus.size());
  for (std::size_t c = 0; c < left.u_minus.size(); ++c) {
    CHECK(left.u_minus[c] == Catch::Approx(right.u_plus[c]).margin(1e-14));
    CHECK(left.u_plus[c] == Catch::Approx(right.u_minus[c]).margin(1e-14));
    CHECK(left.center_grad_minus[c][0] == Catch::Approx(right.center_grad_plus[c][0]).margin(1e-14));
    CHECK(left.center_grad_plus[c][0] == Catch::Approx(right.center_grad_minus[c][0]).margin(1e-14));
    CHECK(left.face_grad_minus[c][0] == Catch::Approx(right.face_grad_plus[c][0]).margin(1e-14));
    CHECK(left.face_grad_plus[c][0] == Catch::Approx(right.face_grad_minus[c][0]).margin(1e-14));
  }
}

TEST_CASE("KT FV reconstruction diagnostics write expected HDF5 maps", "[FV][KT][hdf5]")
{
#ifndef H5CPP
  SUCCEED("HDF5 support is disabled.");
#else
  using Model = DiFfRG::Testing::ModelBurgersKT<1>;
  using Discretization = DiFfRG::FV::Discretization<typename Model::Components, NumberType, DiFfRG::RectangularMesh<1>>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_reconstruction_diagnostics_json();
  DiFfRG::Testing::PhysicalParameters prm;
  prm.initial_x0[0] = 1.0;
  prm.initial_x1[0] = 0.5;

  Model model(prm);
  DiFfRG::RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  DiFfRG::FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &solution = state.spatial_data();
  REQUIRE(solution.size() > 3);

  const auto output_dir = make_unique_test_directory("kt_fv_reconstruction_diagnostics");
  const std::string output_name = "kt_diag_unit";
  DiFfRG::DataOutput<1, VectorType> data_out(output_dir.string(), output_name, "output", json);

  assembler.attach_data_output(data_out, solution, VectorType());
  data_out.flush(0.25);

  const auto hdf5_file = output_dir / (output_name + "_fv_reconstruction_diagnostics.h5");
  REQUIRE(std::filesystem::exists(hdf5_file));
  REQUIRE(std::filesystem::is_regular_file(hdf5_file));

  auto file = DiFfRG::hdf5::File::open(hdf5_file.string(), DiFfRG::hdf5::Access::ReadOnly);
  auto root = file.root();
  const auto configuration_json = root.read_attribute<std::string>("configuration_json");
  CHECK(configuration_json.find("\"physical\"") != std::string::npos);
  CHECK(configuration_json.find("\"Lambda\"") != std::string::npos);
  REQUIRE(root.has_group("maps"));
  REQUIRE(root.has_group("coordinates"));
  auto maps = root.open_group("maps");

  const auto read_map = [&](const std::string &name, const std::size_t expected_size) {
    REQUIRE(maps.has_group(name));
    auto map_group = maps.open_group(name);
    REQUIRE(map_group.has_group("0"));
    auto series_group = map_group.open_group("0");
    REQUIRE(series_group.has_dataset("data"));
    CHECK(series_group.read_attribute<double>("time") == Catch::Approx(0.25));
    CHECK(series_group.read_attribute<int>("series_number") == 0);

    auto dataset = series_group.open_dataset("data");
    const auto extents = dataset.dataspace().extents();
    REQUIRE(extents.size() == 1);
    CHECK(extents[0] == expected_size);

    std::vector<std::array<double, 1>> values(expected_size);
    dataset.read(values);
    REQUIRE(values.size() == expected_size);
    for (const auto &value : values)
      CHECK(std::isfinite(value[0]));
    return values;
  };

  const std::size_t n_cells = solution.size();
  const auto cell_u = read_map("cell_u", n_cells);
  const auto cell_du_dx = read_map("cell_du_dx", n_cells);
  const auto cell_u_constant = read_map("cell_u_constant", 2 * n_cells);
  const auto cell_u_reconstruction = read_map("cell_u_reconstruction", 2 * n_cells);
  const auto face_u_minus = read_map("face_u_minus", n_cells + 1);
  const auto face_u_plus = read_map("face_u_plus", n_cells + 1);
  const auto face_du_dx_minus = read_map("face_du_dx_minus", n_cells + 1);
  const auto face_du_dx_plus = read_map("face_du_dx_plus", n_cells + 1);
  const auto face_advection_flux = read_map("face_advection_flux", n_cells + 1);
  const auto face_diffusion_flux = read_map("face_diffusion_flux", n_cells + 1);
  const auto face_total_flux = read_map("face_total_flux", n_cells + 1);

  const auto &support_points = discretization.get_support_points();
  REQUIRE(support_points.size() == n_cells);
  for (std::size_t i = 0; i < n_cells; ++i) {
    const double x = support_points[i][0];
    CHECK(cell_u[i][0] == Catch::Approx(1.0 + 0.5 * x).margin(1e-14));
    CHECK(cell_du_dx[i][0] == Catch::Approx(0.5).margin(1e-14));
    CHECK(cell_u_constant[2 * i][0] == Catch::Approx(cell_u[i][0]).margin(1e-14));
    CHECK(cell_u_constant[2 * i + 1][0] == Catch::Approx(cell_u[i][0]).margin(1e-14));
    CHECK(cell_u_reconstruction[2 * i][0] == Catch::Approx(1.0 + 0.5 * (0.25 * i)).margin(1e-14));
    CHECK(cell_u_reconstruction[2 * i + 1][0] == Catch::Approx(1.0 + 0.5 * (0.25 * (i + 1))).margin(1e-14));
  }

  for (std::size_t i = 0; i < n_cells + 1; ++i) {
    const double x_face = 0.25 * static_cast<double>(i);
    const double expected_u = 1.0 + 0.5 * x_face;
    const double expected_advection_flux = 0.5 * expected_u * expected_u;
    CHECK(face_u_minus[i][0] == Catch::Approx(expected_u).margin(1e-14));
    CHECK(face_u_plus[i][0] == Catch::Approx(expected_u).margin(1e-14));
    CHECK(face_du_dx_minus[i][0] == Catch::Approx(0.5).margin(1e-14));
    CHECK(face_du_dx_plus[i][0] == Catch::Approx(0.5).margin(1e-14));
    CHECK(face_advection_flux[i][0] == Catch::Approx(expected_advection_flux).margin(1e-14));
    CHECK(face_diffusion_flux[i][0] == Catch::Approx(0.0).margin(1e-14));
    CHECK(face_total_flux[i][0] == Catch::Approx(expected_advection_flux).margin(1e-14));
  }

  std::filesystem::remove_all(output_dir);
#endif
}

TEST_CASE("KT FV residual contribution diagnostics reconstruct assembled residual", "[FV][KT][hdf5]")
{
#ifndef H5CPP
  SUCCEED("HDF5 support is disabled.");
#else
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<typename Model::Components, NumberType, DiFfRG::RectangularMesh<1>>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_residual_contribution_diagnostics_json();
  Model model;
  DiFfRG::RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  DiFfRG::FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &solution = state.spatial_data();
  REQUIRE(solution.size() > 3);

  VectorType solution_dot(solution.size());
  for (unsigned int i = 0; i < solution_dot.size(); ++i)
    solution_dot[i] = 0.05 + 0.01 * static_cast<double>(i);

  VectorType residual(solution.size());
  residual = 0.;
  assembler.residual(residual, solution, 1.0, solution_dot, 1.0);

  const auto output_dir = make_unique_test_directory("kt_fv_residual_contribution_diagnostics");
  const std::string output_name = "kt_residual_diag_unit";
  DiFfRG::DataOutput<1, VectorType> data_out(output_dir.string(), output_name, "output", json);

  assembler.attach_data_output(data_out, solution, VectorType(), solution_dot, residual);
  data_out.flush(0.5);

  const auto hdf5_file = output_dir / (output_name + "_fv_residual_contribution_diagnostics.h5");
  REQUIRE(std::filesystem::exists(hdf5_file));
  REQUIRE(std::filesystem::is_regular_file(hdf5_file));

  auto file = DiFfRG::hdf5::File::open(hdf5_file.string(), DiFfRG::hdf5::Access::ReadOnly);
  auto root = file.root();
  REQUIRE(root.has_group("maps"));
  auto maps = root.open_group("maps");

  const auto read_map = [&](const std::string &name, const std::size_t expected_size) {
    REQUIRE(maps.has_group(name));
    auto map_group = maps.open_group(name);
    REQUIRE(map_group.has_group("0"));
    auto series_group = map_group.open_group("0");
    REQUIRE(series_group.has_dataset("data"));
    CHECK(series_group.read_attribute<double>("time") == Catch::Approx(0.5));
    CHECK(series_group.read_attribute<int>("series_number") == 0);

    auto dataset = series_group.open_dataset("data");
    const auto extents = dataset.dataspace().extents();
    REQUIRE(extents.size() == 1);
    CHECK(extents[0] == expected_size);

    std::vector<std::array<double, 1>> values(expected_size);
    dataset.read(values);
    for (const auto &value : values)
      CHECK(std::isfinite(value[0]));
    return values;
  };

  const std::size_t n_cells = solution.size();
  const auto advection = read_map("cell_advection_contribution", n_cells);
  const auto diffusion = read_map("cell_diffusion_contribution", n_cells);
  const auto source = read_map("cell_source_contribution", n_cells);
  const auto mass = read_map("cell_mass_contribution", n_cells);
  const auto total = read_map("cell_total_residual", n_cells);
  const auto total_minus_residual = read_map("cell_total_minus_residual", n_cells);

  bool has_nonzero_split = false;
  for (std::size_t i = 0; i < n_cells; ++i) {
    const double reconstructed_total = advection[i][0] + diffusion[i][0] + source[i][0] + mass[i][0];
    CHECK(total[i][0] == Catch::Approx(reconstructed_total).margin(1e-12));
    CHECK(total_minus_residual[i][0] == Catch::Approx(0.0).margin(1e-12));
    has_nonzero_split = has_nonzero_split || std::abs(advection[i][0]) > 0.0 || std::abs(diffusion[i][0]) > 0.0 ||
                        std::abs(source[i][0]) > 0.0 || std::abs(mass[i][0]) > 0.0;
  }
  CHECK(has_nonzero_split);

  std::filesystem::remove_all(output_dir);
#endif
}

TEST_CASE("KT reconstruction cache recomputes cell stencil values per solution", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<typename Model::Components, NumberType, DiFfRG::RectangularMesh<1>>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_residual_contribution_diagnostics_json();
  Model model;
  DiFfRG::RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType first(discretization.get_dof_handler().n_dofs());
  VectorType second(first.size());
  for (unsigned int i = 0; i < first.size(); ++i) {
    first[i] = 1.0 + 0.1 * static_cast<double>(i);
    second[i] = 2.0 + 0.3 * static_cast<double>(i);
  }

  auto cell = discretization.get_dof_handler().begin_active();
  ++cell;
  REQUIRE(cell != discretization.get_dof_handler().end());

  const auto first_cache =
      assembler.template build_solution_reconstruction_cache<typename Assembler::Reconstructor>(first);
  const auto second_cache =
      assembler.template build_solution_reconstruction_cache<typename Assembler::Reconstructor>(second);

  const auto cell_index = cell->active_cell_index();
  const auto &first_stencil = first_cache.cell_stencils[cell_index];
  const auto &second_stencil = second_cache.cell_stencils[cell_index];

  CHECK(first_stencil.cell.x == second_stencil.cell.x);
  CHECK(first_stencil.cell.dof_indices == second_stencil.cell.dof_indices);
  CHECK(first_stencil.cell.u[0] == Catch::Approx(first[first_stencil.cell.dof_indices[0]]));
  CHECK(second_stencil.cell.u[0] == Catch::Approx(second[second_stencil.cell.dof_indices[0]]));
  CHECK(first_stencil.cell.u[0] != Catch::Approx(second_stencil.cell.u[0]));

  for (const auto face_index : cell->face_indices()) {
    CHECK(first_stencil.neighbors.x[face_index] == second_stencil.neighbors.x[face_index]);
    CHECK(first_stencil.neighbors.dof_indices[face_index] == second_stencil.neighbors.dof_indices[face_index]);
    const auto neighbor_dof = first_stencil.neighbors.dof_indices[face_index][0];
    CHECK(first_stencil.neighbors.u[face_index][0] == Catch::Approx(first[neighbor_dof]));
    CHECK(second_stencil.neighbors.u[face_index][0] == Catch::Approx(second[neighbor_dof]));
    CHECK(first_stencil.neighbors.u[face_index][0] != Catch::Approx(second_stencil.neighbors.u[face_index][0]));
  }

  const auto face_index = 1U;
  REQUIRE(first_cache.face_reconstruction_valid[cell_index][face_index]);
  REQUIRE(second_cache.face_reconstruction_valid[cell_index][face_index]);
  const auto &first_reconstruction = assembler.get_cached_face_reconstruction(first_cache, cell, face_index);
  const auto &second_reconstruction = assembler.get_cached_face_reconstruction(second_cache, cell, face_index);
  CHECK(first_reconstruction.u_minus[0] != Catch::Approx(second_reconstruction.u_minus[0]));
  CHECK(first_reconstruction.u_plus[0] != Catch::Approx(second_reconstruction.u_plus[0]));
}

TEST_CASE("KT solution reconstruction cache stores reversed interior face orientations", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<typename Model::Components, NumberType, DiFfRG::RectangularMesh<1>>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_residual_contribution_diagnostics_json();
  Model model;
  DiFfRG::RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 1.0 + 0.13 * static_cast<double>(i) + 0.04 * static_cast<double>(i * i);

  auto cell = discretization.get_dof_handler().begin_active();
  ++cell;
  REQUIRE(cell != discretization.get_dof_handler().end());

  const auto face_index = 1U;
  REQUIRE(!cell->at_boundary(face_index));
  const auto neighbor = cell->neighbor(face_index);
  const auto neighbor_face_index = assembler.find_neighbor_face(cell, neighbor);

  const auto cache =
      assembler.template build_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution);
  REQUIRE(cache.face_reconstruction_valid[cell->active_cell_index()][face_index]);
  REQUIRE(cache.face_reconstruction_valid[neighbor->active_cell_index()][neighbor_face_index]);

  const auto &cell_state = assembler.get_cached_face_reconstruction(cache, cell, face_index);
  const auto &neighbor_state = assembler.get_cached_face_reconstruction(cache, neighbor, neighbor_face_index);
  check_face_reconstruction_state_reversed_1d(cell_state, neighbor_state);
}

TEST_CASE("KT solution reconstruction cache matches direct interior reconstruction", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<typename Model::Components, NumberType, DiFfRG::RectangularMesh<1>>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_residual_contribution_diagnostics_json();
  Model model;
  DiFfRG::RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 0.7 + 0.21 * static_cast<double>(i) - 0.02 * static_cast<double>(i * i);

  auto cell = discretization.get_dof_handler().begin_active();
  ++cell;
  REQUIRE(cell != discretization.get_dof_handler().end());

  const auto face_index = 1U;
  REQUIRE(!cell->at_boundary(face_index));
  const auto neighbor = cell->neighbor(face_index);
  const auto x_q = cell->face(face_index)->center();

  const auto cache =
      assembler.template build_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution);
  const auto &cached = assembler.get_cached_face_reconstruction(cache, cell, face_index);

  typename Assembler::CellStencilData cell_stencil;
  typename Assembler::CellStencilData neighbor_stencil;
  assembler.fill_cell_stencil(cell, solution, cell_stencil);
  assembler.fill_cell_stencil(neighbor, solution, neighbor_stencil);
  const auto expected = KT::internal::compute_interior_face_reconstruction_state<typename Assembler::Reconstructor>(
      cell_stencil, neighbor_stencil, x_q);

  check_face_reconstruction_state_1d(cached, expected);
}

TEST_CASE("KT solution reconstruction cache matches direct boundary reconstruction", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<typename Model::Components, NumberType, DiFfRG::RectangularMesh<1>>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_residual_contribution_diagnostics_json();
  Model model;
  DiFfRG::RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType solution(discretization.get_dof_handler().n_dofs());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution[i] = 0.4 + 0.17 * static_cast<double>(i) + 0.03 * static_cast<double>(i * i);

  const auto cell = discretization.get_dof_handler().begin_active();
  const auto face_index = 0U;
  REQUIRE(cell->at_boundary(face_index));
  const auto x_q = cell->face(face_index)->center();

  const auto cache =
      assembler.template build_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution);
  const auto &cached = assembler.get_cached_face_reconstruction(cache, cell, face_index);

  const auto boundary_stencil =
      assembler.template build_boundary_stencil_1d_from_cache<NumberType>(cell, face_index, solution);
  const auto expected = KT::internal::compute_boundary_face_reconstruction_state<typename Assembler::Reconstructor>(
      boundary_stencil, x_q, model);

  check_face_reconstruction_state_1d(cached, expected);
}

TEST_CASE("KT boundary stencil cache recomputes values per solution", "[FV][KT][cache]")
{
  using Model = ResidualContributionModel;
  using Discretization = DiFfRG::FV::Discretization<typename Model::Components, NumberType, DiFfRG::RectangularMesh<1>>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;
  using BoundaryIndex = DiFfRG::FV::KurganovTadmor::internal::BoundaryStencilIndex<1>;

  ensure_logger();

  auto json = make_fv_residual_contribution_diagnostics_json();
  Model model;
  DiFfRG::RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  VectorType first(discretization.get_dof_handler().n_dofs());
  VectorType second(first.size());
  for (unsigned int i = 0; i < first.size(); ++i) {
    first[i] = 1.0 + 0.2 * static_cast<double>(i);
    second[i] = 3.0 + 0.4 * static_cast<double>(i);
  }

  const auto cell = discretization.get_dof_handler().begin_active();
  const auto first_boundary = assembler.template build_boundary_stencil_1d_from_cache<NumberType>(cell, 0, first);
  const auto second_boundary = assembler.template build_boundary_stencil_1d_from_cache<NumberType>(cell, 0, second);

  for (const auto stencil_index :
       {BoundaryIndex::physical_cell, BoundaryIndex::upper_inner, BoundaryIndex::upper_outer}) {
    CHECK(first_boundary.x[stencil_index] == second_boundary.x[stencil_index]);
    CHECK(first_boundary.dof_indices[stencil_index] == second_boundary.dof_indices[stencil_index]);
    const auto dof = first_boundary.dof_indices[stencil_index][0];
    CHECK(first_boundary.u[stencil_index][0] == Catch::Approx(first[dof]));
    CHECK(second_boundary.u[stencil_index][0] == Catch::Approx(second[dof]));
    CHECK(first_boundary.u[stencil_index][0] != Catch::Approx(second_boundary.u[stencil_index][0]));
  }
}

TEST_CASE("KT face hook exposes cached reconstruction before residual flux", "[FV][KT][hook]")
{
  using Model = FaceHookProbeModel;
  using Discretization = DiFfRG::FV::Discretization<typename Model::Components, NumberType, DiFfRG::RectangularMesh<1>>;
  using Assembler = DiFfRG::FV::KurganovTadmor::Assembler<Discretization, Model>;
  using VectorType = typename Discretization::VectorType;

  ensure_logger();

  auto json = make_fv_residual_contribution_diagnostics_json();
  Model model;
  DiFfRG::RectangularMesh<1> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  DiFfRG::FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);
  const VectorType &solution = state.spatial_data();

  const auto cache =
      assembler.template build_solution_reconstruction_cache<typename Assembler::Reconstructor>(solution);
  const auto context = assembler.make_assembly_context_view(cache);
  STATIC_REQUIRE(KT::HasAssemblyContextView<decltype(context)>);
  STATIC_REQUIRE(!CanMakeAssemblyContextViewFromTemporaryCache<Assembler, VectorType>);

  VectorType residual(solution.size());
  VectorType solution_dot(solution.size());
  residual = 0.;
  solution_dot = 0.;

  assembler.residual(residual, solution, 1., solution_dot, 0.);

  CHECK(model.hook_calls == 1);
  CHECK(model.hook_stage == KT::AssemblyStage::residual);
  CHECK(model.hook_face_count == solution.size() + 1);
  CHECK(model.hook_cell_count == solution.size());
  CHECK(model.saw_boundary_face);
  CHECK(model.saw_interior_face);
  CHECK(std::isfinite(model.hook_min_u_plus));
  CHECK(model.hook_parallel_min_u_plus == Catch::Approx(model.hook_min_u_plus).margin(1e-14));
  CHECK(std::isfinite(model.hook_min_cell_u));
  CHECK(std::isfinite(model.last_u_minus));
  CHECK(std::isfinite(model.last_u_plus));
  CHECK(std::isfinite(model.last_grad_minus));
  CHECK(std::isfinite(model.last_grad_plus));
  CHECK(std::isfinite(model.last_cell_u));
  CHECK(std::isfinite(model.last_cell_point));
  CHECK(model.flux_calls > 0);
  CHECK(model.flux_saw_hook_state);
}

TEST_CASE("u_plus u_minus compoutation", "[FV][KT]")
{
  const int dim = 1;
  const uint n_components = 2;
  using GradComponentType = dealii::Tensor<1, dim, NumberType>;

  const DiFfRG::FV::KurganovTadmor::internal::GradientType<dim, NumberType, n_components> u_grad(
      {GradComponentType({-0.5}), GradComponentType({0.7})});
  const Point<dim> x_center(1.0);
  const Point<dim> x_q(2.0);
  const std::array<NumberType, n_components> u_val = {1.0, 2.0};
  const std::array<NumberType, n_components> u_minus_reference({0.5, 2.7});

  const std::array<NumberType, n_components> u_reconstructed = reconstruct_u(u_val, x_center, x_q, u_grad);
  CHECK(u_minus_reference[0] == Catch::Approx(u_reconstructed[0]));
  CHECK(u_minus_reference[1] == Catch::Approx(u_reconstructed[1]));
}

TEST_CASE("reconstruct_u_derivative 1D single component", "[FV][KT]")
{
  // 1D uniform grid: center at x=0, left neighbor at x=-1, right neighbor at x=1
  // u_center=2, u_left=1, u_right=4
  // slopes: du_left = (1-2)/(-1) = 1, du_right = (4-2)/1 = 2
  // minmod(1, 2) = 1 (picks du_left since |1|<|2|), gradient = 1
  // Reconstruct at x_q = 0.5: u_recon = 2 + 1*0.5 = 2.5
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;
  using Reconstructor = DiFfRG::def::TVDReconstructor<dim, DiFfRG::def::MinModLimiter, NumberType>;

  const Point<dim> center(0.0);
  const Point<dim> x_q(0.5);
  const std::array<Point<dim>, 2> x_n = {Point<dim>(-1.0), Point<dim>(1.0)};

  SECTION("derivative w.r.t. center value")
  {
    // minmod picks du_left = (u_left-u_c)/(-1).
    // d(du_left)/d(u_c) = -1/(-1) = 1, so d(grad)/d(u_c) = 1
    // d(u_recon)/d(u_c) = 1 + 1 * 0.5 = 1.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.5));
  }

  SECTION("derivative w.r.t. left neighbor")
  {
    // d(du_left)/d(u_left) = 1/(-1) = -1, minmod picks du_left
    // d(grad)/d(u_left) = -1
    // d(u_recon)/d(u_left) = 0 + (-1) * 0.5 = -0.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(-0.5));
  }

  SECTION("derivative w.r.t. right neighbor (inactive in minmod)")
  {
    // minmod picks du_left (not du_right), so d(grad)/d(u_right) = 0
    // d(u_recon)/d(u_right) = 0 + 0 = 0
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[1][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[1][0]);
    CHECK(result[0] == Catch::Approx(0.0));
  }
}

TEST_CASE("reconstruct_u_derivative at local extremum (vanishing limiter)", "[FV][KT]")
{
  // Opposite slopes => minmod returns 0, gradient derivative also 0
  // center at x=0, u=3; left at x=-1, u=4; right at x=1, u=4
  // du_left = (4-3)/(-1) = -1, du_right = (4-3)/1 = 1
  // minmod(-1, 1) = 0 => flat reconstruction
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;
  using Reconstructor = DiFfRG::def::TVDReconstructor<dim, DiFfRG::def::MinModLimiter, NumberType>;

  const Point<dim> center(0.0);
  const Point<dim> x_q(0.5);
  const std::array<Point<dim>, 2> x_n = {Point<dim>(-1.0), Point<dim>(1.0)};

  SECTION("derivative w.r.t. center is Kronecker delta only")
  {
    std::array<AD, 1> u_c = {AD(3.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(4.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.0));
  }

  SECTION("derivative w.r.t. neighbor is zero")
  {
    std::array<AD, 1> u_c = {AD(3.0)};
    std::array<std::array<AD, 1>, 2> u_n = {std::array<AD, 1>{AD(4.0)}, std::array<AD, 1>{AD(4.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(0.0));
  }
}

TEST_CASE("reconstruct_u_derivative 2D single component", "[FV][KT]")
{
  // 2D: center at (0,0), u=2
  // x-neighbors: left (-1,0) u=1, right (1,0) u=4
  //   du_x_left = (1-2)/(-1)=1, du_x_right = (4-2)/1=2, minmod=1 (picks left)
  // y-neighbors: bottom (0,-1) u=0.5, top (0,1) u=5
  //   du_y_bottom = (0.5-2)/(-1)=1.5, du_y_top = (5-2)/1=3, minmod=1.5 (picks bottom)
  // gradient = (1, 1.5)
  constexpr int dim = 2;
  constexpr size_t n_components = 1;
  using AD = autodiff::Real<1, NumberType>;
  using Reconstructor = DiFfRG::def::TVDReconstructor<dim, DiFfRG::def::MinModLimiter, NumberType>;

  const Point<dim> center(0.0, 0.0);
  const Point<dim> x_q(0.5, 0.3);
  // neighbors ordered: x-left, x-right, y-left(bottom), y-right(top)
  const std::array<Point<dim>, 4> x_n = {Point<dim>(-1.0, 0.0), Point<dim>(1.0, 0.0), Point<dim>(0.0, -1.0),
                                         Point<dim>(0.0, 1.0)};

  SECTION("derivative w.r.t. center")
  {
    // d(grad_x)/d(u_c) = d(du_x_left)/d(u_c) = -1/(-1) = 1
    // d(grad_y)/d(u_c) = d(du_y_bottom)/d(u_c) = -1/(-1) = 1
    // grad_deriv = (1, 1)
    // d(u_recon)/d(u_c) = 1 + 1*0.5 + 1*0.3 = 1.8
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_c[0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_c[0]);
    CHECK(result[0] == Catch::Approx(1.8));
  }

  SECTION("derivative w.r.t. x-left neighbor")
  {
    // d(grad_x)/d(u_xleft) = 1/(-1) = -1 (minmod picks du_x_left)
    // d(grad_y)/d(u_xleft) = 0 (y-dimension independent)
    // d(u_recon)/d(u_xleft) = 0 + (-1)*0.5 + 0*0.3 = -0.5
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_n[0][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[0][0]);
    CHECK(result[0] == Catch::Approx(-0.5));
  }

  SECTION("derivative w.r.t. y-bottom neighbor")
  {
    // d(grad_x)/d(u_ybottom) = 0
    // d(grad_y)/d(u_ybottom) = 1/(-1) = -1 (minmod picks du_y_bottom)
    // d(u_recon)/d(u_ybottom) = 0 + 0*0.5 + (-1)*0.3 = -0.3
    std::array<AD, 1> u_c = {AD(2.0)};
    std::array<std::array<AD, 1>, 4> u_n = {std::array<AD, 1>{AD(1.0)}, std::array<AD, 1>{AD(4.0)},
                                            std::array<AD, 1>{AD(0.5)}, std::array<AD, 1>{AD(5.0)}};
    seed(u_n[2][0]);
    const auto result = KT::internal::reconstruct_u_derivative<Reconstructor, dim, NumberType, n_components>(
        u_c, center, x_q, x_n, u_n);
    unseed(u_n[2][0]);
    CHECK(result[0] == Catch::Approx(-0.3));
  }
}

TEST_CASE("Test compute_numerical_flux in 1D with one component", "[FV][KT]")
{
  constexpr int dim = 1;
  constexpr size_t n_components = 1;

  SECTION("Symmetric flux cancels diffusion term")
  {
    // u_plus == u_minus => H = F (no diffusion)
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({3.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({3.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({5.0})};
    std::array<NumberType, n_components> u_plus = {1.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(3.0));
  }

  SECTION("General case")
  {
    // F_plus = 4.0, F_minus = 2.0, a = 1.0, u_plus = 3.0, u_minus = 1.0
    // H = (4+2)/2 - 1*(3-1)/2 = 3 - 1 = 2
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({4.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({1.0})};
    std::array<NumberType, n_components> u_plus = {3.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(2.0));
  }

  SECTION("Zero wave speed reduces to central flux")
  {
    // a = 0 => H = (F_plus + F_minus)/2
    std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({6.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0})};
    std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({0.0})};
    std::array<NumberType, n_components> u_plus = {5.0};
    std::array<NumberType, n_components> u_minus = {1.0};

    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(4.0));
  }
}

TEST_CASE("Test compute_numerical_flux in 2D with one component", "[FV][KT]")
{
  constexpr int dim = 2;
  constexpr size_t n_components = 1;

  // F_plus = (4, 1), F_minus = (2, 3), a = (1, 2), u_plus = 3, u_minus = 1
  // H[0] = (4+2)/2 - 1*(3-1)/2 = 3 - 1 = 2
  // H[1] = (1+3)/2 - 2*(3-1)/2 = 2 - 2 = 0
  std::array<Tensor<1, dim, NumberType>, n_components> F_plus{Tensor<1, dim, NumberType>({4.0, 1.0})};
  std::array<Tensor<1, dim, NumberType>, n_components> F_minus{Tensor<1, dim, NumberType>({2.0, 3.0})};
  std::array<Tensor<1, dim, NumberType>, n_components> a_half{Tensor<1, dim, NumberType>({1.0, 2.0})};
  std::array<NumberType, n_components> u_plus = {3.0};
  std::array<NumberType, n_components> u_minus = {1.0};

  const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
  CHECK(H[0][0] == Catch::Approx(2.0));
  CHECK(H[0][1] == Catch::Approx(0.0));
}

TEST_CASE("Test compute_kt_flux_and_speeds with Burgers flux in 1D", "[FV][KT]")
{
  // TestModel: F(u) = u^2/2 + x, so dF/du = u
  constexpr int dim = 1;
  constexpr size_t n_components = 1;
  TestModel model;

  SECTION("Flux values and wave speed")
  {
    // At x=0.5, u_plus=2.0, u_minus=1.0
    // F(u_plus)  = 2^2/2 + 0.5 = 2.5
    // F(u_minus) = 1^2/2 + 0.5 = 1.0
    // dF/du(u_plus) = 2.0, dF/du(u_minus) = 1.0
    // a = max(|2|, |1|) = 2.0
    const Point<dim> x_q(0.5);
    const std::array<NumberType, n_components> u_plus = {2.0};
    const std::array<NumberType, n_components> u_minus = {1.0};

    const auto [F_plus, F_minus, a_half] =
        KT::internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(u_plus, u_minus, x_q, model);
    CHECK(F_plus[0][0] == Catch::Approx(2.5));
    CHECK(F_minus[0][0] == Catch::Approx(1.0));
    CHECK(a_half[0][0] == Catch::Approx(2.0));
  }

  SECTION("Symmetric states give symmetric fluxes")
  {
    // u_plus = u_minus = 3.0, x=1.0
    // F = 3^2/2 + 1 = 5.5
    // a = |3| = 3.0
    const Point<dim> x_q(1.0);
    const std::array<NumberType, n_components> u_plus = {3.0};
    const std::array<NumberType, n_components> u_minus = {3.0};

    const auto [F_plus, F_minus, a_half] =
        KT::internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(u_plus, u_minus, x_q, model);
    CHECK(F_plus[0][0] == Catch::Approx(5.5));
    CHECK(F_minus[0][0] == Catch::Approx(5.5));
    CHECK(a_half[0][0] == Catch::Approx(3.0));
  }

  SECTION("Negative velocities")
  {
    // u_plus = -1.0, u_minus = -3.0, x=0
    // F(u_plus)  = (-1)^2/2 + 0 = 0.5
    // F(u_minus) = (-3)^2/2 + 0 = 4.5
    // dF/du(u_plus) = -1, dF/du(u_minus) = -3
    // a = max(|-1|, |-3|) = 3.0
    const Point<dim> x_q(0.0);
    const std::array<NumberType, n_components> u_plus = {-1.0};
    const std::array<NumberType, n_components> u_minus = {-3.0};

    const auto [F_plus, F_minus, a_half] =
        KT::internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(u_plus, u_minus, x_q, model);
    CHECK(F_plus[0][0] == Catch::Approx(0.5));
    CHECK(F_minus[0][0] == Catch::Approx(4.5));
    CHECK(a_half[0][0] == Catch::Approx(3.0));
  }

  SECTION("Full KT numerical flux for Burgers")
  {
    // u_plus=2, u_minus=1, x=0.5
    // F_plus=2.5, F_minus=1.0, a=2.0
    // H = (2.5+1.0)/2 - 2.0*(2.0-1.0)/2 = 1.75 - 1.0 = 0.75
    const Point<dim> x_q(0.5);
    const std::array<NumberType, n_components> u_plus = {2.0};
    const std::array<NumberType, n_components> u_minus = {1.0};

    const auto [F_plus, F_minus, a_half] =
        KT::internal::compute_kt_flux_and_speeds<WaveSpeedStrategy>(u_plus, u_minus, x_q, model);
    const auto H = compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
    CHECK(H[0][0] == Catch::Approx(0.75));
  }
}

// ---------------------------------------------------------------------------
// Tests for tag_cell_dofs
// ---------------------------------------------------------------------------

using KT::internal::make_tagged_neighbors;
using KT::internal::tag_cell_dofs;
template <int dim, typename NT, size_t nc> using KTCellData = KT::internal::CellData<dim, NT, nc>;
template <int dim, typename NT, size_t nc> using KTNeighborData = KT::internal::NeighborData<dim, NT, nc>;
using AD = autodiff::Real<1, NumberType>;

TEST_CASE("tag_cell_dofs 1D single component — matching dof is seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  const KTCellData<dim, NumberType, nc> cell_data{
      .x = Point<dim>(1.0),
      .u = {3.5},
      .dof_indices = {42},
  };

  const auto result = tag_cell_dofs(cell_data, 42);

  CHECK(result.x[0] == Catch::Approx(1.0));
  CHECK(val(result.u[0]) == Catch::Approx(3.5));
  CHECK(result.dof_indices[0] == 42);
  CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
}

TEST_CASE("tag_cell_dofs 1D single component — non-matching dof is not seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  const KTCellData<dim, NumberType, nc> cell_data{
      .x = Point<dim>(1.0),
      .u = {3.5},
      .dof_indices = {42},
  };

  const auto result = tag_cell_dofs(cell_data, 99);

  CHECK(val(result.u[0]) == Catch::Approx(3.5));
  CHECK(derivative(result.u[0]) == Catch::Approx(0.0));
}

TEST_CASE("tag_cell_dofs 1D multi-component — only matching component seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;

  const KTCellData<dim, NumberType, nc> cell_data{
      .x = Point<dim>(0.0),
      .u = {1.0, 2.0},
      .dof_indices = {10, 20},
  };

  SECTION("seed second component")
  {
    const auto result = tag_cell_dofs(cell_data, 20);

    CHECK(val(result.u[0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[0]) == Catch::Approx(0.0));
    CHECK(val(result.u[1]) == Catch::Approx(2.0));
    CHECK(derivative(result.u[1]) == Catch::Approx(1.0));
    CHECK(result.dof_indices[0] == 10);
    CHECK(result.dof_indices[1] == 20);
  }

  SECTION("seed first component")
  {
    const auto result = tag_cell_dofs(cell_data, 10);

    CHECK(val(result.u[0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
    CHECK(val(result.u[1]) == Catch::Approx(2.0));
    CHECK(derivative(result.u[1]) == Catch::Approx(0.0));
  }
}

TEST_CASE("tag_cell_dofs 2D single component — point preserved and seeded", "[FV][KT][tag_cell_dofs]")
{
  constexpr int dim = 2;
  constexpr size_t nc = 1;

  const KTCellData<dim, NumberType, nc> cell_data{
      .x = Point<dim>(0.5, 0.7),
      .u = {4.0},
      .dof_indices = {5},
  };

  const auto result = tag_cell_dofs(cell_data, 5);

  CHECK(result.x[0] == Catch::Approx(0.5));
  CHECK(result.x[1] == Catch::Approx(0.7));
  CHECK(val(result.u[0]) == Catch::Approx(4.0));
  CHECK(derivative(result.u[0]) == Catch::Approx(1.0));
  CHECK(result.dof_indices[0] == 5);
}

// ---------------------------------------------------------------------------
// Tests for make_tagged_neighbors
// ---------------------------------------------------------------------------

TEST_CASE("make_tagged_neighbors 1D single component — matching dof in one face", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  const KTNeighborData<dim, NumberType, nc> nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10},
                      std::array<dealii::types::global_dof_index, nc>{20}},
  };

  const auto result = make_tagged_neighbors(nd, 10);

  // Face 0 matches — seeded
  CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
  CHECK(derivative(result.u[0][0]) == Catch::Approx(1.0));

  // Face 1 does not match — not seeded
  CHECK(val(result.u[1][0]) == Catch::Approx(2.0));
  CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));

  // Positions preserved
  CHECK(result.x[0][0] == Catch::Approx(-1.0));
  CHECK(result.x[1][0] == Catch::Approx(1.0));

  // dof_indices preserved
  CHECK(result.dof_indices[0][0] == 10);
  CHECK(result.dof_indices[1][0] == 20);
}

TEST_CASE("make_tagged_neighbors 1D single component — no matching dof", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 1;

  const KTNeighborData<dim, NumberType, nc> nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10},
                      std::array<dealii::types::global_dof_index, nc>{20}},
  };

  const auto result = make_tagged_neighbors(nd, 99);

  for (size_t face = 0; face < KTNeighborData<dim, NumberType, nc>::n_faces; ++face)
    for (size_t c = 0; c < nc; ++c)
      CHECK(derivative(result.u[face][c]) == Catch::Approx(0.0));
}

TEST_CASE("make_tagged_neighbors 1D multi-component — specific face and component seeded",
          "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 1;
  constexpr size_t nc = 2;

  const KTNeighborData<dim, NumberType, nc> nd{
      .x = {Point<dim>(-1.0), Point<dim>(1.0)},
      .u = {std::array<NumberType, nc>{1.0, 2.0}, std::array<NumberType, nc>{3.0, 4.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{10, 20},
                      std::array<dealii::types::global_dof_index, nc>{30, 40}},
  };

  SECTION("tag dof 20 — face 0, component 1")
  {
    const auto result = make_tagged_neighbors(nd, 20);

    // Only face 0, component 1 should be seeded
    CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[0][1]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[1][1]) == Catch::Approx(0.0));

    // Values preserved
    CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
    CHECK(val(result.u[0][1]) == Catch::Approx(2.0));
    CHECK(val(result.u[1][0]) == Catch::Approx(3.0));
    CHECK(val(result.u[1][1]) == Catch::Approx(4.0));
  }

  SECTION("tag dof 30 — face 1, component 0")
  {
    const auto result = make_tagged_neighbors(nd, 30);

    CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[0][1]) == Catch::Approx(0.0));
    CHECK(derivative(result.u[1][0]) == Catch::Approx(1.0));
    CHECK(derivative(result.u[1][1]) == Catch::Approx(0.0));
  }
}

TEST_CASE("make_tagged_neighbors 2D single component — 4 faces, one matching", "[FV][KT][make_tagged_neighbors]")
{
  constexpr int dim = 2;
  constexpr size_t nc = 1;

  const KTNeighborData<dim, NumberType, nc> nd{
      .x = {Point<dim>(-1.0, 0.0), Point<dim>(1.0, 0.0), Point<dim>(0.0, -1.0), Point<dim>(0.0, 1.0)},
      .u = {std::array<NumberType, nc>{1.0}, std::array<NumberType, nc>{2.0}, std::array<NumberType, nc>{3.0},
            std::array<NumberType, nc>{4.0}},
      .dof_indices = {std::array<dealii::types::global_dof_index, nc>{100},
                      std::array<dealii::types::global_dof_index, nc>{200},
                      std::array<dealii::types::global_dof_index, nc>{300},
                      std::array<dealii::types::global_dof_index, nc>{400}},
  };

  const auto result = make_tagged_neighbors(nd, 300);

  // Only face 2 should be seeded
  CHECK(derivative(result.u[0][0]) == Catch::Approx(0.0));
  CHECK(derivative(result.u[1][0]) == Catch::Approx(0.0));
  CHECK(derivative(result.u[2][0]) == Catch::Approx(1.0));
  CHECK(derivative(result.u[3][0]) == Catch::Approx(0.0));

  // Values preserved
  CHECK(val(result.u[0][0]) == Catch::Approx(1.0));
  CHECK(val(result.u[1][0]) == Catch::Approx(2.0));
  CHECK(val(result.u[2][0]) == Catch::Approx(3.0));
  CHECK(val(result.u[3][0]) == Catch::Approx(4.0));

  // Positions preserved
  CHECK(result.x[0][0] == Catch::Approx(-1.0));
  CHECK(result.x[0][1] == Catch::Approx(0.0));
  CHECK(result.x[2][0] == Catch::Approx(0.0));
  CHECK(result.x[2][1] == Catch::Approx(-1.0));

  // dof_indices preserved
  CHECK(result.dof_indices[2][0] == 300);
}
