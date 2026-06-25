#pragma once

#ifdef H5CPP

namespace diagnostics
{
        class PointCoordinates1D
        {
        public:
          using ctype = double;
          static constexpr size_t dim = 1;

          explicit PointCoordinates1D(std::vector<double> points) : points(std::move(points)) {}

          device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const { return {i}; }
          device::array<double, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<size_t, 1> &i) const
          {
            return {points[i[0]]};
          }
          size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return points.size(); }
          device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {points.size()}; }

          std::string to_string() const
          {
            std::ostringstream out;
            out << "FVKTPointCoordinates1D(" << points.size() << ",{";
            out << std::setprecision(std::numeric_limits<double>::max_digits10);
            for (size_t i = 0; i < points.size(); ++i) {
              if (i > 0) out << ",";
              out << points[i];
            }
            out << "})";
            return out.str();
          }

        private:
          std::vector<double> points;
        };

        template <typename AssemblerType>
        void write_reconstruction_maps(
            const AssemblerType &assembler,
            DataOutput<AssemblerType::dim, typename AssemblerType::VectorType> &data_out,
            const dealii::DoFHandler<AssemblerType::dim> &dof_handler,
            const typename AssemblerType::Model &model, const typename AssemblerType::VectorType &solution)
        {
          static_assert(AssemblerType::dim == 1, "FV reconstruction diagnostics are currently implemented for 1D only.");

          using NumberType = typename AssemblerType::NumberType;
          using Iterator = typename AssemblerType::Iterator;
          using Reconstructor = typename AssemblerType::Reconstructor;
          using CellStencilData = typename AssemblerType::CellStencilData;
          constexpr size_t n_components = AssemblerType::n_components;
          using Value = std::array<double, n_components>;

          struct CellRecord {
            Iterator cell;
            CellStencilData stencil;
            Value u{};
            Value du_dx{};
          };

          std::vector<CellRecord> cells;

          for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
            CellRecord record;
            record.cell = cell;
            assembler.fill_cell_stencil(cell, solution, record.stencil);
            const auto grad = Reconstructor::template compute_gradient<n_components>(
                record.stencil.cell.x, record.stencil.cell.u, record.stencil.neighbors.x, record.stencil.neighbors.u);
            for (size_t c = 0; c < n_components; ++c) {
              record.u[c] = static_cast<double>(record.stencil.cell.u[c]);
              record.du_dx[c] = static_cast<double>(grad[c][0]);
            }
            cells.push_back(record);
          }

          std::sort(cells.begin(), cells.end(),
                    [](const auto &a, const auto &b) { return a.stencil.cell.x[0] < b.stencil.cell.x[0]; });

          auto as_cell_interval_value = [](const auto &values) {
            Value result{};
            for (size_t c = 0; c < n_components; ++c)
              result[c] = static_cast<double>(values[c]);
            return result;
          };

          std::vector<double> cell_x;
          std::vector<double> cell_interval_x;
          std::vector<Value> cell_u;
          std::vector<Value> cell_du_dx;
          std::vector<Value> cell_u_constant;
          std::vector<Value> cell_u_reconstruction;
          cell_x.reserve(cells.size());
          cell_interval_x.reserve(2 * cells.size());
          cell_u.reserve(cells.size());
          cell_du_dx.reserve(cells.size());
          cell_u_constant.reserve(2 * cells.size());
          cell_u_reconstruction.reserve(2 * cells.size());
          for (const auto &cell : cells) {
            cell_x.push_back(cell.stencil.cell.x[0]);
            cell_u.push_back(cell.u);
            cell_du_dx.push_back(cell.du_dx);

            const auto x_left = cell.cell->face(0)->center();
            const auto x_right = cell.cell->face(1)->center();
            const auto grad = Reconstructor::template compute_gradient<n_components>(
                cell.stencil.cell.x, cell.stencil.cell.u, cell.stencil.neighbors.x, cell.stencil.neighbors.u);
            const auto u_left = internal::reconstruct_u(cell.stencil.cell.u, cell.stencil.cell.x, x_left, grad);
            const auto u_right = internal::reconstruct_u(cell.stencil.cell.u, cell.stencil.cell.x, x_right, grad);

            cell_interval_x.push_back(x_left[0]);
            cell_interval_x.push_back(x_right[0]);
            cell_u_constant.push_back(cell.u);
            cell_u_constant.push_back(cell.u);
            cell_u_reconstruction.push_back(as_cell_interval_value(u_left));
            cell_u_reconstruction.push_back(as_cell_interval_value(u_right));
          }

          std::vector<double> face_x;
          std::vector<Value> face_u_minus;
          std::vector<Value> face_u_plus;
          std::vector<Value> face_du_dx_minus;
          std::vector<Value> face_du_dx_plus;
          std::vector<Value> face_advection_flux;
          std::vector<Value> face_diffusion_flux;
          std::vector<Value> face_total_flux;
          face_x.reserve(cells.size() + 1);
          face_u_minus.reserve(cells.size() + 1);
          face_u_plus.reserve(cells.size() + 1);
          face_du_dx_minus.reserve(cells.size() + 1);
          face_du_dx_plus.reserve(cells.size() + 1);
          face_advection_flux.reserve(cells.size() + 1);
          face_diffusion_flux.reserve(cells.size() + 1);
          face_total_flux.reserve(cells.size() + 1);

          auto append_face = [&](const double x, const Value &u_minus, const Value &u_plus, const Value &du_dx_minus,
                                 const Value &du_dx_plus, const Value &advection_flux,
                                 const Value &diffusion_flux, const Value &total_flux) {
            face_x.push_back(x);
            face_u_minus.push_back(u_minus);
            face_u_plus.push_back(u_plus);
            face_du_dx_minus.push_back(du_dx_minus);
            face_du_dx_plus.push_back(du_dx_plus);
            face_advection_flux.push_back(advection_flux);
            face_diffusion_flux.push_back(diffusion_flux);
            face_total_flux.push_back(total_flux);
          };

          auto as_value = [](const auto &values) {
            Value result{};
            for (size_t c = 0; c < n_components; ++c)
              result[c] = static_cast<double>(values[c]);
            return result;
          };

          auto as_gradient_value = [](const auto &grad) {
            Value result{};
            for (size_t c = 0; c < n_components; ++c)
              result[c] = static_cast<double>(grad[c][0]);
            return result;
          };

          struct FluxValues {
            Value advection{};
            Value diffusion{};
            Value total{};
          };

          auto compute_flux_values = [&](const auto &u_plus, const auto &u_minus, const auto &grad_plus,
                                         const auto &grad_minus, const auto &x_q) {
            const auto [F_plus, F_minus, a_half] =
                internal::compute_kt_flux_and_speeds<typename AssemblerType::WaveSpeedStrategy>(u_plus, u_minus, x_q,
                                                                                                model);
            const auto H = internal::compute_numerical_flux(F_plus, F_minus, a_half, u_plus, u_minus);
            const auto D = internal::compute_diffusion_flux(u_plus, u_minus, grad_plus, grad_minus, x_q, model);

            FluxValues result;
            for (size_t c = 0; c < n_components; ++c) {
              result.advection[c] = static_cast<double>(H[c][0]);
              result.diffusion[c] = static_cast<double>(D[c][0]);
              result.total[c] = static_cast<double>(H[c][0] - D[c][0]);
            }
            return result;
          };

          auto reconstruct_boundary = [&](const CellRecord &cell, const unsigned int face_no) {
            const auto x_q = cell.cell->face(face_no)->center();
            auto boundary_stencil =
                assembler.template build_boundary_stencil_from_cache<NumberType>(cell.cell, face_no, solution);
            const auto reconstruction =
                internal::compute_boundary_face_reconstruction_state<Reconstructor>(
                    boundary_stencil, x_q, model);

            if (face_no == 0) {
              const auto fluxes = compute_flux_values(reconstruction.u_minus, reconstruction.u_plus,
                                                       reconstruction.face_grad_minus,
                                                       reconstruction.face_grad_plus, x_q);
              append_face(x_q[0], as_value(reconstruction.u_plus), as_value(reconstruction.u_minus),
                          as_gradient_value(reconstruction.face_grad_plus),
                          as_gradient_value(reconstruction.face_grad_minus), fluxes.advection, fluxes.diffusion,
                          fluxes.total);
            } else {
              const auto fluxes = compute_flux_values(reconstruction.u_plus, reconstruction.u_minus,
                                                       reconstruction.face_grad_plus,
                                                       reconstruction.face_grad_minus, x_q);
              append_face(x_q[0], as_value(reconstruction.u_minus), as_value(reconstruction.u_plus),
                          as_gradient_value(reconstruction.face_grad_minus),
                          as_gradient_value(reconstruction.face_grad_plus), fluxes.advection, fluxes.diffusion,
                          fluxes.total);
            }
          };

          if (!cells.empty()) reconstruct_boundary(cells.front(), 0);

          for (size_t i = 1; i < cells.size(); ++i) {
            const auto &lower = cells[i - 1];
            const auto &upper = cells[i];
            const auto x_q = lower.cell->face(1)->center();
            const auto reconstruction =
                internal::compute_interior_face_reconstruction_state<Reconstructor>(lower.stencil, upper.stencil, x_q);
            const auto fluxes = compute_flux_values(reconstruction.u_plus, reconstruction.u_minus,
                                                    reconstruction.face_grad_plus,
                                                    reconstruction.face_grad_minus, x_q);
            append_face(x_q[0], as_value(reconstruction.u_minus), as_value(reconstruction.u_plus),
                        as_gradient_value(reconstruction.face_grad_minus),
                        as_gradient_value(reconstruction.face_grad_plus), fluxes.advection, fluxes.diffusion,
                        fluxes.total);
          }

          if (!cells.empty()) reconstruct_boundary(cells.back(), 1);

          auto &hdf = data_out.hdf5("fv_reconstruction_diagnostics.h5");
          const PointCoordinates1D cell_coordinates(std::move(cell_x));
          const PointCoordinates1D cell_interval_coordinates(std::move(cell_interval_x));
          const PointCoordinates1D face_coordinates(std::move(face_x));
          hdf.map("cell_u", cell_coordinates, cell_u.data());
          hdf.map("cell_du_dx", cell_coordinates, cell_du_dx.data());
          hdf.map("cell_u_constant", cell_interval_coordinates, cell_u_constant.data());
          hdf.map("cell_u_reconstruction", cell_interval_coordinates, cell_u_reconstruction.data());
          hdf.map("face_u_minus", face_coordinates, face_u_minus.data());
          hdf.map("face_u_plus", face_coordinates, face_u_plus.data());
          hdf.map("face_du_dx_minus", face_coordinates, face_du_dx_minus.data());
          hdf.map("face_du_dx_plus", face_coordinates, face_du_dx_plus.data());
          hdf.map("face_advection_flux", face_coordinates, face_advection_flux.data());
          hdf.map("face_diffusion_flux", face_coordinates, face_diffusion_flux.data());
          hdf.map("face_total_flux", face_coordinates, face_total_flux.data());
        }

        template <typename AssemblerType>
        void write_residual_contribution_maps(
            DataOutput<AssemblerType::dim, typename AssemblerType::VectorType> &data_out,
            const dealii::DoFHandler<AssemblerType::dim> &dof_handler,
            const typename AssemblerType::ResidualContributionDiagnostics &contributions)
        {
          static_assert(AssemblerType::dim == 1,
                        "FV residual contribution diagnostics are currently implemented for 1D only.");

          using Value = std::array<double, AssemblerType::n_components>;

          struct CellRecord {
            double x = 0.;
            std::array<dealii::types::global_dof_index, AssemblerType::n_components> dof_indices{};
          };

          std::vector<CellRecord> cells;
          cells.reserve(dof_handler.get_triangulation().n_active_cells());
          std::vector<dealii::types::global_dof_index> dof_indices(AssemblerType::n_components);
          for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
            CellRecord record;
            record.x = cell->center()[0];
            cell->get_dof_indices(dof_indices);
            for (size_t c = 0; c < AssemblerType::n_components; ++c)
              record.dof_indices[c] = dof_indices[c];
            cells.push_back(record);
          }

          std::sort(cells.begin(), cells.end(), [](const auto &a, const auto &b) { return a.x < b.x; });

          std::vector<double> cell_x;
          cell_x.reserve(cells.size());
          for (const auto &cell : cells)
            cell_x.push_back(cell.x);

          auto values_from_vector = [&](const auto &vector) {
            std::vector<Value> values;
            values.reserve(cells.size());
            for (const auto &cell : cells) {
              Value value{};
              for (size_t c = 0; c < AssemblerType::n_components; ++c)
                value[c] = static_cast<double>(vector[cell.dof_indices[c]]);
              values.push_back(value);
            }
            return values;
          };

          auto advection = values_from_vector(contributions.advection);
          auto diffusion = values_from_vector(contributions.diffusion);
          auto source = values_from_vector(contributions.source);
          auto mass = values_from_vector(contributions.mass);
          auto total = values_from_vector(contributions.total);
          auto total_minus_residual = values_from_vector(contributions.total_minus_residual);

          auto &hdf = data_out.hdf5("fv_residual_contribution_diagnostics.h5");
          const PointCoordinates1D cell_coordinates(std::move(cell_x));
          hdf.map("cell_advection_contribution", cell_coordinates, advection.data());
          hdf.map("cell_diffusion_contribution", cell_coordinates, diffusion.data());
          hdf.map("cell_source_contribution", cell_coordinates, source.data());
          hdf.map("cell_mass_contribution", cell_coordinates, mass.data());
          hdf.map("cell_total_residual", cell_coordinates, total.data());
          if (contributions.has_residual)
            hdf.map("cell_total_minus_residual", cell_coordinates, total_minus_residual.data());
        }
} // namespace diagnostics

#endif
