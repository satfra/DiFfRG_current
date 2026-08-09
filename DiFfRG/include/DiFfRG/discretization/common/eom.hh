#pragma once

// external libraries
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/fe_field_function.h>

#include <DiFfRG/discretization/common/eom_config.hh>

// standard library
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief An owning scalar potential reconstructed from a model EoM vector field.
   *
   * The finite element and DoFHandler are retained alongside the potential vector so the
   * reconstruction can be attached to delayed finite-element output safely.
   */
  template <int dim, typename NumberType> struct ReconstructedEoMPotential {
    dealii::Point<dim> minimum;
    std::unique_ptr<dealii::FiniteElement<dim>> finite_element;
    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler;
    dealii::Vector<NumberType> values;
  };

  /**
   * @brief An owning scalar potential reconstructed from a model-provided raw gradient.
   *
   * Its additive gauge is fixed at the mesh origin. Consequently, value differences and all derivatives are
   * gauge-independent, while the absolute value uses the convention U(origin) = 0.
   */
  template <int dim, typename NumberType> struct ReconstructedRawPotential {
    std::unique_ptr<dealii::FiniteElement<dim>> finite_element;
    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler;
    dealii::Vector<NumberType> values;
  };

  template <typename NumberType> struct ReconstructedRawPotential<0, NumberType> {
    dealii::Vector<NumberType> values;
  };

  /** @brief Value and first two derivatives of a reconstructed raw scalar potential at one point. */
  template <int dim, typename NumberType> struct RawPotentialEvaluation {
    NumberType value{};
    dealii::Tensor<1, dim, NumberType> gradient;
    dealii::Tensor<2, dim, NumberType> hessian;
  };

  template <typename NumberType> struct ReconstructedEoMPotential<0, NumberType> {
    dealii::Point<0> minimum;
    dealii::Vector<NumberType> values;
  };

  /**
   * @brief Result of finding an EoM point, optionally including its reconstructed potential.
   */
  template <int dim, typename NumberType> struct EoMResult {
    dealii::Point<dim> point;
    std::optional<ReconstructedEoMPotential<dim, NumberType>> potential;
  };

  namespace internal
  {
    using namespace dealii;

    template <int dim>
    dealii::Point<dim> get_origin(const dealii::DoFHandler<dim> &dof_handler,
                                  typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell)
    {
      using CellIterator = typename dealii::DoFHandler<dim>::cell_iterator;

      auto l1_norm = [](const Point<dim> &p) {
        double norm = 0.;
        for (uint d = 0; d < dim; ++d)
          norm += std::abs(p[d]);
        return norm;
      };

      std::vector<Point<dim>> candidates;

      auto iterate_cell = [&](const CellIterator &cell) {
        std::array<Point<dim>, GeometryInfo<dim>::vertices_per_cell> vertices;
        for (uint i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
          vertices[i] = cell->vertex(i);

        auto min_it = std::min_element(vertices.begin(), vertices.end(),
                                       [&](const auto &p1, const auto &p2) { return l1_norm(p1) < l1_norm(p2); });
        candidates.push_back(*min_it);
      };

      for (const auto &cell : dof_handler.active_cell_iterators())
        iterate_cell(cell);

      auto min_it = std::min_element(candidates.begin(), candidates.end(),
                                     [&](const auto &p1, const auto &p2) { return l1_norm(p1) < l1_norm(p2); });

      Point<dim> origin = *min_it;
      EoM_cell = GridTools::find_active_cell_around_point(dof_handler, origin);

      return origin;
    }

    template <int dim>
    typename dealii::DoFHandler<dim>::active_cell_iterator
    matching_dof_cell(const dealii::DoFHandler<dim> &target_dof_handler,
                      const typename dealii::DoFHandler<dim>::active_cell_iterator &source_cell)
    {
      using TriaCell = typename dealii::Triangulation<dim>::active_cell_iterator;

      auto target_cell = target_dof_handler.begin_active();
      target_cell->copy_from(*TriaCell(source_cell));
      return target_cell;
    }

    template <int dim> double l1_distance(const dealii::Point<dim> &a, const dealii::Point<dim> &b)
    {
      double distance = 0.;
      for (uint d = 0; d < dim; ++d)
        distance += std::abs(a[d] - b[d]);
      return distance;
    }

    template <int dim>
    double face_normal_cell_width(const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                                  const uint face_no)
    {
      const double face_measure = cell->face(face_no)->measure();
      if (!(face_measure > 0.) || !std::isfinite(face_measure))
        throw std::runtime_error("EoM potential reconstruction encountered an invalid face measure.");

      const double width = cell->measure() / face_measure;
      if (!(width > 0.) || !std::isfinite(width))
        throw std::runtime_error("EoM potential reconstruction encountered an invalid face-normal cell width.");
      return width;
    }

    template <int dim> double minimum_face_normal_cell_width(const dealii::DoFHandler<dim> &dof_handler)
    {
      double minimum_width = std::numeric_limits<double>::max();
      for (const auto &cell : dof_handler.active_cell_iterators())
        for (uint face_no = 0; face_no < cell->n_faces(); ++face_no)
          minimum_width = std::min(minimum_width, face_normal_cell_width<dim>(cell, face_no));

      if (!(minimum_width < std::numeric_limits<double>::max()))
        throw std::runtime_error("EoM potential reconstruction could not determine an initial cell width.");
      return minimum_width;
    }

    template <int dim>
    double resolve_potential_smoothing_length(const dealii::DoFHandler<dim> &dof_handler,
                                              const double configured_length)
    {
      if (!std::isfinite(configured_length) || configured_length < -1. ||
          (configured_length < 0. && configured_length != -1.))
        throw std::invalid_argument(
            "EoM_smoothing_length must be -1 (automatic), zero, or a positive physical length.");

      if (configured_length == -1.) return 2. * minimum_face_normal_cell_width(dof_handler);
      return configured_length;
    }

    template <int dim>
    Config::EoMConfig resolve_eom_config(const dealii::DoFHandler<dim> &dof_handler, Config::EoMConfig config)
    {
      config.validate();
      config.smoothing_length = resolve_potential_smoothing_length(dof_handler, config.smoothing_length);
      return config;
    }

    template <int dim, typename NumberType>
    dealii::types::global_dof_index select_gauge_dof(const dealii::DoFHandler<dim> &potential_dof_handler,
                                                     const dealii::AffineConstraints<NumberType> &constraints,
                                                     const dealii::Mapping<dim> &mapping,
                                                     const dealii::Point<dim> &origin)
    {
      const auto support_points = DoFTools::map_dofs_to_support_points(mapping, potential_dof_handler);

      dealii::types::global_dof_index best_dof = numbers::invalid_dof_index;
      double best_distance = std::numeric_limits<double>::max();
      for (const auto &dof_and_point : support_points) {
        const auto dof = dof_and_point.first;
        if (constraints.is_constrained(dof)) continue;

        const double distance = l1_distance(dof_and_point.second, origin);
        if (distance < best_distance) {
          best_distance = distance;
          best_dof = dof;
        }
      }

      if (best_dof == numbers::invalid_dof_index)
        throw std::runtime_error("EoM potential reconstruction could not find an unconstrained gauge DoF.");

      return best_dof;
    }

    template <int dim> std::unique_ptr<dealii::FiniteElement<dim>> make_eom_potential_fe()
    {
      return std::make_unique<FE_Q<dim>>(2);
    }

    template <int dim> std::unique_ptr<dealii::FiniteElement<dim>> make_raw_potential_fe()
    {
      return std::make_unique<FE_Q<dim>>(2);
    }

    template <int dim, typename EoMValue> dealii::Tensor<1, dim> eom_to_tensor(const EoMValue &eom)
    {
      dealii::Tensor<1, dim> out;
      for (uint d = 0; d < dim; ++d)
        out[d] = eom[d];
      return out;
    }

    template <int dim, typename NumberType> struct DG0GradientModel {
      dealii::Point<dim> center;
      dealii::Tensor<1, dim, NumberType> value;
      dealii::Tensor<2, dim, NumberType> jacobian;
    };

    template <int dim, typename VectorType, typename GradientFUN>
    std::vector<DG0GradientModel<dim, typename VectorType::value_type>>
    recover_dg0_gradient_models(const VectorType &sol, const dealii::DoFHandler<dim> &solution_dof_handler,
                                const dealii::Mapping<dim> &mapping, const GradientFUN &get_gradient)
    {
      using NumberType = typename VectorType::value_type;
      using Iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

      const uint n_cells = solution_dof_handler.get_triangulation().n_active_cells();
      std::vector<Iterator> cells(n_cells);
      std::vector<DG0GradientModel<dim, NumberType>> models(n_cells);

      dealii::QMidpoint<dim> midpoint;
      dealii::FEValues<dim> values(mapping, solution_dof_handler.get_fe(), midpoint,
                                   dealii::update_values | dealii::update_quadrature_points);
      std::vector<dealii::Vector<NumberType>> solution_values(
          1, dealii::Vector<NumberType>(solution_dof_handler.get_fe().n_components()));
      for (const auto &cell : solution_dof_handler.active_cell_iterators()) {
        const uint index = cell->active_cell_index();
        cells[index] = cell;
        values.reinit(cell);
        values.get_function_values(sol, solution_values);
        models[index].center = values.quadrature_point(0);
        models[index].value = eom_to_tensor<dim>(get_gradient(models[index].center, solution_values[0]));
      }

      const uint target_patch_size = std::min<uint>(n_cells, 2 * (dim + 1));
      for (uint cell_index = 0; cell_index < n_cells; ++cell_index) {
        std::vector<uint> patch{cell_index};
        std::vector<uint> frontier{cell_index};
        std::vector<bool> seen(n_cells, false);
        seen[cell_index] = true;

        while (patch.size() < target_patch_size && !frontier.empty()) {
          std::vector<uint> next_frontier;
          for (const uint index : frontier) {
            std::vector<Iterator> neighbors;
            dealii::GridTools::get_active_neighbors<dealii::DoFHandler<dim>>(cells[index], neighbors);
            for (const auto &neighbor : neighbors) {
              const uint neighbor_index = neighbor->active_cell_index();
              if (seen[neighbor_index]) continue;
              seen[neighbor_index] = true;
              patch.push_back(neighbor_index);
              next_frontier.push_back(neighbor_index);
            }
          }
          frontier = std::move(next_frontier);
        }

        if (patch.size() < dim + 1) continue;
        const double scale = std::max(cells[cell_index]->diameter(), std::numeric_limits<double>::epsilon());
        Eigen::MatrixXd design(patch.size(), dim + 1);
        for (uint row = 0; row < patch.size(); ++row) {
          design(row, 0) = 1.;
          for (uint d = 0; d < dim; ++d)
            design(row, d + 1) = (models[patch[row]].center[d] - models[cell_index].center[d]) / scale;
        }

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
        if (qr.rank() < dim + 1) continue;
        for (uint component = 0; component < dim; ++component) {
          Eigen::VectorXd samples(patch.size());
          for (uint row = 0; row < patch.size(); ++row)
            samples[row] = models[patch[row]].value[component];
          const Eigen::VectorXd coefficients = qr.solve(samples);
          for (uint d = 0; d < dim; ++d)
            models[cell_index].jacobian[component][d] = coefficients[d + 1] / scale;
        }
      }

      return models;
    }

    template <int dim, typename NumberType>
    dealii::Tensor<1, dim, NumberType> evaluate_dg0_gradient_model(const DG0GradientModel<dim, NumberType> &model,
                                                                   const dealii::Point<dim> &point)
    {
      auto value = model.value;
      for (uint component = 0; component < dim; ++component)
        for (uint d = 0; d < dim; ++d)
          value[component] += model.jacobian[component][d] * (point[d] - model.center[d]);
      return value;
    }

    template <int dim, typename NumberType> struct PotentialAssemblyScratch {
      PotentialAssemblyScratch(const dealii::Mapping<dim> &mapping, const dealii::FiniteElement<dim> &solution_fe,
                               const dealii::FiniteElement<dim> &potential_fe,
                               const dealii::Quadrature<dim> &quadrature,
                               const dealii::Quadrature<dim - 1> &face_quadrature)
          : solution_fe_values(mapping, solution_fe, quadrature, update_values | update_quadrature_points),
            potential_fe_values(mapping, potential_fe, quadrature,
                                update_gradients | update_JxW_values | update_quadrature_points),
            solution_fe_interface_values(mapping, solution_fe, face_quadrature,
                                         update_values | update_quadrature_points),
            potential_fe_interface_values(mapping, potential_fe, face_quadrature,
                                          update_values | update_gradients | update_JxW_values |
                                              update_quadrature_points | update_normal_vectors)
      {
        solution_values.resize(quadrature.size(), Vector<NumberType>(solution_fe.n_components()));
        solution_interface_values[0].resize(face_quadrature.size(), Vector<NumberType>(solution_fe.n_components()));
        solution_interface_values[1].resize(face_quadrature.size(), Vector<NumberType>(solution_fe.n_components()));
      }

      PotentialAssemblyScratch(const PotentialAssemblyScratch &scratch)
          : solution_fe_values(scratch.solution_fe_values.get_mapping(), scratch.solution_fe_values.get_fe(),
                               scratch.solution_fe_values.get_quadrature(),
                               scratch.solution_fe_values.get_update_flags()),
            potential_fe_values(scratch.potential_fe_values.get_mapping(), scratch.potential_fe_values.get_fe(),
                                scratch.potential_fe_values.get_quadrature(),
                                scratch.potential_fe_values.get_update_flags()),
            solution_fe_interface_values(scratch.solution_fe_interface_values.get_mapping(),
                                         scratch.solution_fe_interface_values.get_fe(),
                                         scratch.solution_fe_interface_values.get_quadrature(),
                                         scratch.solution_fe_interface_values.get_update_flags()),
            potential_fe_interface_values(scratch.potential_fe_interface_values.get_mapping(),
                                          scratch.potential_fe_interface_values.get_fe(),
                                          scratch.potential_fe_interface_values.get_quadrature(),
                                          scratch.potential_fe_interface_values.get_update_flags())
      {
        const uint n_components = scratch.solution_fe_values.get_fe().n_components();
        solution_values.resize(scratch.solution_fe_values.get_quadrature().size(), Vector<NumberType>(n_components));
        solution_interface_values[0].resize(scratch.solution_fe_interface_values.get_quadrature().size(),
                                            Vector<NumberType>(n_components));
        solution_interface_values[1].resize(scratch.solution_fe_interface_values.get_quadrature().size(),
                                            Vector<NumberType>(n_components));
      }

      FEValues<dim> solution_fe_values;
      FEValues<dim> potential_fe_values;
      FEInterfaceValues<dim> solution_fe_interface_values;
      FEInterfaceValues<dim> potential_fe_interface_values;
      std::vector<Vector<NumberType>> solution_values;
      std::array<std::vector<Vector<NumberType>>, 2> solution_interface_values;
    };

    template <typename NumberType> struct PotentialAssemblyCopy {
      struct FaceCopy {
        FullMatrix<NumberType> matrix;
        Vector<NumberType> rhs;
        std::vector<types::global_dof_index> dof_indices;
      };

      template <class Iterator> void reinit_cell(const Iterator &cell, const uint dofs_per_cell)
      {
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_rhs.reinit(dofs_per_cell);
        cell_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(cell_dof_indices);
        face_data.clear();
        face_data.reserve(cell->n_faces());
      }

      template <int dim> FaceCopy &new_face_data(const FEInterfaceValues<dim> &fe_interface_values)
      {
        const uint n_interface_dofs = fe_interface_values.n_current_interface_dofs();
        face_data.emplace_back();
        auto &data = face_data.back();
        data.matrix.reinit(n_interface_dofs, n_interface_dofs);
        data.rhs.reinit(n_interface_dofs);
        data.dof_indices = fe_interface_values.get_interface_dof_indices();
        return data;
      }

      FullMatrix<NumberType> cell_matrix;
      Vector<NumberType> cell_rhs;
      std::vector<types::global_dof_index> cell_dof_indices;
      std::vector<FaceCopy> face_data;
    };

    template <int dim, typename VectorType, typename EoMFUN>
    void assemble_potential_system(const VectorType &sol, const dealii::DoFHandler<dim> &solution_dof_handler,
                                   const dealii::DoFHandler<dim> &potential_dof_handler,
                                   const dealii::FiniteElement<dim> &potential_fe, const dealii::Mapping<dim> &mapping,
                                   const EoMFUN &get_EoM, const dealii::Quadrature<dim> &quadrature,
                                   const dealii::Quadrature<dim - 1> &face_quadrature,
                                   const dealii::AffineConstraints<typename VectorType::value_type> &constraints,
                                   dealii::SparseMatrix<typename VectorType::value_type> &matrix,
                                   dealii::Vector<typename VectorType::value_type> &rhs, const double smoothing_length)
    {
      using NumberType = typename VectorType::value_type;
      using Iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
      using Scratch = PotentialAssemblyScratch<dim, NumberType>;
      using Copy = PotentialAssemblyCopy<NumberType>;

      const FEValuesExtractors::Scalar scalar(0);
      const bool recover_dg0_gradient = solution_dof_handler.get_fe().degree == 0;
      const auto dg0_gradient_models = recover_dg0_gradient
                                           ? recover_dg0_gradient_models(sol, solution_dof_handler, mapping, get_EoM)
                                           : std::vector<DG0GradientModel<dim, NumberType>>{};

      const auto evaluate_gradient = [&](const Iterator &cell, const dealii::Point<dim> &point,
                                         const dealii::Vector<NumberType> &values) {
        if (recover_dg0_gradient)
          return evaluate_dg0_gradient_model(dg0_gradient_models[cell->active_cell_index()], point);
        return eom_to_tensor<dim>(get_EoM(point, values));
      };

      const auto cell_worker = [&](const Iterator &solution_cell, Scratch &scratch, Copy &copy) {
        const auto potential_cell = matching_dof_cell(potential_dof_handler, solution_cell);
        const uint dofs_per_cell = potential_fe.n_dofs_per_cell();

        scratch.solution_fe_values.reinit(solution_cell);
        scratch.potential_fe_values.reinit(potential_cell);
        scratch.solution_fe_values.get_function_values(sol, scratch.solution_values);

        copy.reinit_cell(potential_cell, dofs_per_cell);

        for (const auto q : scratch.potential_fe_values.quadrature_point_indices()) {
          const auto eom = evaluate_gradient(solution_cell, scratch.potential_fe_values.quadrature_point(q),
                                             scratch.solution_values[q]);

          for (uint i = 0; i < dofs_per_cell; ++i) {
            const auto grad_i = scratch.potential_fe_values.shape_grad(i, q);
            copy.cell_rhs(i) += scratch.potential_fe_values.JxW(q) * scalar_product(eom, grad_i);

            for (uint j = 0; j < dofs_per_cell; ++j)
              copy.cell_matrix(i, j) += scratch.potential_fe_values.JxW(q) *
                                        scalar_product(scratch.potential_fe_values.shape_grad(j, q), grad_i);
          }
        }
      };
      const auto boundary_worker = []([[maybe_unused]] const Iterator &cell, [[maybe_unused]] const uint &face_no,
                                      [[maybe_unused]] Scratch &scratch, [[maybe_unused]] Copy &copy) {};
      const auto copier = [&](const Copy &copy) {
        constraints.distribute_local_to_global(copy.cell_matrix, copy.cell_rhs, copy.cell_dof_indices, matrix, rhs);
        for (const auto &face : copy.face_data)
          constraints.distribute_local_to_global(face.matrix, face.rhs, face.dof_indices, matrix, rhs);
      };
      const auto face_worker = [&](const Iterator &solution_cell, const uint &face_no, const uint &subface_no,
                                   const Iterator &solution_neighbor, const uint &neighbor_face_no,
                                   const uint &neighbor_subface_no, Scratch &scratch, Copy &copy) {
        const auto potential_cell = matching_dof_cell(potential_dof_handler, solution_cell);
        const auto potential_neighbor = matching_dof_cell(potential_dof_handler, solution_neighbor);

        scratch.solution_fe_interface_values.reinit(solution_cell, face_no, subface_no, solution_neighbor,
                                                    neighbor_face_no, neighbor_subface_no);
        scratch.potential_fe_interface_values.reinit(potential_cell, face_no, subface_no, potential_neighbor,
                                                     neighbor_face_no, neighbor_subface_no);

        const auto &solution_fe_interface_values = scratch.solution_fe_interface_values;
        const auto &potential_fe_interface_values = scratch.potential_fe_interface_values;

        const uint n_interface_dofs = potential_fe_interface_values.n_current_interface_dofs();
        auto &face_data = copy.new_face_data(potential_fe_interface_values);
        const auto potential_view = potential_fe_interface_values[scalar];
        const auto &solution_fe_values_s = solution_fe_interface_values.get_fe_face_values(0);
        const auto &solution_fe_values_n = solution_fe_interface_values.get_fe_face_values(1);
        const auto &q_points = potential_fe_interface_values.get_quadrature_points();

        auto &solution_values_s = scratch.solution_interface_values[0];
        auto &solution_values_n = scratch.solution_interface_values[1];
        solution_fe_values_s.get_function_values(sol, solution_values_s);
        solution_fe_values_n.get_function_values(sol, solution_values_n);

        const double h_face = std::min(solution_cell->diameter(), solution_neighbor->diameter());
        const double h_normal = std::min(face_normal_cell_width<dim>(solution_cell, face_no),
                                         face_normal_cell_width<dim>(solution_neighbor, neighbor_face_no));
        const uint degree = potential_fe.degree;
        const double tau = 10. * (degree + 1.) * (degree + 1.) / h_face;
        const double gradient_jump_weight = smoothing_length * smoothing_length / h_normal;

        for (const auto q : potential_fe_interface_values.quadrature_point_indices()) {
          const auto normal = potential_fe_interface_values.normal_vector(q);
          const auto eom_s = evaluate_gradient(solution_cell, q_points[q], solution_values_s[q]);
          const auto eom_n = evaluate_gradient(solution_neighbor, q_points[q], solution_values_n[q]);
          const auto average_eom = 0.5 * (eom_s + eom_n);
          const double rhs_flux = scalar_product(average_eom, normal);

          for (uint i = 0; i < n_interface_dofs; ++i) {
            const double jump_i = potential_view.jump_in_values(i, q);
            const Tensor<1, dim> average_grad_i = potential_view.average_of_gradients(i, q);
            const double normal_gradient_jump_i = scalar_product(potential_view.jump_in_gradients(i, q), normal);

            face_data.rhs(i) += -potential_fe_interface_values.JxW(q) * rhs_flux * jump_i;

            for (uint j = 0; j < n_interface_dofs; ++j) {
              const double jump_j = potential_view.jump_in_values(j, q);
              const Tensor<1, dim> average_grad_j = potential_view.average_of_gradients(j, q);
              const double normal_gradient_jump_j = scalar_product(potential_view.jump_in_gradients(j, q), normal);

              face_data.matrix(i, j) +=
                  potential_fe_interface_values.JxW(q) *
                  (-scalar_product(average_grad_j, normal) * jump_i - scalar_product(average_grad_i, normal) * jump_j +
                   tau * jump_j * jump_i + gradient_jump_weight * normal_gradient_jump_i * normal_gradient_jump_j);
            }
          }
        }
      };

      Scratch scratch(mapping, solution_dof_handler.get_fe(), potential_fe, quadrature, face_quadrature);
      Copy copy;
      const MeshWorker::AssembleFlags flags = MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
                                              MeshWorker::assemble_own_interior_faces_once;

      MeshWorker::mesh_loop(solution_dof_handler.begin_active(), solution_dof_handler.end(), cell_worker, copier,
                            scratch, copy, flags, boundary_worker, face_worker);
    }

    template <int dim, typename NumberType> struct PotentialMinimum {
      dealii::Point<dim> point;
      double value = std::numeric_limits<double>::max();
    };

    template <int dim> struct LocalPotentialEvaluation {
      double value = 0.;
      dealii::Tensor<1, dim> gradient;
      dealii::Tensor<2, dim> hessian;
    };

    template <int dim>
    LocalPotentialEvaluation<dim> evaluate_local_potential(const dealii::FiniteElement<dim> &potential_fe,
                                                           const std::vector<double> &local_values,
                                                           const dealii::Point<dim> &point)
    {
      LocalPotentialEvaluation<dim> evaluation;
      for (uint i = 0; i < potential_fe.n_dofs_per_cell(); ++i) {
        evaluation.value += local_values[i] * potential_fe.shape_value(i, point);
        evaluation.gradient += local_values[i] * potential_fe.shape_grad(i, point);
        evaluation.hessian += local_values[i] * potential_fe.shape_grad_grad(i, point);
      }
      return evaluation;
    }

    template <int dim>
    bool projected_newton_direction(const dealii::Tensor<2, dim> &hessian,
                                    const dealii::Tensor<1, dim> &projected_gradient,
                                    const std::array<bool, dim> &fixed, dealii::Tensor<1, dim> &direction)
    {
      std::array<uint, dim> free_indices{};
      uint n_free = 0;
      for (uint d = 0; d < dim; ++d)
        if (!fixed[d]) free_indices[n_free++] = d;

      if (n_free == 0) return false;

      Eigen::MatrixXd reduced_hessian(n_free, n_free);
      Eigen::VectorXd reduced_gradient(n_free);
      double hessian_norm = 0.;
      for (uint i = 0; i < n_free; ++i) {
        reduced_gradient[i] = projected_gradient[free_indices[i]];
        for (uint j = 0; j < n_free; ++j) {
          reduced_hessian(i, j) = hessian[free_indices[i]][free_indices[j]];
          hessian_norm = std::max(hessian_norm, std::abs(reduced_hessian(i, j)));
        }
      }
      if (!reduced_hessian.allFinite() || !reduced_gradient.allFinite()) return false;

      const double positivity_threshold = 100. * std::numeric_limits<double>::epsilon() * std::max(1., hessian_norm);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(reduced_hessian);
      if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
          !(solver.eigenvalues().minCoeff() > positivity_threshold))
        return false;

      const Eigen::VectorXd reduced_direction =
          -solver.eigenvectors() *
          (solver.eigenvectors().transpose() * reduced_gradient).cwiseQuotient(solver.eigenvalues());
      if (!reduced_direction.allFinite()) return false;

      direction = {};
      for (uint i = 0; i < n_free; ++i)
        direction[free_indices[i]] = reduced_direction[i];
      return true;
    }

    template <int dim, typename NumberType>
    PotentialMinimum<dim, NumberType>
    refine_cell_minimum(const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
                        const dealii::FiniteElement<dim> &potential_fe, const dealii::Mapping<dim> &mapping,
                        const dealii::Vector<NumberType> &potential, const dealii::Quadrature<dim> &quadrature,
                        const Config::EoMConfig &config,
                        const std::optional<dealii::Point<dim>> &initial_unit_point = std::nullopt)
    {
      std::vector<dealii::types::global_dof_index> local_dof_indices(potential_fe.n_dofs_per_cell());
      cell->get_dof_indices(local_dof_indices);

      std::vector<double> local_values(potential_fe.n_dofs_per_cell());
      for (uint i = 0; i < potential_fe.n_dofs_per_cell(); ++i)
        local_values[i] = (double)potential[local_dof_indices[i]];

      dealii::Point<dim> sampled_point = quadrature.point(0);
      auto sampled_evaluation = evaluate_local_potential(potential_fe, local_values, sampled_point);
      const auto consider_seed = [&](const dealii::Point<dim> &candidate) {
        const auto candidate_evaluation = evaluate_local_potential(potential_fe, local_values, candidate);
        if (candidate_evaluation.value < sampled_evaluation.value) {
          sampled_point = candidate;
          sampled_evaluation = candidate_evaluation;
        }
      };

      for (const auto &support_point : potential_fe.get_unit_support_points())
        consider_seed(support_point);
      for (const auto &quadrature_point : quadrature.get_points())
        consider_seed(quadrature_point);

      const double gradient_tolerance = std::max(config.abs_tol, 100. * std::numeric_limits<double>::epsilon());

      const auto refine_from_seed = [&](dealii::Point<dim> point) {
        auto evaluation = evaluate_local_potential(potential_fe, local_values, point);
        for (uint iteration = 0; iteration < config.max_iter; ++iteration) {
          std::array<bool, dim> fixed{};
          dealii::Tensor<1, dim> projected_gradient;
          double projected_gradient_norm = 0.;
          for (uint d = 0; d < dim; ++d) {
            const bool fixed_at_lower = point[d] <= config.bound_tolerance && evaluation.gradient[d] > 0.;
            const bool fixed_at_upper = point[d] >= 1. - config.bound_tolerance && evaluation.gradient[d] < 0.;
            fixed[d] = fixed_at_lower || fixed_at_upper;
            projected_gradient[d] = fixed[d] ? 0. : evaluation.gradient[d];
            const double extent = std::max(cell->extent_in_direction(d), std::numeric_limits<double>::epsilon());
            projected_gradient_norm = std::max(projected_gradient_norm, std::abs(projected_gradient[d]) / extent);
          }
          if (projected_gradient_norm <= gradient_tolerance) break;

          dealii::Tensor<1, dim> direction;
          bool used_newton = projected_newton_direction<dim>(evaluation.hessian, projected_gradient, fixed, direction);
          double directional_derivative = dealii::scalar_product(evaluation.gradient, direction);
          if (!used_newton || !(directional_derivative < 0.) || !std::isfinite(directional_derivative)) {
            direction = -projected_gradient;
            directional_derivative = dealii::scalar_product(evaluation.gradient, direction);
            used_newton = false;
          }
          if (!(directional_derivative < 0.) || !std::isfinite(directional_derivative)) break;

          const auto try_step = [&](const dealii::Tensor<1, dim> &search_direction, const double search_derivative,
                                    dealii::Point<dim> &accepted_point,
                                    LocalPotentialEvaluation<dim> &accepted_evaluation) {
            double alpha = 1.;
            for (uint d = 0; d < dim; ++d) {
              if (search_direction[d] > 0.)
                alpha = std::min(alpha, (1. - point[d]) / search_direction[d]);
              else if (search_direction[d] < 0.)
                alpha = std::min(alpha, -point[d] / search_direction[d]);
            }
            if (!(alpha > 0.) || !std::isfinite(alpha)) return false;

            for (uint backtrack = 0; backtrack < config.max_backtracks; ++backtrack) {
              auto candidate = point;
              for (uint d = 0; d < dim; ++d)
                candidate[d] = std::clamp(point[d] + alpha * search_direction[d], 0., 1.);
              const auto candidate_evaluation = evaluate_local_potential(potential_fe, local_values, candidate);
              if (candidate_evaluation.value <=
                  evaluation.value + config.armijo_coefficient * alpha * search_derivative) {
                accepted_point = candidate;
                accepted_evaluation = candidate_evaluation;
                return true;
              }
              alpha *= 0.5;
            }
            return false;
          };

          dealii::Point<dim> next_point;
          LocalPotentialEvaluation<dim> next_evaluation;
          bool accepted = try_step(direction, directional_derivative, next_point, next_evaluation);
          if (!accepted && used_newton) {
            direction = -projected_gradient;
            directional_derivative = dealii::scalar_product(evaluation.gradient, direction);
            if (directional_derivative < 0.)
              accepted = try_step(direction, directional_derivative, next_point, next_evaluation);
          }
          if (!accepted || !(next_evaluation.value < evaluation.value)) break;

          point = next_point;
          evaluation = next_evaluation;
        }
        return std::pair{point, evaluation};
      };

      auto [point, evaluation] = refine_from_seed(sampled_point);
      if (initial_unit_point) {
        auto warm_start = *initial_unit_point;
        bool valid = true;
        for (uint d = 0; d < dim; ++d) {
          valid = valid && std::isfinite(warm_start[d]) && warm_start[d] >= -config.bound_tolerance &&
                  warm_start[d] <= 1. + config.bound_tolerance;
          warm_start[d] = std::clamp(warm_start[d], 0., 1.);
        }
        if (valid) {
          auto [warm_point, warm_evaluation] = refine_from_seed(warm_start);
          if (warm_evaluation.value < evaluation.value) {
            point = warm_point;
            evaluation = warm_evaluation;
          }
        }
      }

      return {.point = mapping.transform_unit_to_real_cell(cell, point), .value = evaluation.value};
    }

    template <int dim, typename NumberType>
    PotentialMinimum<dim, NumberType>
    find_potential_minimum(const dealii::DoFHandler<dim> &potential_dof_handler,
                           const dealii::FiniteElement<dim> &potential_fe, const dealii::Mapping<dim> &mapping,
                           const dealii::Vector<NumberType> &potential, const dealii::Quadrature<dim> &quadrature,
                           const Config::EoMConfig &config,
                           const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt)
    {
      PotentialMinimum<dim, NumberType> minimum;

      const auto support_points = DoFTools::map_dofs_to_support_points(mapping, potential_dof_handler);
      for (const auto &dof_and_point : support_points) {
        const auto dof = dof_and_point.first;
        if ((double)potential[dof] < minimum.value) {
          minimum.value = (double)potential[dof];
          minimum.point = dof_and_point.second;
        }
      }

      FEValues<dim> potential_fe_values(mapping, potential_fe, quadrature, update_values | update_quadrature_points);
      std::vector<NumberType> potential_values(quadrature.size());
      for (const auto &cell : potential_dof_handler.active_cell_iterators()) {
        potential_fe_values.reinit(cell);
        potential_fe_values.get_function_values(potential, potential_values);

        for (const auto q : potential_fe_values.quadrature_point_indices()) {
          if ((double)potential_values[q] < minimum.value) {
            minimum.value = (double)potential_values[q];
            minimum.point = potential_fe_values.quadrature_point(q);
          }
        }
      }

      const auto minimum_cell =
          GridTools::find_active_cell_around_point(mapping, potential_dof_handler, minimum.point).first;
      using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
      struct CandidateCell {
        CellIterator cell;
        std::optional<dealii::Point<dim>> initial_unit_point;
      };
      std::vector<CandidateCell> candidate_cells;
      const auto add_candidate = [&](const CellIterator &cell,
                                     const std::optional<dealii::Point<dim>> &initial_unit_point = std::nullopt) {
        const auto candidate = std::find_if(candidate_cells.begin(), candidate_cells.end(),
                                            [&](const auto &entry) { return entry.cell == cell; });
        if (candidate == candidate_cells.end())
          candidate_cells.push_back({cell, initial_unit_point});
        else if (initial_unit_point)
          candidate->initial_unit_point = initial_unit_point;
      };
      const auto add_cell_and_neighbors = [&](const CellIterator &cell) {
        add_candidate(cell);
        std::vector<CellIterator> active_neighbors;
        GridTools::get_active_neighbors<dealii::DoFHandler<dim>>(cell, active_neighbors);
        for (const auto &neighbor : active_neighbors)
          add_candidate(neighbor);
      };

      add_cell_and_neighbors(minimum_cell);
      if (initial_guess) {
        bool finite = true;
        for (uint d = 0; d < dim; ++d)
          finite = finite && std::isfinite((*initial_guess)[d]);
        const auto bounding_box = GridTools::compute_bounding_box(potential_dof_handler.get_triangulation());
        if (finite && bounding_box.point_inside(*initial_guess)) {
          try {
            const auto initial_cell =
                GridTools::find_active_cell_around_point(mapping, potential_dof_handler, *initial_guess).first;
            if (initial_cell != potential_dof_handler.end()) {
              add_cell_and_neighbors(initial_cell);
              add_candidate(initial_cell, mapping.transform_real_to_unit_cell(initial_cell, *initial_guess));
            }
          } catch (const dealii::ExceptionBase &) {
          }
        }
      }

      for (const auto &candidate : candidate_cells) {
        const auto &cell = candidate.cell;
        const auto refined = refine_cell_minimum<dim, NumberType>(cell, potential_fe, mapping, potential, quadrature,
                                                                  config, candidate.initial_unit_point);
        if (refined.value < minimum.value) minimum = refined;
      }

      return minimum;
    }

    template <int dim, typename VectorType, typename GradientFUN>
    ReconstructedRawPotential<dim, typename VectorType::value_type>
    solve_potential(const VectorType &sol, const dealii::DoFHandler<dim> &solution_dof_handler,
                    const dealii::Mapping<dim> &mapping, const GradientFUN &get_gradient,
                    const Config::EoMConfig &config, std::unique_ptr<dealii::FiniteElement<dim>> potential_fe)
    {
      using NumberType = typename VectorType::value_type;

      auto origin_cell = solution_dof_handler.begin_active();
      const auto origin = get_origin(solution_dof_handler, origin_cell);
      const double smoothing_length = resolve_potential_smoothing_length(solution_dof_handler, config.smoothing_length);

      auto potential_dof_handler = std::make_unique<DoFHandler<dim>>(solution_dof_handler.get_triangulation());
      potential_dof_handler->distribute_dofs(*potential_fe);

      AffineConstraints<NumberType> constraints;
      DoFTools::make_hanging_node_constraints(*potential_dof_handler, constraints);
      const auto gauge_dof = select_gauge_dof(*potential_dof_handler, constraints, mapping, origin);
      constraints.add_line(gauge_dof);
      constraints.set_inhomogeneity(gauge_dof, 0.);
      constraints.close();

      DynamicSparsityPattern dsp(potential_dof_handler->n_dofs());
      DoFTools::make_flux_sparsity_pattern(*potential_dof_handler, dsp, constraints,
                                           /*keep_constrained_dofs = */ true);

      SparsityPattern sparsity_pattern;
      sparsity_pattern.copy_from(dsp);
      SparseMatrix<NumberType> matrix(sparsity_pattern);
      Vector<NumberType> rhs(potential_dof_handler->n_dofs());

      const uint quadrature_order =
          std::max<uint>(std::max<uint>(solution_dof_handler.get_fe().degree, potential_fe->degree) + 2, 2);
      QGauss<dim> quadrature(quadrature_order);
      QGauss<dim - 1> face_quadrature(quadrature_order);

      assemble_potential_system(sol, solution_dof_handler, *potential_dof_handler, *potential_fe, mapping, get_gradient,
                                quadrature, face_quadrature, constraints, matrix, rhs, smoothing_length);

      SparseDirectUMFPACK solver;
      solver.initialize(matrix);

      Vector<NumberType> potential(potential_dof_handler->n_dofs());
      solver.vmult(potential, rhs);
      constraints.distribute(potential);

      return {.finite_element = std::move(potential_fe),
              .dof_handler = std::move(potential_dof_handler),
              .values = std::move(potential)};
    }

    template <int dim, typename VectorType, typename EoMFUN>
    ReconstructedEoMPotential<dim, typename VectorType::value_type>
    reconstruct_potential(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                          const dealii::DoFHandler<dim> &solution_dof_handler, const dealii::Mapping<dim> &mapping,
                          const EoMFUN &get_EoM, const Config::EoMConfig &config,
                          const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt)
    {
      auto potential =
          solve_potential(sol, solution_dof_handler, mapping, get_EoM, config, make_eom_potential_fe<dim>());

      const uint quadrature_order =
          std::max<uint>(std::max<uint>(solution_dof_handler.get_fe().degree, potential.finite_element->degree) + 2, 2);
      QGauss<dim> quadrature(quadrature_order);

      const auto minimum = find_potential_minimum(*potential.dof_handler, *potential.finite_element, mapping,
                                                  potential.values, quadrature, config, initial_guess);
      EoM_cell = GridTools::find_active_cell_around_point(solution_dof_handler, minimum.point);
      return {.minimum = minimum.point,
              .finite_element = std::move(potential.finite_element),
              .dof_handler = std::move(potential.dof_handler),
              .values = std::move(potential.values)};
    }

  } // namespace internal

  /**
   * @brief Reconstruct a scalar raw potential without locating its minimum.
   *
   * The supplied callback must return the unmodified gradient of the desired scalar potential. Unlike an EoM callback,
   * it must not include explicit-breaking or other terms which should be absent from readouts and extractors.
   */
  template <int dim, typename VectorType, typename GradientFUN>
  ReconstructedRawPotential<dim, typename VectorType::value_type>
  reconstruct_raw_potential(const VectorType &sol, const dealii::DoFHandler<dim> &dof_handler,
                            const dealii::Mapping<dim> &mapping, const GradientFUN &get_gradient,
                            const Config::EoMConfig &config)
  {
    static_assert(dim > 0, "A raw spatial potential cannot be reconstructed in zero dimensions.");
    config.validate();
    return internal::solve_potential(sol, dof_handler, mapping, get_gradient, config,
                                     internal::make_raw_potential_fe<dim>());
  }

  /** @brief Evaluate a reconstructed raw potential and its first two derivatives at a real-space point. */
  template <int dim, typename NumberType>
  RawPotentialEvaluation<dim, NumberType>
  evaluate_raw_potential(const ReconstructedRawPotential<dim, NumberType> &potential,
                         const dealii::Mapping<dim> &mapping, const dealii::Point<dim> &point)
  {
    const auto cell = dealii::GridTools::find_active_cell_around_point(mapping, *potential.dof_handler, point).first;
    const auto unit_point = mapping.transform_real_to_unit_cell(cell, point);
    dealii::FEValues<dim> fe_values(mapping, *potential.finite_element, unit_point,
                                    dealii::update_values | dealii::update_gradients | dealii::update_hessians);
    fe_values.reinit(cell);

    std::vector<NumberType> values(1);
    std::vector<dealii::Tensor<1, dim, NumberType>> gradients(1);
    std::vector<dealii::Tensor<2, dim, NumberType>> hessians(1);
    fe_values.get_function_values(potential.values, values);
    fe_values.get_function_gradients(potential.values, gradients);
    fe_values.get_function_hessians(potential.values, hessians);
    return {.value = values[0], .gradient = gradients[0], .hessian = hessians[0]};
  }

  /**
   * @brief Reconstruct a potential whose gradient approximates the model EoM vector field and return a sampled and
   * locally refined minimum of that potential.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  EoMResult<dim, typename VectorType::value_type>
  get_EoM_point_with_potential(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                               const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping,
                               const EoMFUN &get_EoM, const EoMPFUN &EoM_postprocess, const Config::EoMConfig &config,
                               const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt)
  {
    config.validate();
    if (config.max_iter == 0) return {.point = internal::get_origin(dof_handler, EoM_cell), .potential = std::nullopt};

    auto potential =
        internal::reconstruct_potential(EoM_cell, sol, dof_handler, mapping, get_EoM, config, initial_guess);
    auto EoM = potential.minimum;

    dealii::Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
    dealii::Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);
    fe_function.set_active_cell(EoM_cell);
    fe_function.vector_value(EoM, values);

    EoM = EoM_postprocess(EoM, values);
    EoM_cell = dealii::GridTools::find_active_cell_around_point(dof_handler, EoM);

    return {.point = EoM, .potential = std::move(potential)};
  }

  /** Compatibility overload accepting the historical scalar EoM settings. */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  EoMResult<dim, typename VectorType::value_type> get_EoM_point_with_potential(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, [[maybe_unused]] const auto &values) { return p; },
      const double EoM_abs_tol = Config::EoMConfig::default_abs_tol,
      const uint max_iter = Config::EoMConfig::default_max_iter,
      const double EoM_smoothing_length = Config::EoMConfig::default_smoothing_length,
      const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt)
  {
    return get_EoM_point_with_potential(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess,
                                        Config::EoMConfig(EoM_abs_tol, max_iter, EoM_smoothing_length), initial_guess);
  }

  /**
   * @brief Reconstruct a potential whose gradient approximates the model EoM vector field and return a sampled and
   * locally refined minimum of that potential.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim> get_EoM_point(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                                   const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping,
                                   const EoMFUN &get_EoM, const EoMPFUN &EoM_postprocess,
                                   const Config::EoMConfig &config,
                                   const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt)
  {
    return get_EoM_point_with_potential(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess, config,
                                        initial_guess)
        .point;
  }

  /** Compatibility overload accepting the historical scalar EoM settings. */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim> get_EoM_point(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, [[maybe_unused]] const auto &values) { return p; },
      const double EoM_abs_tol = Config::EoMConfig::default_abs_tol,
      const uint max_iter = Config::EoMConfig::default_max_iter,
      const double EoM_smoothing_length = Config::EoMConfig::default_smoothing_length,
      const std::optional<dealii::Point<dim>> &initial_guess = std::nullopt)
  {
    return get_EoM_point(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess,
                         Config::EoMConfig(EoM_abs_tol, max_iter, EoM_smoothing_length), initial_guess);
  }
} // namespace DiFfRG
