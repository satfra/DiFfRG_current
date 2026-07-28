#pragma once

// external libraries
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/numerics/fe_field_function.h>

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

    template <int dim>
    std::unique_ptr<dealii::FiniteElement<dim>> make_potential_fe(const dealii::FiniteElement<dim> &solution_fe)
    {
      const uint degree = solution_fe.degree;
      const bool is_dg = !solution_fe.conforms(FiniteElementData<dim>::H1);

      if (is_dg)
        return std::make_unique<FE_DGQ<dim>>(degree);
      else
        return std::make_unique<FE_Q<dim>>(degree);
    }

    template <int dim, typename EoMValue> dealii::Tensor<1, dim> eom_to_tensor(const EoMValue &eom)
    {
      dealii::Tensor<1, dim> out;
      for (uint d = 0; d < dim; ++d)
        out[d] = eom[d];
      return out;
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
                                   dealii::Vector<typename VectorType::value_type> &rhs)
    {
      using NumberType = typename VectorType::value_type;
      using Iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;
      using Scratch = PotentialAssemblyScratch<dim, NumberType>;
      using Copy = PotentialAssemblyCopy<NumberType>;

      const FEValuesExtractors::Scalar scalar(0);

      const auto cell_worker = [&](const Iterator &solution_cell, Scratch &scratch, Copy &copy) {
        const auto potential_cell = matching_dof_cell(potential_dof_handler, solution_cell);
        const uint dofs_per_cell = potential_fe.n_dofs_per_cell();

        scratch.solution_fe_values.reinit(solution_cell);
        scratch.potential_fe_values.reinit(potential_cell);
        scratch.solution_fe_values.get_function_values(sol, scratch.solution_values);

        copy.reinit_cell(potential_cell, dofs_per_cell);

        for (const auto q : scratch.potential_fe_values.quadrature_point_indices()) {
          const auto eom =
              eom_to_tensor<dim>(get_EoM(scratch.potential_fe_values.quadrature_point(q), scratch.solution_values[q]));

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
        const uint degree = potential_fe.degree;
        const double tau = 10. * (degree + 1.) * (degree + 1.) / h_face;

        for (const auto q : potential_fe_interface_values.quadrature_point_indices()) {
          const auto normal = potential_fe_interface_values.normal_vector(q);
          const auto eom_s = eom_to_tensor<dim>(get_EoM(q_points[q], solution_values_s[q]));
          const auto eom_n = eom_to_tensor<dim>(get_EoM(q_points[q], solution_values_n[q]));
          const auto average_eom = 0.5 * (eom_s + eom_n);
          const double rhs_flux = scalar_product(average_eom, normal);

          for (uint i = 0; i < n_interface_dofs; ++i) {
            const double jump_i = potential_view.jump_in_values(i, q);
            const Tensor<1, dim> average_grad_i = potential_view.average_of_gradients(i, q);

            face_data.rhs(i) += -potential_fe_interface_values.JxW(q) * rhs_flux * jump_i;

            for (uint j = 0; j < n_interface_dofs; ++j) {
              const double jump_j = potential_view.jump_in_values(j, q);
              const Tensor<1, dim> average_grad_j = potential_view.average_of_gradients(j, q);

              face_data.matrix(i, j) += potential_fe_interface_values.JxW(q) *
                                        (-scalar_product(average_grad_j, normal) * jump_i -
                                         scalar_product(average_grad_i, normal) * jump_j + tau * jump_j * jump_i);
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

    template <int dim, typename NumberType>
    PotentialMinimum<dim, NumberType>
    find_potential_minimum(const dealii::DoFHandler<dim> &potential_dof_handler,
                           const dealii::FiniteElement<dim> &potential_fe, const dealii::Mapping<dim> &mapping,
                           const dealii::Vector<NumberType> &potential, const dealii::Quadrature<dim> &quadrature)
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

      return minimum;
    }

    template <int dim, typename VectorType, typename EoMFUN>
    ReconstructedEoMPotential<dim, typename VectorType::value_type>
    reconstruct_potential(typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
                          const dealii::DoFHandler<dim> &solution_dof_handler, const dealii::Mapping<dim> &mapping,
                          const EoMFUN &get_EoM)
    {
      using NumberType = typename VectorType::value_type;

      auto origin_cell = EoM_cell;
      const auto origin = get_origin(solution_dof_handler, origin_cell);

      auto potential_fe = make_potential_fe<dim>(solution_dof_handler.get_fe());

      auto potential_dof_handler =
          std::make_unique<DoFHandler<dim>>(solution_dof_handler.get_triangulation());
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

      const uint quadrature_order = std::max<uint>(potential_fe->degree + 2, 2);
      QGauss<dim> quadrature(quadrature_order);
      QGauss<dim - 1> face_quadrature(quadrature_order);

      assemble_potential_system(sol, solution_dof_handler, *potential_dof_handler, *potential_fe, mapping, get_EoM,
                                quadrature, face_quadrature, constraints, matrix, rhs);

      SparseDirectUMFPACK solver;
      solver.initialize(matrix);

      Vector<NumberType> potential(potential_dof_handler->n_dofs());
      solver.vmult(potential, rhs);
      constraints.distribute(potential);

      const auto minimum =
          find_potential_minimum(*potential_dof_handler, *potential_fe, mapping, potential, quadrature);
      EoM_cell = GridTools::find_active_cell_around_point(solution_dof_handler, minimum.point);
      return {.minimum = minimum.point,
              .finite_element = std::move(potential_fe),
              .dof_handler = std::move(potential_dof_handler),
              .values = std::move(potential)};
    }

  } // namespace internal

  /**
   * @brief Reconstruct a potential whose gradient approximates the model EoM vector field and return a discrete
   * minimum of that potential.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  EoMResult<dim, typename VectorType::value_type> get_EoM_point_with_potential(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, [[maybe_unused]] const auto &values) { return p; },
      [[maybe_unused]] const double EoM_abs_tol = 1e-5, const uint max_iter = 100)
  {
    if (max_iter == 0) return {.point = internal::get_origin(dof_handler, EoM_cell), .potential = std::nullopt};

    auto potential = internal::reconstruct_potential(EoM_cell, sol, dof_handler, mapping, get_EoM);
    auto EoM = potential.minimum;

    dealii::Vector<typename VectorType::value_type> values(dof_handler.get_fe().n_components());
    dealii::Functions::FEFieldFunction<dim, VectorType> fe_function(dof_handler, sol, mapping);
    fe_function.set_active_cell(EoM_cell);
    fe_function.vector_value(EoM, values);

    EoM = EoM_postprocess(EoM, values);
    EoM_cell = dealii::GridTools::find_active_cell_around_point(dof_handler, EoM);

    return {.point = EoM, .potential = std::move(potential)};
  }

  /**
   * @brief Reconstruct a potential whose gradient approximates the model EoM vector field and return a discrete
   * minimum of that potential.
   */
  template <int dim, typename VectorType, typename EoMFUN, typename EoMPFUN>
  dealii::Point<dim> get_EoM_point(
      typename dealii::DoFHandler<dim>::cell_iterator &EoM_cell, const VectorType &sol,
      const dealii::DoFHandler<dim> &dof_handler, const dealii::Mapping<dim> &mapping, const EoMFUN &get_EoM,
      const EoMPFUN &EoM_postprocess = [](const auto &p, [[maybe_unused]] const auto &values) { return p; },
      const double EoM_abs_tol = 1e-5, const uint max_iter = 100)
  {
    return get_EoM_point_with_potential(EoM_cell, sol, dof_handler, mapping, get_EoM, EoM_postprocess, EoM_abs_tol,
                                        max_iter)
        .point;
  }
} // namespace DiFfRG
