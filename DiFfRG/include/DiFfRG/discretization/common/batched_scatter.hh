#pragma once

// DiFfRG
#include <DiFfRG/physics/integration/map_scheduler.hh>

// external libraries
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/tbb.h>

// standard library
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief The locally owned cells of a batched assembler, colored for its scatter.
     *
     * In the scatter every cell adds into the rows of its own dofs and, through the constraints, into
     * those of their constraint masters. No two cells of a color share such a row, so the cells of one color
     * can add into a global matrix or vector concurrently. Columns do not matter: concurrent additions to
     * different rows never touch the same entry.
     */
    template <int dim> class ColoredCells
    {
    public:
      using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

      template <typename NumberType>
      void reinit(const dealii::DoFHandler<dim> &dof_handler, const dealii::AffineConstraints<NumberType> &constraints)
      {
        cells.clear();
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned()) cells.push_back(cell);

        // Greedy coloring, one bit per color and row.
        std::vector<std::uint64_t> row_colors(dof_handler.n_dofs(), 0);
        std::vector<dealii::types::global_dof_index> rows;
        colors.clear();
        constrained.assign(cells.size(), false);
        for (size_t k = 0; k < cells.size(); ++k) {
          rows.resize(cells[k]->get_fe().n_dofs_per_cell());
          cells[k]->get_dof_indices(rows);
          const size_t n_own = rows.size();
          for (size_t i = 0; i < n_own; ++i)
            if (constraints.is_constrained(rows[i])) {
              constrained[k] = true;
              if (const auto *entries = constraints.get_constraint_entries(rows[i]))
                for (const auto &entry : *entries)
                  rows.push_back(entry.first);
            }
          std::uint64_t taken = 0;
          for (const auto row : rows)
            taken |= row_colors[row];
          if (~taken == 0) throw std::runtime_error("ColoredCells: more than 64 cell colors needed.");
          const uint color = std::countr_one(taken);
          for (const auto row : rows)
            row_colors[row] |= std::uint64_t(1) << color;
          if (color >= colors.size()) colors.resize(color + 1);
          colors[color].push_back(k);
        }
      }

      size_t size() const { return cells.size(); }
      const CellIterator &operator[](const size_t k) const { return cells[k]; }
      const std::vector<CellIterator> &all() const { return cells; }
      /// Whether a dof of cell k is constrained, i.e. its insertion has to go through the constraints.
      bool is_constrained(const size_t k) const { return constrained[k]; }

      /**
       * @brief assemble(k, scratch, local) for every cell k, and insert(local) the result into @p global.
       *
       * deal.II's serial vector and matrix are written concurrently, one parallel loop per color. Other types
       * (PETSc, block types) are assembled in parallel into @p buffer and inserted serially in the same order.
       * Either way the result does not depend on the thread count. Ends with global.compress(add).
       *
       * @param local the member of Scratch that holds one cell's result
       */
      template <typename Global, typename Scratch, typename Local, typename Assemble, typename Insert>
      void scatter(Global &global, tbb::enumerable_thread_specific<Scratch> &scratch, Local Scratch::*local,
                   std::vector<Local> &buffer, const Assemble &assemble, const Insert &insert) const
      {
        using NT = typename Global::value_type;
        constexpr bool concurrent =
            std::is_same_v<Global, dealii::Vector<NT>> || std::is_same_v<Global, dealii::SparseMatrix<NT>>;
        // map() is collective and each rank visits only its own cells; see NoMapsHere.
        const NoMapsHere no_maps_during_assembly;
        if constexpr (concurrent) {
          for (const auto &color : colors)
            tbb::parallel_for(tbb::blocked_range<size_t>(0, color.size()), [&](const tbb::blocked_range<size_t> &r) {
              auto &s = scratch.local();
              for (size_t i = r.begin(); i != r.end(); ++i) {
                assemble(color[i], s, s.*local);
                insert(s.*local);
              }
            });
        } else {
          buffer.resize(cells.size());
          tbb::parallel_for(tbb::blocked_range<size_t>(0, cells.size()), [&](const tbb::blocked_range<size_t> &r) {
            auto &s = scratch.local();
            for (size_t k = r.begin(); k != r.end(); ++k)
              assemble(k, s, buffer[k]);
          });
          for (const auto &color : colors)
            for (const size_t k : color)
              insert(buffer[k]);
        }
        global.compress(dealii::VectorOperation::add);
      }

    private:
      std::vector<CellIterator> cells;
      /// Cell indices k by color.
      std::vector<std::vector<size_t>> colors;
      std::vector<bool> constrained;
    };
  } // namespace internal
} // namespace DiFfRG
