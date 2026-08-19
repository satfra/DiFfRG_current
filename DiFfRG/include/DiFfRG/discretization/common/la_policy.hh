#pragma once

// external libraries
#include <deal.II/base/index_set.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#endif

// DiFfRG
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/types.hh>

// std
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Whether a linear algebra type distributes its rows across MPI ranks.
   *
   * Everything the assemblers do differently under distribution keys off this one predicate, so
   * that the assembler bodies read identically for both policies and the branching lives here.
   */
  template <typename T> inline constexpr bool is_distributed_la = false;

#ifdef DEAL_II_WITH_PETSC
  template <> inline constexpr bool is_distributed_la<dealii::PETScWrappers::MPI::Vector> = true;
  template <> inline constexpr bool is_distributed_la<dealii::PETScWrappers::MPI::BlockVector> = true;
  template <> inline constexpr bool is_distributed_la<dealii::PETScWrappers::MPI::SparseMatrix> = true;
  template <> inline constexpr bool is_distributed_la<dealii::PETScWrappers::MPI::BlockSparseMatrix> = true;
#endif

  /**
   * @brief Size a vector to the rank's share of the rows.
   *
   * Serial: the plain size-only reinit, exactly as before. Distributed: PETSc needs the owned
   * IndexSet and a communicator -- there is no size-only reinit that would give a correct
   * partitioning, which is why this indirection exists rather than a plain `vec.reinit(n)`.
   */
  template <typename VectorType>
  void reinit_la_vector(VectorType &vec, const dealii::IndexSet &locally_owned, MPI_Comm comm)
  {
    if constexpr (is_distributed_la<VectorType>) {
      vec.reinit(locally_owned, comm);
    } else {
      (void)comm;
      vec.reinit(locally_owned.size());
    }
  }

  /**
   * @brief Turn a freshly built DynamicSparsityPattern into the pattern type the matrix wants.
   *
   * Distributed: the rows a rank builds are not the rows it owns -- a cell worker touching a
   * partition-boundary cell adds entries to rows owned by a neighbour. distribute_sparsity_pattern
   * ships those to their owner. Skipping it does not fail loudly; it produces a matrix that is
   * missing exactly the couplings across partition boundaries, and PETSc then drops the
   * corresponding matrix entries at assembly time.
   *
   * The distributed branch copies rather than moves because DynamicSparsityPattern declares a copy
   * constructor and so has no implicit move. This runs once per reinit(), not per timestep.
   */
  template <typename SparseMatrixType>
  void finalize_la_sparsity(dealii::DynamicSparsityPattern &dsp,
                            get_type::SparsityPattern<SparseMatrixType> &pattern,
                            const dealii::IndexSet &locally_owned, const dealii::IndexSet &locally_relevant,
                            MPI_Comm comm)
  {
    if constexpr (is_distributed_la<SparseMatrixType>) {
      // Guarded, not merely discarded: SparsityTools::distribute_sparsity_pattern only exists in
      // an MPI-enabled deal.II, and the call's arguments are non-dependent, so an `if constexpr`
      // alone would still fail name lookup in a serial build.
#ifdef DEAL_II_WITH_MPI
      dealii::SparsityTools::distribute_sparsity_pattern(dsp, locally_owned, comm, locally_relevant);
      pattern = dsp;
#endif
    } else {
      (void)locally_owned;
      (void)locally_relevant;
      (void)comm;
      pattern.copy_from(dsp);
    }
  }

  /**
   * @brief Size a matrix from a finalized sparsity pattern.
   */
  template <typename SparseMatrixType>
  void reinit_la_matrix(SparseMatrixType &matrix, const get_type::SparsityPattern<SparseMatrixType> &pattern,
                        const dealii::IndexSet &locally_owned, MPI_Comm comm)
  {
    if constexpr (is_distributed_la<SparseMatrixType>) {
      matrix.reinit(locally_owned, pattern, comm);
    } else {
      (void)locally_owned;
      (void)comm;
      matrix.reinit(pattern);
    }
  }

  /**
   * @brief Restrict a global index set to what this rank may write.
   *
   * deal.II builds IDA's differential/algebraic mask by writing every index of the returned set
   * into a distributed vector and then compress(VectorOperation::insert)-ing it
   * (source/sundials/ida.cc). An unrestricted set means every rank writes every index, which is an
   * insert race on the same entries -- benign only as long as the values agree, and not something
   * to rely on. Intersecting here makes each entry written by exactly its owner.
   */
  template <typename VectorType>
  dealii::IndexSet restrict_to_owned(const dealii::IndexSet &global_set, const dealii::IndexSet &locally_owned)
  {
    if constexpr (is_distributed_la<VectorType>) {
      return global_set & locally_owned;
    } else {
      (void)locally_owned;
      return global_set;
    }
  }

  /**
   * @brief The ownership set for the extra-variables block: everything on rank 0, nothing elsewhere.
   *
   * Factored out so reinit_la_block_vector and reinit_la_variables_vector cannot drift apart. Two
   * places building disagreeing layouts for the same block is a hang, not a wrong number: the
   * blocks would have different global sizes and the first collective on them would not match up.
   */
  inline dealii::IndexSet variables_owner_set(const dealii::types::global_dof_index n_vars, MPI_Comm comm)
  {
    constexpr unsigned int variable_owner = 0;
    dealii::IndexSet variables(n_vars);
    if (MPI::rank(comm) == variable_owner) variables.add_range(0, n_vars);
    variables.compress();
    return variables;
  }

  /**
   * @brief Size a standalone vector holding only the extra variables.
   */
  template <typename VectorType>
  void reinit_la_variables_vector(VectorType &vec, const dealii::types::global_dof_index n_vars, MPI_Comm comm)
  {
    if constexpr (is_distributed_la<VectorType>) {
      vec.reinit(variables_owner_set(n_vars, comm), comm);
    } else {
      (void)comm;
      vec.reinit(n_vars);
    }
  }

  /**
   * @brief Size a block vector: block 0 is the FE dofs, block 1 (if present) the extra variables.
   *
   * The two blocks get different ownership policies, and the reason is not symmetric:
   *
   *  * Block 0 is partitioned by the dof distribution, like any FE vector.
   *  * Block 1 holds the model's extra variables. Its right hand side comes from map(), which is
   *    already computed identically on every rank, and essentially every consumer needs all of it.
   *    PETSc block vectors have no replicated block, so it is owned outright by rank 0 and read
   *    elsewhere through a SolutionView. Giving every rank the whole block instead would make each
   *    entry owned n_ranks times, and IDA would then see a state vector n_ranks times too long.
   *
   * @param variable_owner the rank that owns block 1. Fixed at 0 rather than exposed, so the
   *        layout cannot silently differ between two places that both build one of these.
   */
  template <typename BlockVectorType>
  void reinit_la_block_vector(BlockVectorType &vec, const std::vector<uint> &block_structure,
                              const dealii::IndexSet &locally_owned, MPI_Comm comm)
  {
    if constexpr (is_distributed_la<BlockVectorType>) {
      std::vector<dealii::IndexSet> owned{locally_owned};
      if (block_structure.size() > 1) owned.push_back(variables_owner_set(block_structure[1], comm));
      vec.reinit(owned, comm);
    } else {
      (void)locally_owned;
      (void)comm;
      vec.reinit(block_structure);
    }
  }
} // namespace DiFfRG
