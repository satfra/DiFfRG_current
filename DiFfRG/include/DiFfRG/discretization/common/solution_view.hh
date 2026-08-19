#pragma once

// external libraries
#include <deal.II/base/index_set.h>
#include <deal.II/lac/vector.h>
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_vector.h>
#endif

// DiFfRG
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/types.hh>

// std
#include <stdexcept>

namespace DiFfRG
{
  /**
   * @brief A read-only, fully-ghosted replica of the solution.
   *
   * A large amount of code under discretization/ reads the solution at *arbitrary global dof
   * indices*: the EoM search (common/eom.hh), the KT reconstruction caches
   * (FV/assembler/reconstruction_cache.hh), the output path (data/fe_output.cc) and the support
   * point lookups. None of that is expressible against a vector that only holds this rank's rows.
   *
   * Rather than rewrite all of it, this class maintains one replica whose ghost set is *every*
   * index, refreshed once per residual/Jacobian evaluation. Those call sites then read the replica
   * and need no logic change at all. At replicated-mesh sizes the replica is exactly today's
   * memory footprint.
   *
   * The serial specialization is a pure passthrough -- it stores a pointer, copies nothing, and
   * compiles to the same code as passing the vector directly -- so a serial build pays nothing for
   * the abstraction.
   *
   * @warning The two policies differ in one respect that matters if you hold a view across a
   * write: the serial view aliases the source vector, so it observes later mutations, while the
   * distributed view is a snapshot taken at refresh(). Always refresh() immediately before reading
   * and never hold a view across an assembly step, and the two behave identically.
   *
   * Narrowing the ghost set from "everything" to the locally relevant dofs is the defining step of
   * the fully-distributed rung. It is deliberately confined to this one class so that step has a
   * single place to happen, and so the call sites that will need real work are exactly the ones
   * that name this type.
   */
  template <typename VectorType> class SolutionView
  {
  public:
    using NumberType = get_type::NumberType<VectorType>;
    using size_type = dealii::types::global_dof_index;

    SolutionView() = default;

    /**
     * @brief Establish the layout. No-op for the serial policy.
     */
    void reinit(const dealii::IndexSet &, const dealii::IndexSet &, MPI_Comm) {}

    /**
     * @brief Make the view reflect @p owned.
     */
    void refresh(const VectorType &owned) { source = &owned; }

    const VectorType &get() const
    {
      if (source == nullptr)
        throw std::runtime_error("SolutionView::get() before refresh(): the view holds no solution.");
      return *source;
    }
    operator const VectorType &() const { return get(); }

    NumberType operator[](const size_type i) const { return get()[i]; }
    size_type size() const { return get().size(); }

  private:
    const VectorType *source = nullptr;
  };

#ifdef DEAL_II_WITH_PETSC
  /**
   * @brief Distributed policy: an owned ghosted PETSc vector covering every index.
   */
  template <> class SolutionView<dealii::PETScWrappers::MPI::Vector>
  {
  public:
    using VectorType = dealii::PETScWrappers::MPI::Vector;
    using NumberType = VectorType::value_type;
    using size_type = dealii::types::global_dof_index;

    SolutionView() = default;

    void reinit(const dealii::IndexSet &locally_owned, const dealii::IndexSet &ghost, MPI_Comm comm)
    {
      replica.reinit(locally_owned, ghost, comm);
      initialized = true;
    }

    /**
     * @brief Communicate @p owned into the replica.
     *
     * This relies on deal.II's documented assignment semantics for PETSc MPI vectors: when the
     * left hand side is already correctly sized *and ghosted*, assigning a non-ghosted vector to it
     * fills both the locally owned and the ghost entries and leaves the left hand side ghosted.
     * (This is the step-40 idiom.) It is only true because reinit() ran first -- assigning into a
     * default-constructed vector would instead copy the right hand side's non-ghosted layout and
     * silently produce a view that throws on every ghost read.
     */
    void refresh(const VectorType &owned)
    {
      if (!initialized)
        throw std::runtime_error("SolutionView::refresh() before reinit(): assigning into an "
                                 "unsized PETSc vector would drop the ghost layout.");
      replica = owned;
    }

    const VectorType &get() const
    {
      if (!initialized)
        throw std::runtime_error("SolutionView::get() before reinit(): the view holds no solution.");
      return replica;
    }
    operator const VectorType &() const { return get(); }

    NumberType operator[](const size_type i) const { return get()(i); }
    size_type size() const { return get().size(); }

  private:
    VectorType replica;
    bool initialized = false;
  };
#endif
} // namespace DiFfRG
