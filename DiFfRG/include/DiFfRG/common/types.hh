#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>

// external libraries
#include <autodiff/forward/real.hpp>
#include <deal.II/base/config.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>

// The distributed linear algebra is PETSc-backed. deal.II ships these headers
// unconditionally but leaves them empty without PETSc, so guard on the feature
// macro (defined by deal.II/base/config.h above) rather than on their presence.
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#endif

#include <type_traits>

namespace DiFfRG
{
  template <typename T>
  concept is_container = requires(T x) { x[0]; };

  template <size_t N, typename T>
  concept is_sized_container = requires(T x) {
    x[0];
    requires x.size() == N;
  };

  // we need a concept to check if a type has an operator() with N arguments of type ValueType
  template <typename T, typename ValueType, size_t N> struct has_n_call_operator_helper {
    template <size_t... Is> static constexpr bool tryit(std::integer_sequence<size_t, Is...>)
    {
      if constexpr (requires(T t, ValueType v) { t.operator()((ValueType{} * Is)...); }) {
        return true;
      } else {
        return false;
      }
    }

    static constexpr bool value = tryit(std::make_index_sequence<N>{});
  };

  template <typename T, typename ValueType, size_t N>
  concept has_n_call_operator = has_n_call_operator_helper<T, ValueType, N>::value;

  namespace get_type
  {
    namespace internal
    {
      //--------------------------------------------------
      // Hidden unspecified type helpers
      //--------------------------------------------------

      template <typename VectorType> struct _NumberType;

      template <typename SparseMatrixType> struct _SparsityPattern;

      template <typename SparseMatrixType> struct _InverseSparseMatrixType;

      template <typename VectorType> struct _BlockVectorType;

      //--------------------------------------------------
      // Specified type helpers for standard vectors
      //--------------------------------------------------

      template <typename NT> struct _NumberType<dealii::Vector<NT>> {
        using value = NT;
      };
      template <typename NT> struct _SparsityPattern<dealii::SparseMatrix<NT>> {
        using value = dealii::SparsityPattern;
      };
      template <typename NT> struct _InverseSparseMatrixType<dealii::SparseMatrix<NT>> {
        using value = dealii::SparseDirectUMFPACK;
      };

      //--------------------------------------------------
      // Specified type helpers for block vectors
      //--------------------------------------------------

      template <typename NT> struct _NumberType<dealii::BlockVector<NT>> {
        using value = NT;
      };
      template <typename NT> struct _SparsityPattern<dealii::BlockSparseMatrix<NT>> {
        using value = dealii::BlockSparsityPattern;
      };
      template <typename NT> struct _InverseSparseMatrixType<dealii::BlockSparseMatrix<NT>> {
        using value = dealii::SparseDirectUMFPACK;
      };

      template <typename NT> struct _BlockVectorType<dealii::Vector<NT>> {
        using value = dealii::BlockVector<NT>;
      };
      template <typename NT> struct _BlockVectorType<dealii::BlockVector<NT>> {
        using value = dealii::BlockVector<NT>;
      };

      //--------------------------------------------------
      // Specified type helpers for distributed (PETSc) vectors
      //--------------------------------------------------
      //
      // PETSc's vectors are not templated on the number type -- PetscScalar is fixed
      // at PETSc configure time -- so these are full specializations and the number
      // type is read off the vector itself.
      //
      // Note what is deliberately NOT specialized here:
      //
      //  * _InverseSparseMatrixType. It exists only to give the *explicit* steppers
      //    (explicit_euler, boost_rk, boost_abm) a mass-matrix inverse, and those are
      //    not instantiated for PETSc -- they are not viable for these stiff flows and
      //    they route through common/eigen.hh, which assumes contiguous serial storage.
      //    Leaving it unspecialized turns an accidental instantiation into a clear
      //    "incomplete type" error instead of silently selecting a serial UMFPACK.
      //
      //  * A preconditioner trait for GMRES.hh's `PreconditionJacobi<SparseMatrixType>`
      //    default, which does not exist for PETSc matrices. None is needed: a default
      //    template argument is only instantiated when used, and the PETSc timestepper
      //    instantiations use PETScKrylov/PETScDirect, never GMRES or ScaledGMRES.
#ifdef DEAL_II_WITH_PETSC
      template <> struct _NumberType<dealii::PETScWrappers::MPI::Vector> {
        using value = dealii::PETScWrappers::MPI::Vector::value_type;
      };
      template <> struct _NumberType<dealii::PETScWrappers::MPI::BlockVector> {
        using value = dealii::PETScWrappers::MPI::BlockVector::value_type;
      };

      // A PETSc matrix owns its sparsity internally and is reinit'ed from a
      // DynamicSparsityPattern plus IndexSets (petsc_sparse_matrix.h:510), so there is
      // no persistent SparsityPattern object to hand back. Assemblers therefore keep
      // the DynamicSparsityPattern they build alive instead of discarding it after
      // copy_from().
      template <> struct _SparsityPattern<dealii::PETScWrappers::MPI::SparseMatrix> {
        using value = dealii::DynamicSparsityPattern;
      };
      template <> struct _SparsityPattern<dealii::PETScWrappers::MPI::BlockSparseMatrix> {
        using value = dealii::BlockDynamicSparsityPattern;
      };

      template <> struct _BlockVectorType<dealii::PETScWrappers::MPI::Vector> {
        using value = dealii::PETScWrappers::MPI::BlockVector;
      };
      template <> struct _BlockVectorType<dealii::PETScWrappers::MPI::BlockVector> {
        using value = dealii::PETScWrappers::MPI::BlockVector;
      };
#endif
    } // namespace internal

    template <typename VectorType> using NumberType = typename internal::_NumberType<VectorType>::value;

    template <typename SparseMatrixType>
    using SparsityPattern = typename internal::_SparsityPattern<SparseMatrixType>::value;

    template <typename SparseMatrixType>
    using InverseSparseMatrixType = typename internal::_InverseSparseMatrixType<SparseMatrixType>::value;

    /**
     * @brief The block-vector type belonging to a given vector type.
     *
     * Timesteppers need this for the FE+variables run path, where the state is one
     * block vector (block 0 = FE dofs, block 1 = variables). It used to be hard-wired
     * to dealii::BlockVector, which silently excluded every distributed vector type.
     */
    template <typename VectorType> using BlockVectorType = typename internal::_BlockVectorType<VectorType>::value;

    namespace internal
    {
      template <typename CT> struct _ctype;

      template <> struct _ctype<float> {
        using value = float;
      };

      template <> struct _ctype<double> {
        using value = double;
      };

      template <> struct _ctype<complex<float>> {
        using value = float;
      };

      template <> struct _ctype<complex<double>> {
        using value = double;
      };

      template <size_t N, typename T> struct _ctype<autodiff::Real<N, T>> : _ctype<T> {
      };

      template <typename T> inline constexpr bool _is_autodiff = false;
      template <size_t N, typename U> inline constexpr bool _is_autodiff<autodiff::Real<N, U>> = true;
    } // namespace internal

    template <typename CT> using ctype = typename internal::_ctype<CT>::value;

    template <typename T> inline constexpr bool is_autodiff = internal::_is_autodiff<T>;
  } // namespace get_type

  /**
   * @brief A vector type DiFfRG's timesteppers and assemblers can work with.
   *
   * Satisfied exactly when the get_type trait table knows the vector's number type,
   * i.e. for dealii::Vector/BlockVector and -- in an MPI build -- the PETSc MPI
   * vectors. This replaces a hard `static_assert(is_same_v<VectorType, Vector<...>>)`
   * that predated any distributed vector support.
   */
  template <typename VectorType>
  concept SupportedVectorType = requires { typename get_type::internal::_NumberType<VectorType>::value; };
} // namespace DiFfRG
