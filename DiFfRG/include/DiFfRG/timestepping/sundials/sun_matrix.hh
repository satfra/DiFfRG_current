#pragma once

#include <DiFfRG/timestepping/sundials/common.hh>

#include <sundials/sundials_matrix.h>

#include <new>

namespace DiFfRG::sundials
{
  namespace matrix_ops
  {
    inline SUNMatrix_ID get_id(SUNMatrix) { return SUNMATRIX_CUSTOM; }

    inline SUNMatrix clone(SUNMatrix A);

    inline void destroy(SUNMatrix A)
    {
      SUNMatFreeEmpty(A);
    }

    inline SUNErrCode success(SUNMatrix) { return SUN_SUCCESS; }
    inline SUNErrCode copy(SUNMatrix, SUNMatrix) { return SUN_SUCCESS; }
    inline SUNErrCode scale_add(sunrealtype, SUNMatrix, SUNMatrix) { return SUN_SUCCESS; }
    inline SUNErrCode scale_add_i(sunrealtype, SUNMatrix) { return SUN_SUCCESS; }
    inline SUNErrCode matvec(SUNMatrix, N_Vector, N_Vector) { return SUN_ERR_OP_FAIL; }
    inline SUNErrCode hermitian_transpose_vec(SUNMatrix, N_Vector, N_Vector) { return SUN_ERR_OP_FAIL; }
    inline SUNErrCode space(SUNMatrix, long int *lrw, long int *liw)
    {
      if (lrw != nullptr) *lrw = 0;
      if (liw != nullptr) *liw = 0;
      return SUN_SUCCESS;
    }
  } // namespace matrix_ops

  inline SUNMatrix create_matrix(SUNContext context)
  {
    SUNMatrix matrix = SUNMatNewEmpty(context);
    if (matrix == nullptr) throw std::bad_alloc();
    matrix->ops->getid = matrix_ops::get_id;
    matrix->ops->clone = matrix_ops::clone;
    matrix->ops->destroy = matrix_ops::destroy;
    matrix->ops->zero = matrix_ops::success;
    matrix->ops->copy = matrix_ops::copy;
    matrix->ops->scaleadd = matrix_ops::scale_add;
    matrix->ops->scaleaddi = matrix_ops::scale_add_i;
    matrix->ops->matvecsetup = matrix_ops::success;
    matrix->ops->matvec = matrix_ops::matvec;
    matrix->ops->mathermitiantransposevec = matrix_ops::hermitian_transpose_vec;
    matrix->ops->space = matrix_ops::space;
    return matrix;
  }

  namespace matrix_ops
  {
    inline SUNMatrix clone(SUNMatrix A) { return create_matrix(A->sunctx); }
  }

  class MatrixHandle
  {
  public:
    MatrixHandle() = default;
    explicit MatrixHandle(SUNMatrix matrix_) : matrix(matrix_) {}
    MatrixHandle(const MatrixHandle &) = delete;
    MatrixHandle &operator=(const MatrixHandle &) = delete;

    MatrixHandle(MatrixHandle &&other) noexcept : matrix(other.matrix) { other.matrix = nullptr; }

    MatrixHandle &operator=(MatrixHandle &&other) noexcept
    {
      if (this != &other) {
        reset();
        matrix = other.matrix;
        other.matrix = nullptr;
      }
      return *this;
    }

    ~MatrixHandle() { reset(); }

    SUNMatrix get() const { return matrix; }

    void reset(SUNMatrix replacement = nullptr)
    {
      if (matrix != nullptr) SUNMatDestroy(matrix);
      matrix = replacement;
    }

  private:
    SUNMatrix matrix = nullptr;
  };
} // namespace DiFfRG::sundials
