#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace DiFfRG
{
  class ExecutionSpaces
  {
  public:
    using GPU_exec_space = Kokkos::DefaultExecutionSpace;
    using GPU_memory_space = GPU_exec_space::memory_space;

    using CPU_exec_space = Kokkos::DefaultHostExecutionSpace;
    using CPU_memory_space = CPU_exec_space::memory_space;
  };

  using GPU_memory = ExecutionSpaces::GPU_memory_space;
  using CPU_memory = ExecutionSpaces::CPU_memory_space;
  using GPU_exec = ExecutionSpaces::GPU_exec_space;
  using CPU_exec = ExecutionSpaces::CPU_exec_space;

  /**
   * @brief An extension of the Kokkos::Sum reducer that adds a constant value to the result.
   *
   * @tparam Scalar the type of the scalar value to be summed.
   * @tparam Space execution space for the Kokkos::Sum reducer.
   */
  template <class Scalar, class SavedScalar, class Space> struct SumPlus {
  public:
    // Required
    using reducer = SumPlus<Scalar, SavedScalar, Space>;
    using value_type = std::remove_cv_t<Scalar>;
    using saved_type = std::remove_cv_t<SavedScalar>;
    static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

    using result_view_type = Kokkos::View<value_type, Space>;

  private:
    result_view_type value;
    bool references_scalar_v;

    const saved_type plus_value;

  public:
    KOKKOS_INLINE_FUNCTION
    SumPlus(value_type &value_, const saved_type &plus_value_)
        : value(&value_), references_scalar_v(true), plus_value(plus_value_)
    {
    }

    KOKKOS_INLINE_FUNCTION
    SumPlus(const result_view_type &value_, const saved_type &plus_value_)
        : value(value_), references_scalar_v(false), plus_value(plus_value_)
    {
    }

    // Required
    KOKKOS_INLINE_FUNCTION
    void join(value_type &dest, const value_type &src) const { dest += src; }

    KOKKOS_INLINE_FUNCTION
    void init(value_type &val) const { val = Kokkos::reduction_identity<value_type>::sum(); }

    KOKKOS_INLINE_FUNCTION
    value_type &reference() const { return *value.data(); }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const { return value; }

    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const { return references_scalar_v; }

    KOKKOS_INLINE_FUNCTION
    void final(value_type &update) const { update += plus_value; }
  };

} // namespace DiFfRG

#include <autodiff/forward/real.hpp>

namespace Kokkos
{ // reduction identity must be defined in Kokkos namespace
  template <size_t N, class T> struct reduction_identity<autodiff::Real<N, T>> {
    KOKKOS_FORCEINLINE_FUNCTION static autodiff::Real<N, T> sum() { return autodiff::Real<N, T>(); }
  };
} // namespace Kokkos