#pragma once

#include <DiFfRG/common/tuples.hh>
#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#else
#include <array>
#include <tuple>
#include <utility>
#endif

namespace DiFfRG
{
  /**
   * @brief This execution space is optimal when used in conjunction with the FE discretizations.
   *
   */
  struct TBB_ExecutionSpace {
    using memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    void fence() const {}
  };

  class ExecutionSpaces
  {
  public:
    using GPU_exec_space = Kokkos::DefaultExecutionSpace;
    using GPU_memory_space = GPU_exec_space::memory_space;

    using Threads_exec_space = Kokkos::DefaultHostExecutionSpace;
    using Threads_memory_space = Threads_exec_space::memory_space;

    using TBB_exec_space = TBB_ExecutionSpace;
    using TBB_memory_space = TBB_exec_space::memory_space;
  };

  using GPU_memory = ExecutionSpaces::GPU_memory_space;
  using Threads_memory = ExecutionSpaces::Threads_memory_space;
  using TBB_memory = ExecutionSpaces::TBB_memory_space;

  using CPU_memory = Kokkos::DefaultHostExecutionSpace::memory_space;

  using GPU_exec = ExecutionSpaces::GPU_exec_space;
  using Threads_exec = ExecutionSpaces::Threads_exec_space;
  using TBB_exec = ExecutionSpaces::TBB_exec_space;

  // Ensure that CPU memory space is the same as Threads memory space and TBB memory space
  // We assume that this is true, and when switching to a different memory space, it is always unique.
  static_assert(std::is_same_v<CPU_memory, Threads_memory>,
                "CPU memory space must be the same as Threads memory space");
  static_assert(std::is_same_v<CPU_memory, TBB_memory>, "CPU memory space must be the same as TBB memory space");

  template <typename MemorySpace, typename Enable = void> class GetOtherMemorySpaceHelper;
  template <> struct GetOtherMemorySpaceHelper<GPU_memory> {
    using type = CPU_memory;
  };
  // template <>
  template <typename T>
  struct GetOtherMemorySpaceHelper<
      T, std::enable_if_t<std::is_same_v<T, GPU_memory> && !std::is_same_v<CPU_memory, GPU_memory>>> {
    using type = GPU_memory;
  };
  template <> struct GetOtherMemorySpaceHelper<GPU_exec> {
    using type = CPU_memory;
  };
  template <typename T>
  struct GetOtherMemorySpaceHelper<
      T, std::enable_if_t<std::is_same_v<T, Threads_exec> && !std::is_same_v<Threads_exec, GPU_exec>>> {
    using type = GPU_memory;
  };
  template <> struct GetOtherMemorySpaceHelper<TBB_exec> {
    using type = GPU_memory;
  };
  template <typename MemorySpace> using other_memory_space_t = typename GetOtherMemorySpaceHelper<MemorySpace>::type;

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

  namespace device
  {
#ifdef KOKKOS_ENABLE_CUDA
    template <typename... T> using tuple = cuda::std::tuple<T...>;
    template <typename T, std::size_t N> using array = cuda::std::array<T, N>;
    using cuda::std::apply;
    using cuda::std::forward;
    using cuda::std::forward_as_tuple;
    using cuda::std::get;
    using cuda::std::index_sequence;
    using cuda::std::integer_sequence;
    using cuda::std::make_integer_sequence;
    using cuda::std::make_tuple;
    using cuda::std::tie;
    using cuda::std::tuple_cat;
    using cuda::std::tuple_element;
#else
    template <typename... T> using tuple = std::tuple<T...>;
    template <typename T, std::size_t N> using array = std::array<T, N>;
    using std::apply;
    using std::forward;
    using std::forward_as_tuple;
    using std::get;
    using std::index_sequence;
    using std::integer_sequence;
    using std::make_integer_sequence;
    using std::make_tuple;
    using std::tie;
    using std::tuple_cat;
    using std::tuple_element;
#endif
  } // namespace device

  template <int dim, typename T> struct GetKokkosNDStarType {
    using type = typename GetKokkosNDStarType<dim - 1, T>::type *;
  };
  template <typename T> struct GetKokkosNDStarType<1, T> {
    using type = T *;
  };

  // ------------------------------------------------
  // Getting View types
  // ------------------------------------------------

  template <int dim, typename T, typename ExecutionSpace>
  using KokkosNDView = Kokkos::View<typename GetKokkosNDStarType<dim, T>::type, // Get the star syntax for
                                                                                // dimensionality recursively with
                                    ExecutionSpace                              // Choice between GPU and CPU
                                    >;
  template <int dim, typename T, typename ExecutionSpace>
  using KokkosNDViewUnmanaged =
      Kokkos::View<typename GetKokkosNDStarType<dim, T>::type, // Get the star syntax for dimensionality recursively
                                                               // with a helper
                   ExecutionSpace,                             // Choice between GPU and CPU
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>     // No allocation: Attach to existing memory
                   >;

  // ------------------------------------------------
  // Getting ranges to iterate over
  // ------------------------------------------------
  template <int dim, typename ExecutionSpace> struct KokkosNDRangeHelper {
    using type = Kokkos::MDRangePolicy<Kokkos::Rank<dim>, ExecutionSpace>;
  };
  template <typename ExecutionSpace> struct KokkosNDRangeHelper<1, ExecutionSpace> {
    using type = Kokkos::RangePolicy<ExecutionSpace>;
  };
  template <int dim, typename ExecutionSpace> using KokkosNDRange = KokkosNDRangeHelper<dim, ExecutionSpace>::type;

  template <int dim, typename ExecutionSpace>
  auto makeKokkosNDRange(ExecutionSpace &space, const device::array<size_t, dim> start,
                         const device::array<size_t, dim> end)
  {
    if constexpr (dim == 1) {
      return KokkosNDRange<dim, ExecutionSpace>(space, start[0], end[0]);
    } else {
      Kokkos::Array<size_t, dim> start_view;
      Kokkos::Array<size_t, dim> end_view;
      for (size_t i = 0; i < dim; ++i) {
        start_view[i] = start[i];
        end_view[i] = end[i];
      }
      return KokkosNDRange<dim, ExecutionSpace>(space, start_view, end_view);
    }
  }

  // ------------------------------------------------
  // Wrap Kokkos lambdas
  // ------------------------------------------------

  /**
   * @brief This is a functor which wraps a lambda.
   * Basically, this is necessary when one wants to call a variadic lambda on an NVIDIA GPU.
   * CUDA seems to be unable to expand the variadic arguments - in contrast, a direct approach does indeed work for
   * openMP or serial compilation.
   * To get around this limitation, the KokkosNDLambdaWrapper packs the indices into an array.
   * If you wonder, whether there's a difference when using tie and tuples: https://godbolt.org/z/M3bG39rsM
   * No. Therefore, we spare the ourselves the hassle and simply use an array.
   *
   * @tparam dim Number of arguments taken
   * @tparam FUN The lambda to which we forward the indices
   */
  template <int dim, typename FUN> struct KokkosNDLambdaWrapper {
    KokkosNDLambdaWrapper(const FUN &_fun) : fun(_fun) {};

    template <typename... Args>
      requires(sizeof...(Args) == dim)
    KOKKOS_FORCEINLINE_FUNCTION void operator()(Args &&...args) const
    {
      fun({{std::forward<Args>(args)...}});
    }

    FUN fun;
  };

  template <int i, typename Head, typename... Tail> constexpr auto tuple_first(const device::tuple<Head, Tail...> &t)
  {
    if constexpr (i == 0)
      return device::tuple();
    else if constexpr (i == 1)
      return device::apply([](auto &head, auto &.../*tail*/) { return device::tie(head); }, t);
    else
      return device::apply(
          [](auto &head, auto &...tail) {
            return device::tuple_cat(device::tie(head), tuple_first<i - 1>(device::tie(tail...)));
          },
          t);
  }

  /**
   * @brief This is a functor which wraps a lambda for reduction.
   * Basically, this is necessary when one wants to call a variadic lambda on an NVIDIA GPU.
   * CUDA seems to be unable to expand the variadic arguments - in contrast, a direct approach does indeed work for
   * openMP or serial compilation.
   * To get around this limitation, the KokkosNDLambdaWrapperReduction packs the indices into an array.
   * If you wonder, whether there's a difference when using tie and tuples: https://godbolt.org/z/M3bG39rsM
   * No. Therefore, we spare the ourselves the hassle and simply use an array.
   *
   * @tparam dim Number of arguments taken
   * @tparam FUN The lambda to which we forward the indices
   */
  template <int dim, typename FUN> struct KokkosNDLambdaWrapperReduction {
    KokkosNDLambdaWrapperReduction(const FUN &_fun) : fun(_fun) {};

    template <typename... Args>
      requires(sizeof...(Args) == dim + 1)
    KOKKOS_FORCEINLINE_FUNCTION void operator()(Args &&...args) const
    {
      auto tuple = device::tie(args...);
      fun(makeArray(tuple_first<dim>(tuple)), device::get<dim>(tuple)); // the last argument is the reduction result
    }

    FUN fun;

    template <typename... Args>
      requires(sizeof...(Args) == dim)
    KOKKOS_FORCEINLINE_FUNCTION auto makeArray(device::tuple<Args...> &&tuple) const
    {
      return device::apply([](auto &&...args) { return device::array<size_t, dim>{{args...}}; }, tuple);
    }
  };

} // namespace DiFfRG

#include <autodiff/forward/real.hpp>

namespace Kokkos
{ // reduction identity must be defined in Kokkos namespace
  template <size_t N, class T> struct reduction_identity<autodiff::Real<N, T>> {
    KOKKOS_FORCEINLINE_FUNCTION static autodiff::Real<N, T> sum() { return autodiff::Real<N, T>(); }
  };
} // namespace Kokkos