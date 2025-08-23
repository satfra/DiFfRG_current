#pragma once

#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/types.hh>

namespace DiFfRG
{
  /**
   * @brief A concept for what is an interpolator class
   *
   * An interpolator class must provide the following methods:
   * - get_coordinates(): returns the coordinates object used for the interpolation
   * - get_on<MemorySpace>(): returns the interpolator object on the specified memory space (CPU_memory or GPU_memory)
   *
   * The interpolator class must also define the following type aliases:
   * - value_type: the type of the values being interpolated
   *
   * @tparam T The type to check
   */
  template <typename T>
  concept is_interpolator = requires(T t) {
    typename T::value_type;
    typename T::ctype;
    t.get_coordinates();
    t.template get_on<CPU_memory>();
    t.template get_on<GPU_memory>();
    requires has_n_call_operator<T, typename T::ctype, T::dim>;
  };
} // namespace DiFfRG

#include <DiFfRG/discretization/coordinates/combined_coordinates.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/discretization/coordinates/stack_coordinates.hh>

#include <DiFfRG/physics/interpolation/linear_interpolator.hh>

#include <DiFfRG/physics/interpolation/spline_interpolator_1D.hh>
#include <DiFfRG/physics/interpolation/spline_interpolator_1D_stack.hh>