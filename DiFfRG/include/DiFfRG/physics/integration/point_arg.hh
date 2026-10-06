#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/types.hh>

// external libraries
#include <autodiff/forward/real.hpp>

// standard library
#include <cstddef>
#include <type_traits>

namespace DiFfRG
{
  /**
   * @brief Host-side array of per-point values, one per evaluation point of a map_points() call.
   */
  template <typename T> struct PointArray {
    const T *data;
    size_t size;
  };

  /**
   * @brief One kernel argument of QuadratureIntegrator::map_points(): either a single value shared by
   * all points, or a PointArray holding one value per point.
   *
   * Implicitly constructible from both, so a map_points() signature can take every argument as a
   * `const PointArg<T> &` and the caller decides per call which arguments vary.
   */
  template <typename T> struct PointArg {
    PointArg(const T &value) : value(value) {}
    PointArg(const PointArray<T> &array) : values(array.data), size(array.size) {}

    bool per_point() const { return values != nullptr; }
    const T &operator[](const size_t i) const { return values != nullptr ? values[i] : value; }

    const T *values = nullptr;
    size_t size = 0;
    T value{};
  };

  namespace internal
  {
    template <typename T> struct _single_precision {
      using value = T;
    };
    template <> struct _single_precision<double> {
      using value = float;
    };
    template <> struct _single_precision<complex<double>> {
      using value = complex<float>;
    };
    template <size_t N> struct _single_precision<autodiff::Real<N, double>> {
      using value = autodiff::Real<N, float>;
    };

    /**
     * @brief The type a map_points() argument of type T is evaluated in by an integrator computing
     * in ctype: a single-precision integrator receives single-precision arguments, so its kernel does
     * not silently promote back to double.
     */
    template <typename T, typename ctype>
    using compute_arg_t = std::conditional_t<std::is_same_v<ctype, float>, typename _single_precision<T>::value, T>;

    /**
     * @brief Device-side view of a PointArg: a pointer into staged per-point values, or the broadcast
     * value. Broadcast values are returned by reference, so an interpolator passed as a shared
     * argument is not copied per thread.
     */
    template <typename T> struct DevicePointArg {
      const T *values;
      T value;

      KOKKOS_FORCEINLINE_FUNCTION const T &operator()(const size_t i) const
      {
        return values != nullptr ? values[i] : value;
      }
    };
  } // namespace internal
} // namespace DiFfRG
