#pragma once

// DiFfRG
#include <DiFfRG/physics/interpolation/linear_interpolator_1D.hh>
#include <DiFfRG/physics/interpolation/linear_interpolator_2D.hh>
#include <DiFfRG/physics/interpolation/linear_interpolator_3D.hh>

namespace DiFfRG
{
  namespace internal
  {
    template <typename T, int dim> struct _has_dim {
      static constexpr bool value = (T::dim == dim);
    };
    template <typename T, int dim> constexpr bool has_dim = _has_dim<T, dim>::value;
  } // namespace internal

  template <size_t dim, typename NT, typename Coordinates, typename DefaultMemorySpace = CPU_memory>
  struct LinearInterpolatorND_helper;

  template <typename NT, typename Coordinates, typename DefaultMemorySpace>
  struct LinearInterpolatorND_helper<1, NT, Coordinates, DefaultMemorySpace> {
    using type = LinearInterpolator1D<NT, Coordinates, DefaultMemorySpace>;
  };

  template <typename NT, typename Coordinates, typename DefaultMemorySpace>
  struct LinearInterpolatorND_helper<2, NT, Coordinates, DefaultMemorySpace> {
    using type = LinearInterpolator2D<NT, Coordinates, DefaultMemorySpace>;
  };

  template <typename NT, typename Coordinates, typename DefaultMemorySpace>
  struct LinearInterpolatorND_helper<3, NT, Coordinates, DefaultMemorySpace> {
    using type = LinearInterpolator3D<NT, Coordinates, DefaultMemorySpace>;
  };

  /**
   * @brief A linear interpolator for ND data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates, typename DefaultMemorySpace = CPU_memory>
  using LinearInterpolatorND =
      typename LinearInterpolatorND_helper<Coordinates::dim, NT, Coordinates, DefaultMemorySpace>::type;
} // namespace DiFfRG