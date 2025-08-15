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

  /**
   * @brief A linear interpolator for ND data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates, typename MemorySpace> class LinearInterpolatorND;

  // spacializations for 1D, 2D and 3D
  template <typename NT, typename Coordinates, typename MemorySpace>
    requires internal::has_dim<Coordinates, 1>
  class LinearInterpolatorND<NT, Coordinates, MemorySpace> : public LinearInterpolator1D<NT, Coordinates, MemorySpace>
  {
  public:
    using LinearInterpolator1D<NT, Coordinates, MemorySpace>::LinearInterpolator1D;
    using LinearInterpolator1D<NT, Coordinates, MemorySpace>::operator();
    using LinearInterpolator1D<NT, Coordinates, MemorySpace>::update;
  };

  template <typename NT, typename Coordinates, typename MemorySpace>
    requires internal::has_dim<Coordinates, 2>
  class LinearInterpolatorND<NT, Coordinates, MemorySpace> : public LinearInterpolator2D<NT, Coordinates, MemorySpace>
  {
  public:
    using LinearInterpolator2D<NT, Coordinates, MemorySpace>::LinearInterpolator2D;
    using LinearInterpolator2D<NT, Coordinates, MemorySpace>::operator();
    using LinearInterpolator2D<NT, Coordinates, MemorySpace>::update;
  };

  template <typename NT, typename Coordinates, typename MemorySpace>
    requires internal::has_dim<Coordinates, 3>
  class LinearInterpolatorND<NT, Coordinates, MemorySpace> : public LinearInterpolator3D<NT, Coordinates, MemorySpace>
  {
  public:
    using LinearInterpolator3D<NT, Coordinates, MemorySpace>::LinearInterpolator3D;
    using LinearInterpolator3D<NT, Coordinates, MemorySpace>::operator();
    using LinearInterpolator3D<NT, Coordinates, MemorySpace>::update;
  };
} // namespace DiFfRG