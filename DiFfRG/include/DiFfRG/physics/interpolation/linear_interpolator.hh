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
  template <typename MemorySpace, typename NT, typename Coordinates> class LinearInterpolatorND;

  // spacializations for 1D, 2D and 3D
  template <typename MemorySpace, typename NT, typename Coordinates>
    requires internal::has_dim<Coordinates, 1>
  class LinearInterpolatorND<MemorySpace, NT, Coordinates> : public LinearInterpolator1D<MemorySpace, NT, Coordinates>
  {
  public:
    using LinearInterpolator1D<MemorySpace, NT, Coordinates>::LinearInterpolator1D;
    using LinearInterpolator1D<MemorySpace, NT, Coordinates>::operator();
    using LinearInterpolator1D<MemorySpace, NT, Coordinates>::update;
  };

  template <typename MemorySpace, typename NT, typename Coordinates>
    requires internal::has_dim<Coordinates, 2>
  class LinearInterpolatorND<MemorySpace, NT, Coordinates> : public LinearInterpolator2D<MemorySpace, NT, Coordinates>
  {
  public:
    using LinearInterpolator2D<MemorySpace, NT, Coordinates>::LinearInterpolator2D;
    using LinearInterpolator2D<MemorySpace, NT, Coordinates>::operator();
    using LinearInterpolator2D<MemorySpace, NT, Coordinates>::update;
  };

  template <typename MemorySpace, typename NT, typename Coordinates>
    requires internal::has_dim<Coordinates, 3>
  class LinearInterpolatorND<MemorySpace, NT, Coordinates> : public LinearInterpolator3D<MemorySpace, NT, Coordinates>
  {
  public:
    using LinearInterpolator3D<MemorySpace, NT, Coordinates>::LinearInterpolator3D;
    using LinearInterpolator3D<MemorySpace, NT, Coordinates>::operator();
    using LinearInterpolator3D<MemorySpace, NT, Coordinates>::update;
  };
} // namespace DiFfRG