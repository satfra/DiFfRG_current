#pragma once

// standard library
#include <memory>

// external libraries
#include <autodiff/forward/real.hpp>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/physics/interpolation/common.hh>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 3D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class LinearInterpolator3D
  {
    static_assert(Coordinates::dim == 3, "LinearInterpolator3D requires 3D coordinates");

  public:
    /**
     * @brief Construct a LinearInterpolator3D object from a vector of data and a coordinate system.
     *
     * @param data vector containing the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const std::vector<NT> &data, const Coordinates &coordinates);
    /**
     * @brief Construct a LinearInterpolator3D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param size size of the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const NT *data, const Coordinates &coordinates);
    /**
     * @brief Construct a LinearInterpolator3D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const Coordinates &coordinates);
    /**
     * @brief Construct a copy of a LinearInterpolator3D object.
     *
     * @param other the object to copy
     */
    LinearInterpolator3D(const LinearInterpolator3D &other);

    template <typename NT2> void update(const NT2 *data)
    {
      if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

      for (uint i = 0; i < size; ++i)
        m_data[i] = NT(data[i]);

      update();
    }
    void update();

    NT *data() const;

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    __device__ __host__ NT operator()(const typename Coordinates::ctype x, const typename Coordinates::ctype y,
                                      const typename Coordinates::ctype z) const;
    NT &operator[](const uint i);
    const NT &operator[](const uint i) const;

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const uint size;
    const Coordinates coordinates;
    const std::array<uint, 3> shape;

    std::shared_ptr<NT[]> m_data;
    std::shared_ptr<thrust::device_vector<NT>> device_data;
    const NT *device_data_ptr;

    const bool owner;
  };
} // namespace DiFfRG
