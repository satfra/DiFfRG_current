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
   * @brief A linear interpolator for 3D data, using texture memory on the GPU and floating point arithmetic on the CPU.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class TexLinearInterpolator3D
  {
    static_assert(Coordinates::dim == 3, "TexLinearInterpolator3D requires 3D coordinates");

  public:
    /**
     * @brief Construct a TexLinearInterpolator3D object from a vector of data and a coordinate system.
     *
     * @param data vector containing the data
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator3D(const std::vector<NT> &data, const Coordinates &coordinates);
    /**
     * @brief Construct a TexLinearInterpolator3D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator3D(const NT *data, const Coordinates &coordinates);
    /**
     * @brief Construct a TexLinearInterpolator3D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator3D(const Coordinates &coordinates);
    /**
     * @brief Construct a copy of a TexLinearInterpolator3D object.
     *
     * @param other the object to copy
     */
    TexLinearInterpolator3D(const TexLinearInterpolator3D &other);

    ~TexLinearInterpolator3D();

    template <typename NT2> void update(const NT2 *data)
    {
      if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

      if constexpr (std::is_same_v<NT2, autodiff::real>)
        for (uint i = 0; i < size; ++i) {
          m_data[i] = static_cast<float>(val(data[i]));
          m_data_AD[i] = static_cast<float>(derivative(data[i]));
        }
      else
        for (uint i = 0; i < size; ++i)
          m_data[i] = static_cast<float>(data[i]);

      update();
    }
    void update();

    float *data() const;
    float *data_AD() const;

    using ReturnType = typename internal::__TLITypes<NT>::ReturnType;

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x x-coordinate of the point
     * @param y y-coordinate of the point
     * @param z z-coordinate of the point
     * @return ReturnType the interpolated value
     */
    __device__ __host__ ReturnType operator()(const float x, const float y, const float z) const;
    ReturnType &operator[](const uint i);
    const ReturnType &operator[](const uint i) const;

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const uint size;
    const std::array<uint, 3> shape;
    const Coordinates coordinates;

    std::shared_ptr<float[]> m_data;
    cudaArray_t device_array;
    cudaTextureObject_t texture;

    std::shared_ptr<float[]> m_data_AD;
    cudaArray_t device_array_AD;
    cudaTextureObject_t texture_AD;

    const bool owner;
  };
} // namespace DiFfRG