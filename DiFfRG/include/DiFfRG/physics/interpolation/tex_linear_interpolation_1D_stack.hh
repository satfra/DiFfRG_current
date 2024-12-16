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
   * @brief A linear interpolator for 1D data, using texture memory on the GPU and floating point arithmetic on the CPU.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates, uint max_stack_size = 64> class TexLinearInterpolator1DStack
  {
    static_assert(Coordinates::dim == 2, "TexLinearInterpolator1DStack requires 2D coordinates");

  public:
    /**
     * @brief Construct a TexLinearInterpolator1D object from a vector of data and a coordinate system.
     *
     * @param data vector containing the data
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator1DStack(const std::vector<NT> &data, const Coordinates &coordinates);
    /**
     * @brief Construct a TexLinearInterpolator1D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator1DStack(const NT *data, const Coordinates &coordinates);
    /**
     * @brief Construct a TexLinearInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator1DStack(const Coordinates &coordinates);

    template <typename Coordinates2>
    TexLinearInterpolator1DStack(const Coordinates2 &coordinates, const uint m_start, const uint m_end, const double T)
        : TexLinearInterpolator1DStack(Coordinates(coordinates, m_start, m_end, T))
    {
    }
    /**
     * @brief Construct a copy of a TexLinearInterpolator1D object.
     *
     * @param other the object to copy
     */
    TexLinearInterpolator1DStack(const TexLinearInterpolator1DStack &other);

    ~TexLinearInterpolator1DStack();

    template <typename NT2> void update(const NT2 *data)
    {
      if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

      if constexpr (std::is_same_v<NT2, autodiff::real>)
        for (uint s = 0; s < stack_size; ++s)
          for (uint i = 0; i < p_size; ++i) {
            m_data[s][i] = static_cast<float>(val(data[s * p_size + i]));
            m_data_AD[s][i] = static_cast<float>(derivative(data[s * p_size + i]));
          }
      else
        for (uint s = 0; s < stack_size; ++s)
          for (uint i = 0; i < p_size; ++i)
            m_data[s][i] = static_cast<float>(data[s * p_size + i]);

      update();
    }
    void update();

    float *data(const uint i) const;
    float *data_AD(const uint i) const;

    using ReturnType = typename internal::__TLITypes<NT>::ReturnType;

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return ReturnType the interpolated value
     */
    __device__ __host__ ReturnType operator()(const float m, const float p) const;
    ReturnType &operator[](const uint i);
    const ReturnType &operator[](const uint i) const;

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const uint stack_size, p_size;
    const Coordinates coordinates;

    std::vector<std::shared_ptr<float[]>> m_data;
    std::vector<cudaArray_t> device_array;
    cudaTextureObject_t texture[max_stack_size];

    std::vector<std::shared_ptr<float[]>> m_data_AD;
    std::vector<cudaArray_t> device_array_AD;
    cudaTextureObject_t texture_AD[max_stack_size];

    const bool owner;
  };
} // namespace DiFfRG