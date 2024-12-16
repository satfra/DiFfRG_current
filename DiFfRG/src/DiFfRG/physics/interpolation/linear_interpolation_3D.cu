// DiFfRG
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/linear_interpolation_3D.hh>

namespace DiFfRG
{
  template <typename NT, typename Coordinates>
  LinearInterpolator3D<NT, Coordinates>::LinearInterpolator3D(const std::vector<NT> &data,
                                                              const Coordinates &coordinates)
      : LinearInterpolator3D(data.data(), coordinates)
  {
    if (data.size() != coordinates.size()) throw std::runtime_error("Data and coordinates must have the same size");
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator3D<NT, Coordinates>::LinearInterpolator3D(const Coordinates &coordinates)
      : LinearInterpolator3D(std::vector<NT>(coordinates.size(), NT(0)), coordinates)
  {
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator3D<NT, Coordinates>::LinearInterpolator3D(const NT *in_data, const Coordinates &coordinates)
      : size(coordinates.size()), shape(coordinates.sizes()), coordinates(coordinates), m_data(nullptr), owner(true)
  {
    // Copy input data
    m_data = std::shared_ptr<NT[]>(new NT[size]);
    for (uint i = 0; i < size; ++i)
      m_data[i] = NT(in_data[i]);
      // Create the thrust::device_vector
#ifdef __CUDACC__
    device_data = std::shared_ptr<thrust::device_vector<NT>>(new thrust::device_vector<NT>());
    device_data->resize(size);
    thrust::copy(m_data.get(), m_data.get() + size, device_data->begin());
    check_cuda("LinearInterpolator3D::LinearInterpolator3D");
    device_data_ptr = thrust::raw_pointer_cast(device_data->data());
#endif
  }

  template <typename NT, typename Coordinates>
  LinearInterpolator3D<NT, Coordinates>::LinearInterpolator3D(const LinearInterpolator3D<NT, Coordinates> &other)
      : size(other.size), coordinates(other.coordinates), owner(false), shape(other.shape)
  {
    m_data = other.m_data;
#ifdef __CUDACC__
    device_data_ptr = other.device_data_ptr;
#endif
  }

  template <typename NT, typename Coordinates> void LinearInterpolator3D<NT, Coordinates>::update()
  {
    if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

#ifdef __CUDACC__
    thrust::copy(m_data.get(), m_data.get() + size, device_data->begin());
    check_cuda("LinearInterpolator3D::update");
#endif
  }

  template <typename NT, typename Coordinates> NT *LinearInterpolator3D<NT, Coordinates>::data() const
  {
    return m_data.get();
  }

  template <typename NT, typename Coordinates>
  __forceinline__ __device__ __host__ NT LinearInterpolator3D<NT, Coordinates>::operator()(
      const typename Coordinates::ctype x, const typename Coordinates::ctype y,
      const typename Coordinates::ctype z) const
  {
#ifndef __CUDA_ARCH__
    using std::ceil;
    using std::floor;
    using std::max;
    using std::min;
#endif

    auto [idx_x, idx_y, idx_z] = coordinates.backward(x, y, z);
    idx_x = max(static_cast<decltype(idx_x)>(0), min(idx_x, static_cast<decltype(idx_x)>(shape[0] - 1)));
    idx_y = max(static_cast<decltype(idx_y)>(0), min(idx_y, static_cast<decltype(idx_y)>(shape[1] - 1)));
    idx_z = max(static_cast<decltype(idx_z)>(0), min(idx_z, static_cast<decltype(idx_z)>(shape[2] - 1)));

#ifndef __CUDA_ARCH__
    const auto *d_ptr = m_data.get();
#else
    const auto *d_ptr = device_data_ptr;
#endif

    uint x1 = min(ceil(idx_x + static_cast<decltype(idx_x)>(1e-16)), static_cast<decltype(idx_x)>(shape[0] - 1));
    const auto x0 = x1 - 1;
    uint y1 = min(ceil(idx_y + static_cast<decltype(idx_y)>(1e-16)), static_cast<decltype(idx_y)>(shape[1] - 1));
    const auto y0 = y1 - 1;
    uint z1 = min(ceil(idx_z + static_cast<decltype(idx_z)>(1e-16)), static_cast<decltype(idx_z)>(shape[2] - 1));
    const auto z0 = z1 - 1;

    const auto corner000 = d_ptr[x0 * shape[1] * shape[2] + y0 * shape[2] + z0];
    const auto corner001 = d_ptr[x0 * shape[1] * shape[2] + y0 * shape[2] + z1];
    const auto corner010 = d_ptr[x0 * shape[1] * shape[2] + y1 * shape[2] + z0];
    const auto corner011 = d_ptr[x0 * shape[1] * shape[2] + y1 * shape[2] + z1];
    const auto corner100 = d_ptr[x1 * shape[1] * shape[2] + y0 * shape[2] + z0];
    const auto corner101 = d_ptr[x1 * shape[1] * shape[2] + y0 * shape[2] + z1];
    const auto corner110 = d_ptr[x1 * shape[1] * shape[2] + y1 * shape[2] + z0];
    const auto corner111 = d_ptr[x1 * shape[1] * shape[2] + y1 * shape[2] + z1];

    return corner000 * (x1 - idx_x) * (y1 - idx_y) * (z1 - idx_z) +
           corner001 * (x1 - idx_x) * (y1 - idx_y) * (idx_z - z0) +
           corner010 * (x1 - idx_x) * (idx_y - y0) * (z1 - idx_z) +
           corner011 * (x1 - idx_x) * (idx_y - y0) * (idx_z - z0) +
           corner100 * (idx_x - x0) * (y1 - idx_y) * (z1 - idx_z) +
           corner101 * (idx_x - x0) * (y1 - idx_y) * (idx_z - z0) +
           corner110 * (idx_x - x0) * (idx_y - y0) * (z1 - idx_z) +
           corner111 * (idx_x - x0) * (idx_y - y0) * (idx_z - z0);
  }

  template <typename NT, typename Coordinates> NT &LinearInterpolator3D<NT, Coordinates>::operator[](const uint i)
  {
    return m_data[i];
  }
  template <typename NT, typename Coordinates>
  const NT &LinearInterpolator3D<NT, Coordinates>::operator[](const uint i) const
  {
    return m_data[i];
  }

  template class LinearInterpolator3D<double, CoordinatePackND<LogarithmicCoordinates1D<float>,
                                                               LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      float, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class LinearInterpolator3D<
      double,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      float,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class LinearInterpolator3D<
      double, CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      float, CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class LinearInterpolator3D<
      double,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<
      float,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;

  template class LinearInterpolator3D<double,
                                      CoordinatePackND<LogarithmicCoordinates1D<double>,
                                                       LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<float,
                                      CoordinatePackND<LogarithmicCoordinates1D<double>,
                                                       LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<autodiff::real,
                                      CoordinatePackND<LogarithmicCoordinates1D<double>,
                                                       LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>>;

  template class LinearInterpolator3D<
      double, CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<
      float, CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class LinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;

} // namespace DiFfRG
