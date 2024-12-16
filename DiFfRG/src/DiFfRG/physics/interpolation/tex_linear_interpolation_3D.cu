// DiFfRG
#include <DiFfRG/discretization/grid/coordinates.hh>
#include <DiFfRG/physics/interpolation/tex_linear_interpolation_3D.hh>

namespace DiFfRG
{
  template <typename NT, typename Coordinates>
  TexLinearInterpolator3D<NT, Coordinates>::TexLinearInterpolator3D(const std::vector<NT> &data,
                                                                    const Coordinates &coordinates)
      : TexLinearInterpolator3D(data.data(), coordinates)
  {
    if (data.size() != coordinates.size()) throw std::runtime_error("Data and coordinates must have the same size");
  }

  template <typename NT, typename Coordinates>
  TexLinearInterpolator3D<NT, Coordinates>::TexLinearInterpolator3D(const Coordinates &coordinates)
      : TexLinearInterpolator3D(std::vector<NT>(coordinates.size(), NT(0)), coordinates)
  {
  }

  template <typename NT, typename Coordinates>
  TexLinearInterpolator3D<NT, Coordinates>::TexLinearInterpolator3D(const NT *in_data, const Coordinates &coordinates)
      : size(coordinates.size()), shape(coordinates.sizes()), coordinates(coordinates), m_data(nullptr),
        device_array(nullptr), m_data_AD(nullptr), device_array_AD(nullptr), owner(true)
  {
    if constexpr (std::is_same_v<float, ReturnType>) {
      // Copy input data and possibly cast to float
      m_data = std::shared_ptr<float[]>(new float[size]);
      for (uint i = 0; i < size; ++i)
        m_data[i] = static_cast<float>(in_data[i]);
#ifdef __CUDACC__
      check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D, precheck");
      // Allocate device array and copy data
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
      cudaMalloc3DArray(&device_array, &channelDesc, make_cudaExtent(shape[2], shape[1], shape[0]));
      cudaDeviceSynchronize();
      check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D, cudaMalloc3DArray");

      // Array creation
      cudaMemcpy3DParms copyParams = {0};
      copyParams.srcPtr = make_cudaPitchedPtr(m_data.get(), shape[2] * sizeof(float), shape[2], shape[1]);
      copyParams.dstArray = device_array;
      copyParams.extent = make_cudaExtent(shape[2], shape[1], shape[0]);
      copyParams.kind = cudaMemcpyHostToDevice;
      cudaMemcpy3D(&copyParams);
      cudaDeviceSynchronize();
      check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D, cudaMemcpy3D");

      // Specify texture
      struct cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = device_array;

      // Specify texture object parameters
      struct cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.addressMode[2] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.readMode = cudaReadModeElementType;
      texDesc.normalizedCoords = 0;

      // Create texture object
      cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL);
      cudaDeviceSynchronize();
      check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D, cudaCreateTextureObject");
#endif
    } else if constexpr (std::is_same_v<autodiff::real, ReturnType>) {
      // Copy input data and possibly cast to float
      m_data = std::shared_ptr<float[]>(new float[size]);
      m_data_AD = std::shared_ptr<float[]>(new float[size]);
      for (uint i = 0; i < size; ++i) {
        m_data[i] = static_cast<float>(val(in_data[i]));
        m_data_AD[i] = static_cast<float>(derivative(in_data[i]));
      }

#ifdef __CUDACC__
      // Allocate device array and copy data
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
      cudaMalloc3DArray(&device_array, &channelDesc, make_cudaExtent(shape[2], shape[1], shape[0]));
      cudaMalloc3DArray(&device_array_AD, &channelDesc, make_cudaExtent(shape[2], shape[1], shape[0]));
      check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D");

      // Array creation
      cudaMemcpy3DParms copyParams = {0};
      copyParams.srcPtr = make_cudaPitchedPtr(m_data.get(), shape[2] * sizeof(float), shape[2], shape[1]);
      copyParams.dstArray = device_array;
      copyParams.extent = make_cudaExtent(shape[2], shape[1], shape[0]);
      copyParams.kind = cudaMemcpyHostToDevice;
      cudaMemcpy3D(&copyParams);
      check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D");
      cudaMemcpy3DParms copyParams_AD = {0};
      copyParams_AD.srcPtr = make_cudaPitchedPtr(m_data_AD.get(), shape[2] * sizeof(float), shape[2], shape[1]);
      copyParams_AD.dstArray = device_array_AD;
      copyParams_AD.extent = make_cudaExtent(shape[2], shape[1], shape[0]);
      copyParams_AD.kind = cudaMemcpyHostToDevice;
      cudaMemcpy3D(&copyParams_AD);
      check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D");

      // Specify texture
      struct cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = device_array;

      struct cudaResourceDesc resDesc_AD;
      memset(&resDesc_AD, 0, sizeof(resDesc_AD));
      resDesc_AD.resType = cudaResourceTypeArray;
      resDesc_AD.res.array.array = device_array_AD;

      // Specify texture object parameters
      struct cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.addressMode[2] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.readMode = cudaReadModeElementType;
      texDesc.normalizedCoords = 0;

      // Create texture object
      cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL);
      cudaCreateTextureObject(&texture_AD, &resDesc_AD, &texDesc, NULL);
      cudaDeviceSynchronize();
      check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D");
#endif
    }
  }

  template <typename NT, typename Coordinates>
  TexLinearInterpolator3D<NT, Coordinates>::TexLinearInterpolator3D(
      const TexLinearInterpolator3D<NT, Coordinates> &other)
      : size(other.size), shape(other.shape), coordinates(other.coordinates), owner(false)
  {
    m_data = other.m_data;
    m_data_AD = other.m_data_AD;
#ifdef __CUDACC__
    device_array = other.device_array;
    texture = other.texture;
    device_array_AD = other.device_array_AD;
    texture_AD = other.texture_AD;
    check_cuda("TexLinearInterpolator3D::TexLinearInterpolator3D(const TexLinearInterpolator3D&)");
#endif
  }

  template <typename NT, typename Coordinates> TexLinearInterpolator3D<NT, Coordinates>::~TexLinearInterpolator3D()
  {
#ifdef __CUDACC__
    if (owner && m_data != nullptr) {
      cudaDestroyTextureObject(texture);
      cudaFreeArray(device_array);
      check_cuda("TexLinearInterpolator3D::~TexLinearInterpolator3D");
    }
    if (owner && m_data_AD != nullptr) {
      cudaDestroyTextureObject(texture_AD);
      cudaFreeArray(device_array_AD);
      check_cuda("TexLinearInterpolator3D::~TexLinearInterpolator3D");
    }
#endif
  }

  template <typename NT, typename Coordinates> void TexLinearInterpolator3D<NT, Coordinates>::update()
  {
    if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");
#ifdef __CUDACC__
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(m_data.get(), shape[2] * sizeof(float), shape[2], shape[1]);
    copyParams.dstArray = device_array;
    copyParams.extent = make_cudaExtent(shape[2], shape[1], shape[0]);
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    if constexpr (std::is_same_v<ReturnType, autodiff::real>) {
      cudaMemcpy3DParms copyParams_AD = {0};
      copyParams_AD.srcPtr = make_cudaPitchedPtr(m_data_AD.get(), shape[2] * sizeof(float), shape[2], shape[1]);
      copyParams_AD.dstArray = device_array_AD;
      copyParams_AD.extent = make_cudaExtent(shape[2], shape[1], shape[0]);
      copyParams_AD.kind = cudaMemcpyHostToDevice;
      cudaMemcpy3D(&copyParams_AD);
    }

    cudaDeviceSynchronize();
    check_cuda("TexLinearInterpolator3D::update");
#endif
  }

  template <typename NT, typename Coordinates> float *TexLinearInterpolator3D<NT, Coordinates>::data() const
  {
    return m_data.get();
  }

  template <typename NT, typename Coordinates> float *TexLinearInterpolator3D<NT, Coordinates>::data_AD() const
  {
    return m_data_AD.get();
  }

  template <typename NT, typename Coordinates>
  __forceinline__ __device__ __host__ typename TexLinearInterpolator3D<NT, Coordinates>::ReturnType
  TexLinearInterpolator3D<NT, Coordinates>::operator()(const float x, const float y, const float z) const
  {
    auto [idx_x, idx_y, idx_z] = coordinates.backward(x, y, z);
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<ReturnType, autodiff::real>)
      return std::array<double, 2>{tex3D<float>(texture, idx_z + 0.5f, idx_y + 0.5f, idx_x + 0.5f),
                                   tex3D<float>(texture_AD, idx_z + 0.5f, idx_y + 0.5f, idx_x + 0.5f)};
    else if constexpr (std::is_same_v<ReturnType, float>)
      return tex3D<float>(texture, idx_z + 0.5f, idx_y + 0.5f, idx_x + 0.5f);
#else
    using std::ceil;
    using std::floor;
    using std::max;
    using std::min;

    idx_x = max(static_cast<decltype(idx_x)>(0), min(idx_x, static_cast<decltype(idx_x)>(shape[0] - 1)));
    idx_y = max(static_cast<decltype(idx_y)>(0), min(idx_y, static_cast<decltype(idx_y)>(shape[1] - 1)));
    idx_z = max(static_cast<decltype(idx_z)>(0), min(idx_z, static_cast<decltype(idx_z)>(shape[2] - 1)));

    uint x1 = min(ceil(idx_x + static_cast<decltype(idx_x)>(1e-16)), static_cast<decltype(idx_x)>(shape[0] - 1));
    const auto x0 = x1 - 1;
    uint y1 = min(ceil(idx_y + static_cast<decltype(idx_y)>(1e-16)), static_cast<decltype(idx_y)>(shape[1] - 1));
    const auto y0 = y1 - 1;
    uint z1 = min(ceil(idx_z + static_cast<decltype(idx_z)>(1e-16)), static_cast<decltype(idx_z)>(shape[2] - 1));
    const auto z0 = z1 - 1;

    const auto corner000 = m_data[x0 * shape[1] * shape[2] + y0 * shape[2] + z0];
    const auto corner001 = m_data[x0 * shape[1] * shape[2] + y0 * shape[2] + z1];
    const auto corner010 = m_data[x0 * shape[1] * shape[2] + y1 * shape[2] + z0];
    const auto corner011 = m_data[x0 * shape[1] * shape[2] + y1 * shape[2] + z1];
    const auto corner100 = m_data[x1 * shape[1] * shape[2] + y0 * shape[2] + z0];
    const auto corner101 = m_data[x1 * shape[1] * shape[2] + y0 * shape[2] + z1];
    const auto corner110 = m_data[x1 * shape[1] * shape[2] + y1 * shape[2] + z0];
    const auto corner111 = m_data[x1 * shape[1] * shape[2] + y1 * shape[2] + z1];

    const auto ret = corner000 * (x1 - idx_x) * (y1 - idx_y) * (z1 - idx_z) +
                     corner001 * (x1 - idx_x) * (y1 - idx_y) * (idx_z - z0) +
                     corner010 * (x1 - idx_x) * (idx_y - y0) * (z1 - idx_z) +
                     corner011 * (x1 - idx_x) * (idx_y - y0) * (idx_z - z0) +
                     corner100 * (idx_x - x0) * (y1 - idx_y) * (z1 - idx_z) +
                     corner101 * (idx_x - x0) * (y1 - idx_y) * (idx_z - z0) +
                     corner110 * (idx_x - x0) * (idx_y - y0) * (z1 - idx_z) +
                     corner111 * (idx_x - x0) * (idx_y - y0) * (idx_z - z0);

    if constexpr (std::is_same_v<ReturnType, autodiff::real>) {
      const auto corner000_AD = m_data_AD[x0 * shape[1] * shape[2] + y0 * shape[2] + z0];
      const auto corner001_AD = m_data_AD[x0 * shape[1] * shape[2] + y0 * shape[2] + z1];
      const auto corner010_AD = m_data_AD[x0 * shape[1] * shape[2] + y1 * shape[2] + z0];
      const auto corner011_AD = m_data_AD[x0 * shape[1] * shape[2] + y1 * shape[2] + z1];
      const auto corner100_AD = m_data_AD[x1 * shape[1] * shape[2] + y0 * shape[2] + z0];
      const auto corner101_AD = m_data_AD[x1 * shape[1] * shape[2] + y0 * shape[2] + z1];
      const auto corner110_AD = m_data_AD[x1 * shape[1] * shape[2] + y1 * shape[2] + z0];
      const auto corner111_AD = m_data_AD[x1 * shape[1] * shape[2] + y1 * shape[2] + z1];

      const auto ret_AD = corner000_AD * (x1 - idx_x) * (y1 - idx_y) * (z1 - idx_z) +
                          corner001_AD * (x1 - idx_x) * (y1 - idx_y) * (idx_z - z0) +
                          corner010_AD * (x1 - idx_x) * (idx_y - y0) * (z1 - idx_z) +
                          corner011_AD * (x1 - idx_x) * (idx_y - y0) * (idx_z - z0) +
                          corner100_AD * (idx_x - x0) * (y1 - idx_y) * (z1 - idx_z) +
                          corner101_AD * (idx_x - x0) * (y1 - idx_y) * (idx_z - z0) +
                          corner110_AD * (idx_x - x0) * (idx_y - y0) * (z1 - idx_z) +
                          corner111_AD * (idx_x - x0) * (idx_y - y0) * (idx_z - z0);

      return std::array<double, 2>{{ret, ret_AD}};
    } else if constexpr (std::is_same_v<ReturnType, float>)
      return ret;
#endif
  }

  template <typename NT, typename Coordinates>
  typename TexLinearInterpolator3D<NT, Coordinates>::ReturnType &
  TexLinearInterpolator3D<NT, Coordinates>::operator[](const uint i)
  {
    if constexpr (std::is_same_v<ReturnType, autodiff::real>)
      throw std::runtime_error("Cannot access autodiff::real data directly");
    else if constexpr (std::is_same_v<ReturnType, float>)
      return m_data[i];
  }

  template <typename NT, typename Coordinates>
  const typename TexLinearInterpolator3D<NT, Coordinates>::ReturnType &
  TexLinearInterpolator3D<NT, Coordinates>::operator[](const uint i) const
  {
    if constexpr (std::is_same_v<ReturnType, autodiff::real>)
      throw std::runtime_error("Cannot access autodiff::real data directly");
    else if constexpr (std::is_same_v<ReturnType, float>)
      return m_data[i];
  }

  template class TexLinearInterpolator3D<
      double,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class TexLinearInterpolator3D<
      float, CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class TexLinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class TexLinearInterpolator3D<
      double,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class TexLinearInterpolator3D<
      float,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class TexLinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<float>, LogarithmicCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class TexLinearInterpolator3D<
      double, CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class TexLinearInterpolator3D<
      float, CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;
  template class TexLinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LinearCoordinates1D<float>, LinearCoordinates1D<float>, LinearCoordinates1D<float>>>;

  template class TexLinearInterpolator3D<
      double,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class TexLinearInterpolator3D<
      float,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class TexLinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;

  template class TexLinearInterpolator3D<
      double, CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>,
                               LinearCoordinates1D<double>>>;
  template class TexLinearInterpolator3D<
      float, CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>,
                              LinearCoordinates1D<double>>>;
  template class TexLinearInterpolator3D<
      autodiff::real, CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>,
                                       LinearCoordinates1D<double>>>;

  template class TexLinearInterpolator3D<
      double, CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class TexLinearInterpolator3D<
      float, CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
  template class TexLinearInterpolator3D<
      autodiff::real,
      CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>>;
} // namespace DiFfRG
