// standard library
#include <string>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>

namespace DiFfRG
{
  void check_cuda(std::string pos)
  {
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      throw std::runtime_error(pos + " : CUDA failed with error:\n    " + std::string(cudaGetErrorString(status)));
  }
} // namespace DiFfRG