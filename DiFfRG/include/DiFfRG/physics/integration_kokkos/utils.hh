#pragma once

#include <Kokkos_Core.hpp>

namespace DiFfRG
{
  class ExecutionSpaces
  {
  public:
    static constexpr auto GPU_exec_space = Kokkos::DefaultExecutionSpace();
    static constexpr auto GPU_memory_space = GPU_exec_space::memory_space();

    static constexpr auto CPU_exec_space = Kokkos::DefaultHostExecutionSpace();
    static constexpr auto CPU_memory_space = CPU_exec_space::memory_space();
  }
} // namespace DiFfRG