#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace DiFfRG
{
  class ExecutionSpaces
  {
  public:
    using GPU_exec_space = Kokkos::DefaultExecutionSpace;
    using GPU_memory_space = GPU_exec_space::memory_space;

    using CPU_exec_space = Kokkos::DefaultHostExecutionSpace;
    using CPU_memory_space = CPU_exec_space::memory_space;
  };

  using GPU_memory = ExecutionSpaces::GPU_memory_space;
  using CPU_memory = ExecutionSpaces::CPU_memory_space;
  using GPU_exec = ExecutionSpaces::GPU_exec_space;
  using CPU_exec = ExecutionSpaces::CPU_exec_space;
} // namespace DiFfRG
