#include <DiFfRG/common/init.hh>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

// standard libraries
#include <cstdlib>
#include <oneapi/tbb/task_arena.h>

// external libraries
#include <tbb/tbb.h>

namespace DiFfRG
{
  bool Init::initialized = false;

  dealii::Utilities::MPI::MPI_InitFinalize *Init::mpi_initialization = nullptr;

  Init::Init(int argc, char *argv[], const std::string parameter_file)
      : argc(argc), argv(argv), parameter_file(parameter_file)
  {
    if (!initialized) {
      mpi_initialization = new dealii::Utilities::MPI::MPI_InitFinalize(
          argc, argv, tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));
      Kokkos::initialize(argc, argv);
      initialized = true;

      std::atexit([]() {
        if (initialized) {
          Kokkos::finalize();
          delete Init::mpi_initialization;
          initialized = false;
        }
      });
    }
  }

  Init::Init(const std::string parameter_file) : Init(0, nullptr, parameter_file) {}

  const ConfigurationHelper Init::get_configuration_helper() const
  {
    ConfigurationHelper configuration_helper(argc, argv, parameter_file);
    return configuration_helper;
  }

  bool Init::is_initialized() { return initialized; }
} // namespace DiFfRG