#include <DiFfRG/common/init.hh>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

// standard libraries
#include <cstdlib>

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
      // deal.II's MPI_InitFinalize initializes (and, in its destructor, finalizes) Kokkos via
      // InitializeLibrary::Kokkos since deal.II 9.6. Initializing Kokkos a second time here would
      // abort with "Kokkos::initialize() has already been called", so we let deal.II own the
      // Kokkos lifecycle and only construct/destroy the MPI_InitFinalize object ourselves.
      mpi_initialization = new dealii::Utilities::MPI::MPI_InitFinalize(
          argc, argv, tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));
      initialized = true;

      std::atexit([]() {
        if (initialized) {
          // ~MPI_InitFinalize() finalizes Kokkos (and MPI) on the main thread.
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