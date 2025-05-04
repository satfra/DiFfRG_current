#include <DiFfRG/common/initialize.hh>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

// standard libraries
#include <cstdlib>

namespace DiFfRG
{
  bool Initialize::initialized = false;

  dealii::Utilities::MPI::MPI_InitFinalize *Initialize::mpi_initialization = nullptr;

  Initialize::Initialize(int argc, char *argv[], const std::string parameter_file)
      : argc(argc), argv(argv), parameter_file(parameter_file)
  {
    if (!initialized) {
      mpi_initialization = new dealii::Utilities::MPI::MPI_InitFinalize(argc, argv, 1);
      Kokkos::initialize(argc, argv);
      initialized = true;

      std::atexit([]() {
        if (initialized) {
          Kokkos::finalize();
          delete Initialize::mpi_initialization;
          initialized = false;
        }
      });
    }
  }

  Initialize::Initialize(const std::string parameter_file) : argc(0), argv(nullptr), parameter_file(parameter_file)
  {
    if (!initialized) {
      mpi_initialization = new dealii::Utilities::MPI::MPI_InitFinalize(argc, argv, 1);
      Kokkos::initialize();
      initialized = true;

      std::atexit([]() {
        if (initialized) {
          Kokkos::finalize();
          delete mpi_initialization;
          initialized = false;
        }
      });
    }
  }

  const ConfigurationHelper Initialize::get_configuration_helper() const
  {
    ConfigurationHelper configuration_helper(argc, argv, parameter_file);
    return configuration_helper;
  }

  bool Initialize::is_initialized() { return initialized; }
} // namespace DiFfRG