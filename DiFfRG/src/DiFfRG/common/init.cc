#include <DiFfRG/common/init.hh>

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

// standard libraries
#include <cstdlib>

namespace DiFfRG
{
  bool Init::initialized = false;

  dealii::Utilities::MPI::MPI_InitFinalize *Init::mpi_initialization = nullptr;

  Init::Init(int argc, char *argv[], const std::string parameter_file)
      : argc(argc), argv(argv), parameter_file(parameter_file)
  {
    if (!initialized) {
      mpi_initialization = new dealii::Utilities::MPI::MPI_InitFinalize(argc, argv, 1);
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

  Init::Init(const std::string parameter_file) : argc(0), argv(nullptr), parameter_file(parameter_file)
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

  const ConfigurationHelper Init::get_configuration_helper() const
  {
    ConfigurationHelper configuration_helper(argc, argv, parameter_file);
    return configuration_helper;
  }

  bool Init::is_initialized() { return initialized; }
} // namespace DiFfRG