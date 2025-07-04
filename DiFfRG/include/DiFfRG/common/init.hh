#pragma once

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>

// external libraries
#include <deal.II/base/mpi.h>

// standard libraries
#include <memory>

namespace DiFfRG
{
  class Init
  {
  public:
    Init(int argc, char *argv[], const std::string parameter_file = "parameter.json");
    Init(const std::string parameter_file = "parameter.json");

    const ConfigurationHelper get_configuration_helper() const;

    static bool is_initialized();

  private:
    static bool initialized;

    int argc;
    char **argv;
    std::string parameter_file;
    static dealii::Utilities::MPI::MPI_InitFinalize *mpi_initialization;
  };
} // namespace DiFfRG