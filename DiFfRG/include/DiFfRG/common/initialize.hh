#pragma once

// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>

namespace DiFfRG
{
  class Initialize
  {
  public:
    Initialize(int argc, char *argv[], const std::string parameter_file = "parameter.json");
    Initialize(const std::string parameter_file = "parameter.json");

    const ConfigurationHelper get_configuration_helper() const;

    static bool is_initialized();

  private:
    static bool initialized;

    std::string parameter_file;
    int argc;
    char **argv;
  };
} // namespace DiFfRG