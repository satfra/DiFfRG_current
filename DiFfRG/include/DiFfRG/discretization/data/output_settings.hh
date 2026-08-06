#pragma once

#include <DiFfRG/common/json.hh>

#include <spdlog/common.h>

#include <cstddef>
#include <string>

namespace DiFfRG
{
  namespace Config
  {
    /** Typed configuration consumed by output sinks. */
    struct OutputSettings {
      inline static constexpr std::size_t default_max_pending_bytes = 2ull * 1024ull * 1024ull * 1024ull;

      OutputSettings() = default;
      explicit OutputSettings(const JSONValue &json);

      bool asynchronous = true;
      bool write_vtk = true;
      bool write_hdf5 = true;
      unsigned int max_pending_frames = 2;
      std::size_t max_pending_bytes = default_max_pending_bytes;
      unsigned int subdivisions = 1;
      std::size_t log_queue_size = 8192;
      spdlog::level::level_enum log_level = spdlog::level::info;
      double Lambda = -1.;
      std::string configuration_json = "{}";
      std::string configuration_log = "{}";
    };
  } // namespace Config

  // Compatibility alias for applications written before typed settings were grouped under Config.
  using OutputSettings = Config::OutputSettings;
} // namespace DiFfRG
