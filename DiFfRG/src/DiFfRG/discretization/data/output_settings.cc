#include <DiFfRG/discretization/data/output_settings.hh>

#include <sstream>

namespace DiFfRG
{
  OutputSettings::OutputSettings(const JSONValue &json)
      : write_vtk(json.get_bool("/output/vtk", true)), write_hdf5(json.get_bool("/output/hdf5", true)),
        max_pending_frames(json.get_uint("/output/max_pending_frames", 2)),
        subdivisions(json.get_uint("/discretization/output_subdivisions", 1)),
        log_queue_size(json.get_uint("/output/log_queue_size", 8192)),
        log_level(static_cast<spdlog::level::level_enum>(json.get_int("/output/log_level", SPDLOG_LEVEL_INFO))),
        Lambda(json.get_double("/physical/Lambda", -1.)),
        configuration_json(json::serialize(static_cast<json::value>(json)))
  {
    std::ostringstream stream;
    json.print(stream);
    configuration_log = stream.str();
  }
} // namespace DiFfRG
