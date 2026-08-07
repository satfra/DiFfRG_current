#include <DiFfRG/common/run_logger.hh>

#include <DiFfRG/common/utils.hh>

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <vector>

namespace DiFfRG
{
  RunLogger::RunLogger(const OutputPath &path, const Config::OutputSettings &settings, const bool active)
  {
    if (!active) return;

    const auto top_folder = path.root().string();
    create_folder(top_folder);
    const auto queue_size = std::max<std::size_t>(256, settings.log_queue_size);
    thread_pool = std::make_shared<spdlog::details::thread_pool>(queue_size, 1);

    std::vector<spdlog::sink_ptr> sinks;
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console->set_pattern("[%v]");
    sinks.push_back(console);
    sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.run_file(".log").string(), true));

    logger = std::make_shared<spdlog::async_logger>("run", sinks.begin(), sinks.end(), thread_pool,
                                                    spdlog::async_overflow_policy::block);
    logger->set_level(settings.log_level);
  }

  RunLogger::~RunLogger() noexcept
  {
    try {
      finish();
    } catch (...) {
    }
  }

  void RunLogger::finish()
  {
    if (logger) logger->flush();
    logger.reset();
    thread_pool.reset();
  }
} // namespace DiFfRG
