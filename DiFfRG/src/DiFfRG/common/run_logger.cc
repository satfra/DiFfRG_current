#include <DiFfRG/common/run_logger.hh>

#include <DiFfRG/common/utils.hh>

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace DiFfRG
{
  namespace
  {
    /**
     * @brief Process-wide registry of log file sinks, keyed by the resolved path.
     *
     * Several RunLoggers legitimately target the same file: a QuadratureProvider opens the run log from within the
     * integrator constructors, long before the OutputSession that later writes into it. Two independent sinks on one
     * path would each keep their own file offset and overwrite each other, so they share a single sink, which
     * serializes their records behind its own mutex.
     *
     * A path is truncated the first time it is opened in a process and appended to afterwards, so a second session
     * writing the same file does not discard what the first one recorded.
     */
    spdlog::sink_ptr shared_file_sink(const std::filesystem::path &file)
    {
      static std::mutex mutex;
      static std::map<std::string, std::weak_ptr<spdlog::sinks::basic_file_sink_mt>> sinks;

      const auto key = std::filesystem::weakly_canonical(file).string();

      const std::lock_guard<std::mutex> lock(mutex);
      const auto it = sinks.find(key);
      if (it != sinks.end())
        if (auto live = it->second.lock()) return live;

      // A known key that no longer resolves was opened earlier in this process and must not be truncated again.
      const bool truncate = it == sinks.end();
      auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file.string(), truncate);
      sinks[key] = sink;
      return sink;
    }
  } // namespace

  struct RunLogger::PeriodicFlusher {
    PeriodicFlusher(std::shared_ptr<spdlog::logger> logger, const std::chrono::milliseconds interval)
        : logger(std::move(logger)), interval(interval), worker([this] { run(); })
    {
    }

    ~PeriodicFlusher() noexcept { stop(); }

    PeriodicFlusher(const PeriodicFlusher &) = delete;
    PeriodicFlusher &operator=(const PeriodicFlusher &) = delete;

    void stop() noexcept
    {
      {
        const std::lock_guard<std::mutex> lock(mutex);
        stopped = true;
      }
      condition.notify_all();
      if (worker.joinable()) worker.join();
    }

  private:
    void run()
    {
      std::unique_lock<std::mutex> lock(mutex);
      while (!condition.wait_for(lock, interval, [this] { return stopped; })) {
        lock.unlock();
        // async_logger::flush blocks until the backend has drained, which is why this runs off the simulation threads.
        try {
          logger->flush();
        } catch (...) {
        }
        lock.lock();
      }
    }

    std::shared_ptr<spdlog::logger> logger;
    std::chrono::milliseconds interval;
    std::mutex mutex;
    std::condition_variable condition;
    bool stopped = false;
    std::thread worker;
  };

  RunLogger::RunLogger(const OutputPath &path, const Config::OutputSettings &settings, const bool active,
                       RunLoggerOptions options)
  {
    if (!active) return;

    const auto top_folder = path.root().string();
    create_folder(top_folder);
    const auto queue_size = std::max<std::size_t>(256, settings.log_queue_size);
    thread_pool = std::make_shared<spdlog::details::thread_pool>(queue_size, 1);

    std::vector<spdlog::sink_ptr> sinks;
    auto level = settings.log_level;
    if (options.console) {
      // The console follows /output/verbosity: at 0 only problems are reported, at 1 the ordinary run messages, from
      // 2 upwards also the per-frame diagnostics. The file sink is never gated this way and keeps everything.
      const auto console_level = settings.verbosity <= 0   ? spdlog::level::warn
                                 : settings.verbosity == 1 ? spdlog::level::info
                                                           : spdlog::level::debug;
      auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      console->set_pattern("[%v]");
      console->set_level(console_level);
      sinks.push_back(console);
      // The logger level filters ahead of every sink, so it has to admit whatever the console asks for.
      level = std::min(level, console_level);
    }
    sinks.push_back(shared_file_sink(path.run_file(".log")));

    logger = std::make_shared<spdlog::async_logger>(options.logger_name, sinks.begin(), sinks.end(), thread_pool,
                                                    spdlog::async_overflow_policy::block);
    logger->set_level(level);

    // The file sink is only flushed on shutdown, so without this a killed job loses the tail of its log and a running
    // job shows a log file that lags by a full stdio buffer.
    if (settings.log_flush_interval > 0.) {
      const auto interval = std::chrono::milliseconds(
          std::max<long long>(100, static_cast<long long>(settings.log_flush_interval * 1000.)));
      flusher = std::make_shared<PeriodicFlusher>(logger, interval);
    }
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
    // Stop the timer before the final flush, so that the worker cannot touch the logger while it is being released.
    if (flusher) flusher->stop();
    flusher.reset();
    if (logger) logger->flush();
    logger.reset();
    thread_pool.reset();
  }
} // namespace DiFfRG
