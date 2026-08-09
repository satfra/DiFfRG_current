#include <DiFfRG/common/run_logger.hh>

#include <DiFfRG/common/utils.hh>

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace DiFfRG
{
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
