#pragma once

#include <DiFfRG/discretization/data/output_path.hh>
#include <DiFfRG/discretization/data/output_settings.hh>

#include <spdlog/logger.h>

#include <memory>
#include <string>
#include <utility>

namespace spdlog
{
  namespace details
  {
    class thread_pool;
  }
} // namespace spdlog

namespace DiFfRG
{
  /** Copyable, thread-safe handle to a run-owned logger. */
  class LogPort
  {
  public:
    LogPort() = default;

    template <typename... Args> void info(spdlog::format_string_t<Args...> format, Args &&...args) const
    {
      if (logger) logger->info(format, std::forward<Args>(args)...);
    }

    template <typename... Args> void warn(spdlog::format_string_t<Args...> format, Args &&...args) const
    {
      if (logger) logger->warn(format, std::forward<Args>(args)...);
    }

    template <typename... Args> void error(spdlog::format_string_t<Args...> format, Args &&...args) const
    {
      if (logger) logger->error(format, std::forward<Args>(args)...);
    }

    template <typename... Args> void debug(spdlog::format_string_t<Args...> format, Args &&...args) const
    {
      if (logger) logger->debug(format, std::forward<Args>(args)...);
    }

    void info(const std::string &message) const
    {
      if (logger) logger->info(message);
    }
    void error(const std::string &message) const
    {
      if (logger) logger->error(message);
    }
    void flush() const
    {
      if (logger) logger->flush();
    }
    explicit operator bool() const noexcept { return static_cast<bool>(logger); }

  private:
    explicit LogPort(std::shared_ptr<spdlog::logger> logger) : logger(std::move(logger)) {}
    std::shared_ptr<spdlog::logger> logger;
    friend class RunLogger;
  };

  namespace internal
  {
    template <typename Tag, typename Value> class LegacyDefaultLogPortArgument
    {
    public:
      [[deprecated("Pass output.log_port() or an intentional LogPort{}")]] LegacyDefaultLogPortArgument(
          const Value &value)
          : referenced_value(value)
      {
      }

      const Value &value() const noexcept { return referenced_value; }

    private:
      const Value &referenced_value;
    };

    template <typename Tag>
    [[deprecated("Pass output.log_port() or an intentional LogPort{}")]] LogPort legacy_default_log_port()
    {
      return LogPort{};
    }
  } // namespace internal

  /** Sink layout of a RunLogger: which file it writes to and whether it also echoes to the console. */
  struct RunLoggerOptions {
    /** Inserted between the run name and ".log", e.g. "_quadrature" for "output_quadrature.log". */
    std::string file_suffix;
    /** spdlog logger name, written to the "[%n]" field of the file records. */
    std::string logger_name = "run";
    /** Whether records are echoed to stdout. Side-channel logs keep this off so they do not
     * interleave with the timestepper progress report. */
    bool console = true;
  };

  /** Owns a private bounded asynchronous logger for one simulation run. */
  class RunLogger
  {
  public:
    RunLogger() = default;
    RunLogger(const OutputPath &path, const Config::OutputSettings &settings, bool active,
              RunLoggerOptions options = {});
    ~RunLogger() noexcept;

    RunLogger(const RunLogger &) = delete;
    RunLogger &operator=(const RunLogger &) = delete;
    RunLogger(RunLogger &&) noexcept = default;
    RunLogger &operator=(RunLogger &&) noexcept = default;

    LogPort port() const { return LogPort(logger); }
    void finish();

  private:
    /** Flushes the file sink on a timer, so that a run log stays readable while the job runs and survives a kill.
     * Defined in the translation unit; held by shared_ptr so that RunLogger stays movable. */
    struct PeriodicFlusher;

    std::shared_ptr<spdlog::details::thread_pool> thread_pool;
    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<PeriodicFlusher> flusher;
  };
} // namespace DiFfRG
