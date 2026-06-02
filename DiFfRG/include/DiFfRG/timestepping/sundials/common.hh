#pragma once

#include <sundials/sundials_context.h>
#include <sundials/sundials_types.h>

#include <exception>
#include <stdexcept>
#include <string>

namespace DiFfRG::sundials
{
  class SolveFailure : public std::runtime_error
  {
  public:
    SolveFailure(const int flag_, const char *function)
        : std::runtime_error(std::string(function) + " failed with SUNDIALS flag " + std::to_string(flag_)),
          flag_value(flag_)
    {}

    int flag() const noexcept { return flag_value; }

  private:
    int flag_value;
  };

  inline void check_flag(const int flag, const char *function)
  {
    if (flag < 0)
      throw std::runtime_error(std::string(function) + " failed with SUNDIALS flag " + std::to_string(flag));
  }

  inline void check_solve_flag(const int flag, const char *function)
  {
    if (flag < 0) throw SolveFailure(flag, function);
  }

  class Context
  {
  public:
    Context()
    {
      check_flag(SUNContext_Create(SUN_COMM_NULL, &context), "SUNContext_Create");
    }

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    Context(Context &&other) noexcept : context(other.context) { other.context = nullptr; }

    Context &operator=(Context &&other) noexcept
    {
      if (this != &other) {
        reset();
        context = other.context;
        other.context = nullptr;
      }
      return *this;
    }

    ~Context() { reset(); }

    SUNContext get() const { return context; }

  private:
    void reset() noexcept
    {
      if (context != nullptr) {
        SUNContext_Free(&context);
        context = nullptr;
      }
    }

    SUNContext context = nullptr;
  };

  inline int callback_failed(std::exception_ptr &pending_exception)
  {
    try {
      throw;
    } catch (...) {
      pending_exception = std::current_exception();
      return -1;
    }
  }

  inline void rethrow_pending(std::exception_ptr &pending_exception)
  {
    if (pending_exception) {
      auto exception = pending_exception;
      pending_exception = nullptr;
      std::rethrow_exception(exception);
    }
  }
} // namespace DiFfRG::sundials
