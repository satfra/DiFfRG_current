#pragma once

// external libraries
#include <autodiff/forward/real.hpp>

namespace DiFfRG
{
  namespace internal
  {
    template <typename NT> struct __TLITypes;

    template <> struct __TLITypes<float> {
      using ReturnType = float;
    };

    template <> struct __TLITypes<double> {
      using ReturnType = float;
    };

    template <> struct __TLITypes<autodiff::real> {
      using ReturnType = autodiff::real;
    };
  } // namespace internal
} // namespace DiFfRG