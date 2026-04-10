#pragma once

#include <DiFfRG/common/tuples.hh>
#include <tuple>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        template <typename... T> auto advection_flux_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions">>(std::tie(t...));
        }

        template <typename... T> auto flux_tie(T &&...t)
        {
          return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives">>(std::tie(t...));
        }
      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
