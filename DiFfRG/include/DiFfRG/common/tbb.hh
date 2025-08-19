#pragma once

#include <DiFfRG/common/kokkos.hh>

#include <tbb/tbb.h>

#include <cstdlib>

namespace DiFfRG
{
  namespace internal
  {
    template <int startDim, int stopDim, int dim, typename NT, typename FUN>
    NT TBBReductionHelper(const device::array<size_t, dim> &grid_size, const FUN &functor,
                          const device::array<size_t, dim> &idx)
    {
      static_assert(stopDim > startDim, "stopDim must be greater than startDim");

      if constexpr (stopDim - startDim == 3) {
        return tbb::parallel_reduce(
            tbb::blocked_range3d<size_t, size_t, size_t>(0, grid_size[startDim], 0, grid_size[startDim + 1], 0,
                                                         grid_size[startDim + 2]),
            NT(0),
            [&](const tbb::blocked_range3d<size_t, size_t, size_t> &r, NT value) -> NT {
              auto l_idx = idx;
              for (size_t i = r.pages().begin(); i != r.pages().end(); ++i) {
                l_idx[startDim] = i;
                for (size_t j = r.rows().begin(); j != r.rows().end(); ++j) {
                  l_idx[startDim + 1] = j;
                  for (size_t k = r.cols().begin(); k != r.cols().end(); ++k) {
                    l_idx[startDim + 2] = k;
                    value += functor(l_idx);
                  }
                }
              }
              return value;
            },
            std::plus<NT>());
      } else if constexpr (stopDim - startDim == 2) {
        return tbb::parallel_reduce(
            tbb::blocked_range2d<size_t, size_t>(0, grid_size[startDim], 0, grid_size[startDim + 1]), NT(0),
            [&](const tbb::blocked_range2d<size_t, size_t> &r, NT value) -> NT {
              auto l_idx = idx;
              for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
                l_idx[startDim] = i;
                for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
                  l_idx[startDim + 1] = j;
                  value += functor(l_idx);
                }
              }
              return value;
            },
            std::plus<NT>());
      } else if constexpr (stopDim - startDim == 1) {
        return tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, grid_size[startDim]), NT(0),
            [&](const tbb::blocked_range<size_t> &r, NT value) -> NT {
              auto l_idx = idx;
              for (size_t i = r.begin(); i != r.end(); ++i) {
                l_idx[startDim] = i;
                value += functor(l_idx);
              }
              return value;
            },
            std::plus<NT>());
      } else {
        return tbb::parallel_reduce(
            tbb::blocked_range3d<size_t, size_t, size_t>(0, grid_size[startDim], 0, grid_size[startDim + 1], 0,
                                                         grid_size[startDim + 2]),
            NT(0),
            [&](const tbb::blocked_range3d<size_t, size_t, size_t> &r, NT value) -> NT {
              auto l_idx = idx;
              for (size_t i = r.pages().begin(); i != r.pages().end(); ++i) {
                l_idx[startDim] = i;
                for (size_t j = r.rows().begin(); j != r.rows().end(); ++j) {
                  l_idx[startDim + 1] = j;
                  for (size_t k = r.cols().begin(); k != r.cols().end(); ++k) {
                    l_idx[startDim + 2] = k;
                    value += TBBReductionHelper<startDim + 3, stopDim, dim, NT, FUN>(grid_size, functor, l_idx);
                  }
                }
              }
              return value;
            },
            std::plus<NT>());
      }
    }
  } // namespace internal

  template <int dim, typename NT, typename FUN>
  NT TBBReduction(const device::array<size_t, dim> &grid_size, const FUN &functor)
  {
    device::array<size_t, dim> idx{{}};
    return internal::TBBReductionHelper<0, dim, dim, NT, FUN>(grid_size, functor, idx);
  }
} // namespace DiFfRG