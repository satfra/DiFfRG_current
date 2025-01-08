#pragma once

namespace DiFfRG
{
  template <typename T>
  concept MeshIsRectangular = requires(T x) { T::is_rectangular; } && T::is_rectangular;
} // namespace DiFfRG