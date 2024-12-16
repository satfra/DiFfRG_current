#pragma once

namespace DiFfRG
{
  /**
   * @brief A fixed size compile-time string.
   *
   * @tparam N The size of the string.
   */
  template <unsigned N> struct FixedString {
    char buf[N + 1]{};

    /**
     * @brief Construct a new Fixed String object
     *
     * @param s The string to be copied.
     * @note The string must be of size N.
     */
    constexpr FixedString(char const *s)
    {
      for (unsigned i = 0; i != N; ++i)
        buf[i] = s[i];
    }

    constexpr operator char const *() const { return buf; }

    auto operator<=>(const FixedString &) const = default;
  };

  template <unsigned N> FixedString(char const (&)[N]) -> FixedString<N - 1>;
} // namespace DiFfRG
